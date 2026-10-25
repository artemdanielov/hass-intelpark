"""Камеры Intelpark (HLS-поток + снимок)."""
from __future__ import annotations

import asyncio
import logging
import time

from homeassistant.components.camera import Camera as HACamera, CameraEntityFeature
from homeassistant.components.camera.helper import async_get_stream_image
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .api import Camera, Gate, IntelparkError
from .const import DOMAIN, TOKEN_REFRESH_MARGIN
from .coordinator import IntelparkConfigEntry, IntelparkCoordinator
from .entity import gate_device_info, object_device_info, strip_number_sign

_LOGGER = logging.getLogger(__name__)

# Leave time for a stream frame within HA's default image request timeout.
GATE_SNAPSHOT_TIMEOUT = 3
GATE_SNAPSHOT_COOLDOWN = 60


async def async_setup_entry(
    hass: HomeAssistant,
    entry: IntelparkConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator = entry.runtime_data
    entities: list[HACamera] = [
        IntelparkCamera(coordinator, cam) for cam in coordinator.data.cameras.values()
    ]
    entities += [
        IntelparkGateCamera(coordinator, gate)
        for gate in coordinator.data.gates
        if gate.live_url
    ]
    async_add_entities(entities)


class IntelparkCamera(HACamera):
    """Одна камера двора."""

    _attr_has_entity_name = True
    _attr_supported_features = CameraEntityFeature.STREAM

    def __init__(self, coordinator: IntelparkCoordinator, camera: Camera) -> None:
        super().__init__()
        self.coordinator = coordinator
        self._cam_id = camera.cam_id
        self._attr_unique_id = f"{DOMAIN}_camera_{camera.cam_id}"
        self._attr_name = strip_number_sign(camera.name) or camera.cam_id
        self._attr_device_info = object_device_info(
            coordinator.config_entry.entry_id, camera.object_name
        )

    @property
    def _camera(self) -> Camera | None:
        return self.coordinator.data.cameras.get(self._cam_id)

    async def _ensure_fresh(self) -> Camera | None:
        cam = self._camera
        if cam is None:
            return None
        if cam.stream_token_valid_till and cam.stream_token_valid_till <= (
            time.time() + TOKEN_REFRESH_MARGIN
        ):
            refreshed = await self.coordinator.async_refresh_camera(self._cam_id)
            if refreshed is not None:
                return refreshed
        return cam

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            self.coordinator.async_add_listener(self.async_write_ha_state)
        )

    async def stream_source(self) -> str | None:
        cam = await self._ensure_fresh()
        return cam.stream_url() if cam else None

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        cam = self._camera
        if cam is None or not cam.snapshot:
            return None
        try:
            return await self.coordinator.client.async_fetch_snapshot(cam.snapshot)
        except IntelparkError as err:
            _LOGGER.warning(
                "Не удалось получить снимок камеры %s: %s", self._cam_id, err
            )
            return None

    @property
    def available(self) -> bool:
        return (
            super().available
            and self.coordinator.last_update_success
            and self._camera is not None
        )


class IntelparkGateCamera(HACamera):
    """Камера шлагбаума (`get_gates`): live-поток без токена."""

    _attr_has_entity_name = True
    _attr_supported_features = CameraEntityFeature.STREAM
    _attr_icon = "mdi:cctv"

    def __init__(self, coordinator: IntelparkCoordinator, gate: Gate) -> None:
        super().__init__()
        self.coordinator = coordinator
        self._gid = gate.gid
        self._attr_unique_id = f"{DOMAIN}_gatecam_{gate.gid}"
        self._attr_name = "Камера"
        self._attr_device_info = gate_device_info(
            coordinator.config_entry.entry_id, gate
        )
        self._snapshot_lock = asyncio.Lock()
        self._snapshot_failures = 0
        self._snapshot_open_until = 0.0
        self._image_inflight: dict[tuple[int | None, int | None], list] = {}
        self._image_tasks: set[asyncio.Task] = set()
        self._image_removing = False

    @property
    def _gate(self) -> Gate | None:
        return next(
            (g for g in self.coordinator.data.gates if g.gid == self._gid), None
        )

    @property
    def use_stream_for_stills(self) -> bool:
        # Prefer snapshots without starting HLS/go2rtc just for a preview.
        gate = self._gate
        return gate is None or not bool(gate.snapshot_image)

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            self.coordinator.async_add_listener(self.async_write_ha_state)
        )

    async def stream_source(self) -> str | None:
        gate = self._gate
        return gate.live_stream_url() if gate and gate.live_url else None

    async def async_will_remove_from_hass(self) -> None:
        """Stop shared image work before the entity is unloaded."""
        self._image_removing = True
        tasks = tuple(self._image_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._image_inflight.clear()
        await super().async_will_remove_from_hass()

    def _image_done(self, key, entry, task) -> None:
        # Identity checks protect a newer request with the same dimensions.
        if self._image_inflight.get(key) is entry:
            del self._image_inflight[key]
        self._image_tasks.discard(task)
        if not task.cancelled():
            task.exception()  # Retrieve even when every waiter has left.

    async def async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        """Share concurrent work only; never retain a completed image."""
        if self._image_removing:
            return None
        key = (width, height)
        entry = self._image_inflight.get(key)
        if entry is None or entry[0].done():
            task = asyncio.create_task(self._async_camera_image(width, height))
            entry = [task, 0]
            self._image_inflight[key] = entry
            self._image_tasks.add(task)
            task.add_done_callback(
                lambda done: self._image_done(key, entry, done)
            )
        task = entry[0]
        entry[1] += 1
        try:
            # A cancelled consumer must not cancel the other consumers' work.
            return await asyncio.shield(task)
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                if self._image_inflight.get(key) is entry:
                    del self._image_inflight[key]
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def _async_camera_image(
        self, width: int | None = None, height: int | None = None
    ) -> bytes | None:
        snapshot_started = time.monotonic()
        snapshot_failure = "circuit_open"
        # Serialize snapshot decisions across dimensions, not stream fallback.
        # This also prevents duplicate half-open probes and stale completions
        # overwriting a newer circuit state. No lock spans the fallback helper.
        async with self._snapshot_lock:
            gate = self._gate
            if self._image_removing or gate is None or not gate.snapshot_image:
                return None
            if time.monotonic() >= self._snapshot_open_until:
                probe = self._snapshot_open_until > 0
                snapshot_failure = "empty"
                try:
                    async with asyncio.timeout(GATE_SNAPSHOT_TIMEOUT):
                        image = await self.coordinator.client.async_fetch_snapshot(
                            gate.snapshot_image
                        )
                    if image:
                        self._snapshot_failures = 0
                        self._snapshot_open_until = 0.0
                        if probe:
                            _LOGGER.warning(
                                "Gate image gate=%s snapshot=probe-recovered",
                                self.entity_id,
                            )
                        return image
                except (IntelparkError, TimeoutError) as err:
                    # No upstream exception text or URL in diagnostic records.
                    snapshot_failure = (
                        "timeout" if isinstance(err, TimeoutError) else "intelpark_error"
                    )
                self._snapshot_failures = min(self._snapshot_failures + 1, 2)
                if probe or self._snapshot_failures >= 2:
                    self._snapshot_open_until = (
                        time.monotonic() + GATE_SNAPSHOT_COOLDOWN
                    )
                    _LOGGER.warning(
                        "Gate image gate=%s snapshot=circuit-open cooldown_s=%s "
                        "reason=%s probe=%s",
                        self.entity_id, GATE_SNAPSHOT_COOLDOWN,
                        snapshot_failure, probe,
                    )
        snapshot_ms = (time.monotonic() - snapshot_started) * 1000
        if self._gate is None:
            return None
        request = getattr(self, "_snapshot_diag_request", 0) + 1
        self._snapshot_diag_request = request
        fallback_started = time.monotonic()
        result = "exception"
        try:
            image = await async_get_stream_image(
                self, width=width, height=height, wait_for_next_keyframe=False
            )
            result = "image" if image else "none"
            return image
        except asyncio.CancelledError:
            result = "cancelled"
            raise
        finally:
            finished = time.monotonic()
            # Defer the pair until the outcome is known. Each outcome category
            # gets its own cooldown, so a new failure is not hidden by success.
            # At most 4 snapshot reasons x 4 fallback outcomes; no URL/ID keys.
            diagnostics = getattr(self, "_snapshot_diag_outcomes", None)
            if diagnostics is None:
                diagnostics = self._snapshot_diag_outcomes = {}
            key = (snapshot_failure, result)
            last, suppressed = diagnostics.get(key, (float("-inf"), 0))
            if finished - last >= 60:
                diagnostics[key] = (finished, 0)
                _LOGGER.warning(
                    "Gate image gate=%s request=%s snapshot_failure=%s "
                    "snapshot_ms=%.1f fallback=start deferred=true suppressed=%s",
                    self.entity_id, request, snapshot_failure, snapshot_ms, suppressed,
                )
                _LOGGER.warning(
                    "Gate image gate=%s request=%s fallback=result result=%s "
                    "fallback_ms=%.1f",
                    self.entity_id, request, result,
                    (finished - fallback_started) * 1000,
                )
            else:
                diagnostics[key] = (last, suppressed + 1)

    @property
    def extra_state_attributes(self) -> dict:
        gate = self._gate
        if gate is None:
            return {}
        return {"live_cam": gate.live_cam}

    @property
    def available(self) -> bool:
        gate = self._gate
        return super().available and gate is not None and bool(gate.live_url)
