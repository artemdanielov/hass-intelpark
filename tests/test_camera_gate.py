"""Камеры шлагбаумов (`get_gates`): live-поток без токена."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.exceptions import HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.intelpark.api import AccountPlace, Gate, IntelparkError
from custom_components.intelpark.const import CONF_HASH, CONF_PHONE, DOMAIN


def _gate(gid="1460", name="№1", live_url="https://v/1460-abc", live_cam="0",
          snapshot_image="https://s/img"):
    return Gate(gid, name, 743, "Дом", live_url, live_cam, snapshot_image)


async def _setup(hass, gates):
    entry = MockConfigEntry(
        domain=DOMAIN, data={CONF_PHONE: "7900", CONF_HASH: "H"}, unique_id="7900"
    )
    entry.add_to_hass(hass)
    with patch.multiple(
        "custom_components.intelpark.coordinator.IntelparkApiClient",
        async_get_objects=AsyncMock(return_value=[AccountPlace(743, "Дом", "1")]),
        async_get_gates=AsyncMock(return_value=gates),
        async_get_cameras=AsyncMock(return_value=[]),
        async_get_passes=AsyncMock(return_value=[]),
        async_get_notifications=AsyncMock(return_value=[]),
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return entry


async def test_gate_camera_created_when_live_url_present(hass):
    entry = await _setup(hass, [_gate()])
    ids = hass.states.async_entity_ids("camera")
    assert ids == ["camera.shlagbaum_1_kamera"]
    assert hass.states.get("camera.shlagbaum_1_kamera") is not None


async def test_gate_camera_not_created_without_live_url(hass):
    await _setup(hass, [_gate(live_url="")])
    assert hass.states.async_entity_ids("camera") == []


async def test_stream_source_no_token(hass):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    assert await ent.stream_source() == "https://v/1460-abc/mono.m3u8"


async def test_gate_camera_unavailable_when_gate_gone(hass):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    assert ent.available is True

    coordinator.data.gates.clear()
    assert ent.available is False
    assert await ent.stream_source() is None


@pytest.mark.parametrize("snapshot_error", [IntelparkError("HTTP 502"), TimeoutError()])
async def test_camera_image_stream_fallback_on_error(hass, snapshot_error):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    with (
        patch.object(coordinator.client, "async_fetch_snapshot", side_effect=snapshot_error) as mock_fetch,
        patch("custom_components.intelpark.camera.async_get_stream_image",
              new_callable=AsyncMock, return_value=b"stream frame") as mock_stream,
    ):
        assert await ent.async_camera_image(width=640, height=360) == b"stream frame"
    mock_fetch.assert_awaited_once_with("https://s/img")
    mock_stream.assert_awaited_once_with(
        ent, width=640, height=360, wait_for_next_keyframe=False
    )
    assert ent.use_stream_for_stills is False


@pytest.mark.parametrize("snapshot", [b"", None])
async def test_camera_image_stream_fallback_on_empty_snapshot(hass, snapshot):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    with (
        patch.object(coordinator.client, "async_fetch_snapshot", return_value=snapshot),
        patch("custom_components.intelpark.camera.async_get_stream_image",
              new_callable=AsyncMock, return_value=b"stream frame") as mock_stream,
    ):
        assert await ent.async_camera_image() == b"stream frame"
    mock_stream.assert_awaited_once_with(
        ent, width=None, height=None, wait_for_next_keyframe=False
    )


async def test_camera_image_bounds_snapshot_wait(hass):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera
    from custom_components.intelpark import camera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    snapshot_cancelled = asyncio.Event()

    async def stalled_snapshot(url):
        try:
            await asyncio.Event().wait()
        finally:
            snapshot_cancelled.set()

    with (
        patch.object(coordinator.client, "async_fetch_snapshot", side_effect=stalled_snapshot),
        patch.object(camera, "GATE_SNAPSHOT_TIMEOUT", 0),
        patch.object(camera, "async_get_stream_image",
                     new_callable=AsyncMock, return_value=b"stream frame") as mock_stream,
    ):
        async with asyncio.timeout(1):
            assert await ent.async_camera_image() == b"stream frame"
    assert snapshot_cancelled.is_set()
    mock_stream.assert_awaited_once()
    assert camera.GATE_SNAPSHOT_TIMEOUT == 3


@pytest.mark.parametrize("snapshot_error", [asyncio.CancelledError(), ValueError("unexpected")])
async def test_camera_image_does_not_swallow_cancellation_or_unexpected_error(hass, snapshot_error):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    with (
        patch.object(coordinator.client, "async_fetch_snapshot", side_effect=snapshot_error),
        patch("custom_components.intelpark.camera.async_get_stream_image",
              new_callable=AsyncMock) as mock_stream,
        pytest.raises(type(snapshot_error)),
    ):
        await ent.async_camera_image()
    mock_stream.assert_not_awaited()


async def test_camera_image_task_cancellation_propagates(hass):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    snapshot_started = asyncio.Event()

    async def stalled_snapshot(url):
        snapshot_started.set()
        await asyncio.Event().wait()

    with (
        patch.object(coordinator.client, "async_fetch_snapshot", side_effect=stalled_snapshot),
        patch("custom_components.intelpark.camera.async_get_stream_image",
              new_callable=AsyncMock) as mock_stream,
    ):
        task = asyncio.create_task(ent.async_camera_image())
        await snapshot_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    mock_stream.assert_not_awaited()


@pytest.mark.parametrize("stream_error", [None, HomeAssistantError("stream failed"), TimeoutError(), asyncio.CancelledError()])
async def test_camera_image_preserves_stream_helper_failure(hass, stream_error):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    with (
        patch.object(coordinator.client, "async_fetch_snapshot", side_effect=IntelparkError("HTTP 502")),
        patch("custom_components.intelpark.camera.async_get_stream_image",
              new_callable=AsyncMock, return_value=None, side_effect=stream_error) as mock_stream,
    ):
        if stream_error is None:
            assert await ent.async_camera_image() is None
        else:
            with pytest.raises(type(stream_error)):
                await ent.async_camera_image()
    mock_stream.assert_awaited_once()


async def test_camera_image_fallback_uses_actual_helper_and_existing_stream(hass):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    ent.hass = hass
    ent.stream = MagicMock()
    ent.stream.async_get_image = AsyncMock(return_value=b"existing stream frame")
    with (
        patch.object(coordinator.client, "async_fetch_snapshot", side_effect=IntelparkError("HTTP 502")),
        patch.object(ent, "async_create_stream", new_callable=AsyncMock) as create_stream,
    ):
        assert await ent.async_camera_image(width=640, height=360) == b"existing stream frame"
    ent.stream.async_get_image.assert_awaited_once_with(
        width=640, height=360, wait_for_next_keyframe=False
    )
    create_stream.assert_not_awaited()


async def test_camera_image_gate_disappears_during_snapshot(hass):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])

    async def disappeared_snapshot(url):
        coordinator.data.gates.clear()
        raise IntelparkError("HTTP 502")

    with (
        patch.object(coordinator.client, "async_fetch_snapshot", side_effect=disappeared_snapshot),
        patch("custom_components.intelpark.camera.async_get_stream_image",
              new_callable=AsyncMock) as mock_stream,
    ):
        assert await ent.async_camera_image() is None
    mock_stream.assert_not_awaited()


async def test_camera_image_no_snapshot_image(hass):
    entry = await _setup(hass, [_gate(snapshot_image="")])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    assert await ent.async_camera_image() is None


async def test_use_snapshot_for_stills_when_available(hass):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    assert ent.use_stream_for_stills is False

    with (
        patch.object(coordinator.client, "async_fetch_snapshot", return_value=b"snapshot") as mock_fetch,
        patch("custom_components.intelpark.camera.async_get_stream_image",
              new_callable=AsyncMock) as mock_stream,
    ):
        assert await ent.async_camera_image() == b"snapshot"
    mock_fetch.assert_awaited_once_with("https://s/img")
    mock_stream.assert_not_awaited()
    assert await ent.stream_source() == "https://v/1460-abc/mono.m3u8"


async def test_use_stream_for_stills_without_snapshot(hass):
    entry = await _setup(hass, [_gate(snapshot_image="")])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    assert ent.use_stream_for_stills is True
    assert await ent.stream_source() == "https://v/1460-abc/mono.m3u8"


async def test_stills_source_tracks_coordinator_updates(hass):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    assert ent.use_stream_for_stills is False
    coordinator.data.gates[0].snapshot_image = ""
    assert ent.use_stream_for_stills is True
    coordinator.data.gates[0].snapshot_image = "https://s/new-img"
    assert ent.use_stream_for_stills is False
    coordinator.data.gates.clear()
    assert ent.use_stream_for_stills is True
    assert await ent.async_camera_image() is None


async def test_extra_state_attributes(hass):
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    from custom_components.intelpark.camera import IntelparkGateCamera

    ent = IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    assert ent.extra_state_attributes == {"live_cam": "0"}

    coordinator.data.gates.clear()
    assert ent.extra_state_attributes == {}
