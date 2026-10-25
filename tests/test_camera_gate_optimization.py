"""Регрессии защиты снимков, совместных запросов и безопасной диагностики."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.intelpark import camera
from custom_components.intelpark.api import IntelparkError
from .test_camera_gate import _gate, _setup


@pytest.fixture
async def image_context(hass):
    """Настоящая сущность HA; подменяем только сеть и локальные часы камеры."""
    entry = await _setup(hass, [_gate()])
    coordinator = entry.runtime_data
    entity = camera.IntelparkGateCamera(coordinator, coordinator.data.gates[0])
    entity.hass = hass
    entity.entity_id = "camera.test_gate"
    clock = SimpleNamespace(now=100.0)
    with (
        patch.object(camera, "time", SimpleNamespace(monotonic=lambda: clock.now)),
        patch.object(coordinator.client, "async_fetch_snapshot",
                     new_callable=AsyncMock, return_value=b"snapshot") as fetch,
        patch.object(camera, "async_get_stream_image",
                     new_callable=AsyncMock, return_value=b"frame") as fallback,
    ):
        yield SimpleNamespace(entity=entity, coordinator=coordinator, clock=clock,
                              fetch=fetch, fallback=fallback)
        await entity.async_will_remove_from_hass()


async def _open_circuit(ctx):
    ctx.fetch.return_value = None
    await ctx.entity.async_camera_image()
    await ctx.entity.async_camera_image()
    assert ctx.entity._snapshot_failures == 2
    assert ctx.entity._snapshot_open_until == 160


def _block(mock, result=b"snapshot", error=None):
    """События задают порядок запросов без реальных задержек."""
    started, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def work(*args, **kwargs):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        if error is not None:
            raise error
        return result

    mock.side_effect = work
    return started, release, cancelled


async def _start_pair(ctx, target, first=(320, 180), second=(320, 180)):
    started, release, cancelled = _block(target)
    a = asyncio.create_task(ctx.entity.async_camera_image(*first))
    await started.wait()
    b = asyncio.create_task(ctx.entity.async_camera_image(*second))
    await asyncio.sleep(0)  # Второй потребитель присоединяется к общей задаче.
    return a, b, release, cancelled


async def test_healthy_snapshot_has_no_fallback_diagnostics(image_context, caplog):
    ctx = image_context
    assert await ctx.entity.async_camera_image() == b"snapshot"
    ctx.fallback.assert_not_awaited()
    assert not [record for record in caplog.records if record.name == camera.__name__]


async def test_real_timeout_records_failure_and_cancels_fetch(image_context, caplog):
    ctx = image_context
    _, _, cancelled = _block(ctx.fetch)
    with patch.object(camera, "GATE_SNAPSHOT_TIMEOUT", 0):
        assert await ctx.entity.async_camera_image() == b"frame"
    assert cancelled.is_set()
    assert ctx.entity._snapshot_failures == 1
    assert "snapshot_failure=timeout" in caplog.text
    assert "result=image" in caplog.text


async def test_circuit_threshold_and_cooldown(image_context):
    ctx = image_context
    assert camera.GATE_SNAPSHOT_COOLDOWN == 60
    await _open_circuit(ctx)
    ctx.clock.now = 159.999
    assert await ctx.entity.async_camera_image() == b"frame"
    assert ctx.fetch.await_count == 2
    assert ctx.fallback.await_count == 3


async def test_success_resets_consecutive_failures(image_context):
    ctx = image_context
    ctx.fetch.side_effect = [None, b"healthy", None, None]
    for _ in range(5):
        await ctx.entity.async_camera_image()
    assert ctx.fetch.await_count == 4
    assert ctx.entity._snapshot_failures == 2


async def test_probe_recovery_resets_circuit(image_context, caplog):
    ctx = image_context
    await _open_circuit(ctx)
    ctx.clock.now = 160
    ctx.fetch.return_value = b"healthy"
    assert await ctx.entity.async_camera_image() == b"healthy"
    assert ctx.entity._snapshot_failures == 0
    assert ctx.entity._snapshot_open_until == 0
    assert await ctx.entity.async_camera_image() == b"healthy"
    assert ctx.fetch.await_count == 4
    assert "snapshot=probe-recovered" in caplog.text


async def test_failed_probe_reopens_for_full_cooldown(image_context):
    ctx = image_context
    await _open_circuit(ctx)
    ctx.clock.now = 160
    await ctx.entity.async_camera_image()
    assert ctx.entity._snapshot_open_until == 220
    ctx.clock.now = 219
    await ctx.entity.async_camera_image()
    assert ctx.fetch.await_count == 3
    ctx.clock.now = 220
    await ctx.entity.async_camera_image()
    assert ctx.fetch.await_count == 4


async def test_circuits_are_per_entity(image_context):
    ctx = image_context
    await _open_circuit(ctx)
    other_gate = _gate(gid="other")
    ctx.coordinator.data.gates.append(other_gate)
    other = camera.IntelparkGateCamera(ctx.coordinator, other_gate)
    ctx.fetch.return_value = b"healthy"
    assert await other.async_camera_image() == b"healthy"
    assert await ctx.entity.async_camera_image() == b"frame"
    assert other._snapshot_failures == 0


@pytest.mark.parametrize("phase", ["snapshot", "fallback"])
async def test_identical_requests_share_only_inflight_work(image_context, phase):
    ctx = image_context
    if phase == "fallback":
        ctx.fetch.return_value = None
    target = ctx.fetch if phase == "snapshot" else ctx.fallback
    a, b, release, _ = await _start_pair(ctx, target)
    release.set()
    assert await asyncio.gather(a, b) == [b"snapshot", b"snapshot"]
    assert target.await_count == 1
    assert not ctx.entity._image_inflight
    assert not ctx.entity._image_tasks


@pytest.mark.parametrize("phase", ["snapshot", "fallback"])
async def test_no_completed_or_stale_image_cache(image_context, phase):
    ctx = image_context
    if phase == "fallback":
        ctx.fetch.return_value = None
    target = ctx.fetch if phase == "snapshot" else ctx.fallback
    target.side_effect = [b"first", b"second"]
    assert await ctx.entity.async_camera_image() == b"first"
    assert await ctx.entity.async_camera_image() == b"second"
    assert target.await_count == 2


@pytest.mark.parametrize("phase", ["snapshot", "fallback"])
async def test_cancelling_one_waiter_preserves_other(image_context, phase):
    ctx = image_context
    if phase == "fallback":
        ctx.fetch.return_value = None
    target = ctx.fetch if phase == "snapshot" else ctx.fallback
    a, b, release, cancelled = await _start_pair(ctx, target)
    a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await a
    assert not cancelled.is_set()
    release.set()
    assert await b == b"snapshot"
    assert target.await_count == 1


@pytest.mark.parametrize("phase", ["snapshot", "fallback"])
async def test_last_cancelled_waiter_cleans_work_and_allows_retry(image_context, phase):
    ctx = image_context
    if phase == "fallback":
        ctx.fetch.return_value = None
    target = ctx.fetch if phase == "snapshot" else ctx.fallback
    a, b, _, cancelled = await _start_pair(ctx, target)
    a.cancel()
    b.cancel()
    results = await asyncio.gather(a, b, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert cancelled.is_set()
    assert not ctx.entity._image_inflight
    assert not ctx.entity._image_tasks
    assert ctx.entity._snapshot_failures == (0 if phase == "snapshot" else 1)
    target.side_effect = None
    assert await ctx.entity.async_camera_image() == target.return_value


@pytest.mark.parametrize("dimensions", [((320, 180), (180, 320)),
                                        ((None, 180), (320, 180)),
                                        ((320, None), (320, 180))])
async def test_dimensions_are_exact_and_independent(image_context, dimensions):
    ctx = image_context
    ctx.fetch.return_value = None
    await asyncio.gather(*(ctx.entity.async_camera_image(*size) for size in dimensions))
    assert ctx.fetch.await_count == 2
    assert {(call.kwargs["width"], call.kwargs["height"])
            for call in ctx.fallback.await_args_list} == set(dimensions)
    assert all(call.kwargs["wait_for_next_keyframe"] is False
               for call in ctx.fallback.await_args_list)


async def test_shared_exception_propagates_and_retry_is_fresh(image_context):
    ctx = image_context
    error = ValueError("synthetic failure")
    started, release, _ = _block(ctx.fetch, error=error)
    a = asyncio.create_task(ctx.entity.async_camera_image())
    await started.wait()
    b = asyncio.create_task(ctx.entity.async_camera_image())
    await asyncio.sleep(0)
    release.set()
    assert await asyncio.gather(a, b, return_exceptions=True) == [error, error]
    assert ctx.entity._snapshot_failures == 0
    ctx.fetch.side_effect = None
    assert await ctx.entity.async_camera_image() == b"snapshot"
    ctx.fallback.assert_not_awaited()


@pytest.mark.parametrize("phase", ["snapshot", "fallback"])
async def test_removal_cancels_work_and_rejects_new_requests(image_context, phase):
    ctx = image_context
    if phase == "fallback":
        ctx.fetch.return_value = None
    target = ctx.fetch if phase == "snapshot" else ctx.fallback
    a, b, _, cancelled = await _start_pair(ctx, target)
    with patch.object(camera.HACamera, "async_will_remove_from_hass",
                      new_callable=AsyncMock) as base_remove:
        await ctx.entity.async_will_remove_from_hass()
        base_remove.assert_awaited_once()
    results = await asyncio.gather(a, b, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert cancelled.is_set()
    assert not ctx.entity._image_inflight
    assert not ctx.entity._image_tasks
    assert await ctx.entity.async_camera_image() is None
    assert target.await_count == 1


async def test_half_open_probe_is_serialized_across_dimensions(image_context):
    ctx = image_context
    await _open_circuit(ctx)
    ctx.clock.now = 160
    a, b, release, _ = await _start_pair(ctx, ctx.fetch, (1, 2), (2, 1))
    await asyncio.sleep(0)
    assert ctx.fetch.await_count == 3
    release.set()
    assert await asyncio.gather(a, b) == [b"snapshot", b"snapshot"]
    assert ctx.fetch.await_count == 4


async def test_failed_probe_is_not_duplicated_across_dimensions(image_context):
    ctx = image_context
    await _open_circuit(ctx)
    ctx.clock.now = 160
    await asyncio.gather(ctx.entity.async_camera_image(1, 2),
                         ctx.entity.async_camera_image(2, 1))
    assert ctx.fetch.await_count == 3
    assert ctx.entity._snapshot_open_until == 220


async def test_cancelled_probe_does_not_reset_or_extend_circuit(image_context):
    ctx = image_context
    await _open_circuit(ctx)
    ctx.clock.now = 160
    started, _, cancelled = _block(ctx.fetch)
    task = asyncio.create_task(ctx.entity.async_camera_image())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert ctx.entity._snapshot_failures == 2
    assert ctx.entity._snapshot_open_until == 160
    ctx.fetch.side_effect = None
    await ctx.entity.async_camera_image()
    await ctx.entity.async_camera_image()
    assert ctx.fetch.await_count == 4


async def test_cancellation_while_waiting_for_snapshot_lock(image_context):
    ctx = image_context
    a, b, release, _ = await _start_pair(ctx, ctx.fetch, (1, 2), (2, 1))
    await asyncio.sleep(0)
    b.cancel()
    with pytest.raises(asyncio.CancelledError):
        await b
    release.set()
    assert await a == b"snapshot"
    assert ctx.fetch.await_count == 1
    assert ctx.entity._snapshot_failures == 0


async def test_old_done_callback_does_not_remove_new_request(image_context):
    entity = image_context.entity
    old = asyncio.create_task(asyncio.sleep(0))
    await old
    replacement = [old, 1]
    entity._image_inflight[(1, 2)] = replacement
    entity._image_done((1, 2), [old, 0], old)
    assert entity._image_inflight[(1, 2)] is replacement
    entity._image_inflight.clear()


async def test_open_circuit_does_not_mask_removed_gate(image_context):
    ctx = image_context
    await _open_circuit(ctx)
    ctx.coordinator.data.gates.clear()
    assert await ctx.entity.async_camera_image() is None
    assert ctx.fallback.await_count == 2


async def test_next_request_uses_updated_snapshot_url(image_context):
    ctx = image_context
    await ctx.entity.async_camera_image()
    ctx.coordinator.data.gates[0].snapshot_image = "https://example.invalid/new-snapshot"
    await ctx.entity.async_camera_image()
    ctx.fetch.assert_awaited_with("https://example.invalid/new-snapshot")


async def test_diagnostics_sanitize_and_bound_repeated_errors(image_context, caplog):
    ctx = image_context
    private_marker = "DO_NOT_LOG_SYNTHETIC_MARKER"
    ctx.coordinator.data.gates[0].snapshot_image = (
        "https://example.invalid/image?credential=" + private_marker
    )
    ctx.fetch.side_effect = IntelparkError(private_marker)
    await _open_circuit(ctx)
    for _ in range(20):
        await ctx.entity.async_camera_image()
    records = [record for record in caplog.records if record.name == camera.__name__]
    text = " ".join(record.getMessage() for record in records)
    assert private_marker not in text
    assert "https://" not in text
    assert all(record.exc_info is None for record in records)
    assert len(records) == 5
    assert "snapshot=circuit-open" in text
    assert "snapshot_failure=intelpark_error" in text
    assert "snapshot_failure=circuit_open" in text
    assert "snapshot_ms=" in text and "fallback_ms=" in text
    assert "gate=camera.test_gate" in text
    assert len(ctx.entity._snapshot_diag_outcomes) <= 16


@pytest.mark.parametrize("result", ["image", "none", "exception", "cancelled"])
async def test_diagnostics_record_fallback_outcomes(image_context, caplog, result):
    ctx = image_context
    ctx.fetch.return_value = None
    if result == "none":
        ctx.fallback.return_value = None
    elif result in {"exception", "cancelled"}:
        error = ValueError if result == "exception" else asyncio.CancelledError
        ctx.fallback.side_effect = error("DO_NOT_LOG_SYNTHETIC_MARKER")
        with pytest.raises(error):
            await ctx.entity.async_camera_image()
    if result in {"image", "none"}:
        await ctx.entity.async_camera_image()
    assert f"result={result}" in caplog.text
    assert "DO_NOT_LOG_SYNTHETIC_MARKER" not in caplog.text


async def test_new_failure_category_is_not_suppressed_by_success(image_context, caplog):
    ctx = image_context
    ctx.fetch.return_value = None
    await ctx.entity.async_camera_image()
    ctx.fallback.return_value = None
    await ctx.entity.async_camera_image()
    assert "result=image" in caplog.text
    assert "result=none" in caplog.text


async def test_suppressed_summary_is_reported_after_cooldown(image_context, caplog):
    ctx = image_context
    await _open_circuit(ctx)
    for _ in range(3):
        await ctx.entity.async_camera_image()
    ctx.clock.now = 159
    await ctx.entity.async_camera_image()
    ctx.clock.now = 160
    await ctx.entity.async_camera_image()
    ctx.clock.now = 161
    await ctx.entity.async_camera_image()
    assert "suppressed=3" in caplog.text


async def test_stream_fallback_does_not_hold_snapshot_lock(image_context):
    ctx = image_context
    ctx.fetch.side_effect = [None, b"healthy"]
    started, release, _ = _block(ctx.fallback, result=b"frame")
    a = asyncio.create_task(ctx.entity.async_camera_image(1, 2))
    await started.wait()
    assert await ctx.entity.async_camera_image(2, 1) == b"healthy"
    release.set()
    assert await a == b"frame"
