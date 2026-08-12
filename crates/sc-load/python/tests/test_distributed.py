from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from sc_compress.anndata import write_scc
import sc_load
import sc_load.distributed as distributed_module

pytestmark = pytest.mark.skipif(
    not hasattr(sc_load._core, "_attach_shared"),
    reason="distributed shared rings require Linux with lock-free 64-bit atomics",
)


def _write_matrix(tmp_path: Path, values: np.ndarray) -> Path:
    adata = ad.AnnData(
        X=values,
        obs=pd.DataFrame(index=[f"c{row}" for row in range(values.shape[0])]),
        var=pd.DataFrame(index=[f"g{column}" for column in range(values.shape[1])]),
    )
    return write_scc(adata, tmp_path / "distributed.scc", store="dir")


def _consume_rank(
    iterator: sc_load.DistributedIterator,
    queue: mp.Queue[tuple[int, list[list[float]], bool, bool]],
) -> None:
    with iterator:
        values = iterator.read()
        queue.put(
            (
                iterator.rank,
                values.tolist(),
                bool(values.flags.c_contiguous),
                bool(values.flags.writeable),
            )
        )


def _attach_rank_and_exit(
    iterator: sc_load.DistributedIterator,
    attached: Any,
) -> None:
    iterator.__enter__()
    attached.set()
    os._exit(0)


def _close_inherited_handles(
    iterator: sc_load.DistributedIterator,
    server: Any,
) -> None:
    server.cancel()
    try:
        server.duplicate_fd()
    except sc_load._core.CoreError:
        pass
    else:
        raise AssertionError("inherited producer unexpectedly duplicated its descriptor")
    iterator.close()


def _close_inherited_session_then_consume(
    distributed: sc_load.DistributedSession,
    iterator: sc_load.DistributedIterator,
    queue: Any,
) -> None:
    distributed.close()
    try:
        queue.put(("ok", iterator.read().tolist()))
    except BaseException as error:
        queue.put((type(error).__name__, str(error)))


def _plan(tmp_path: Path, rows: int = 8) -> tuple[sc_load.Plan, np.ndarray]:
    values = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    dataset = sc_load.register(_write_matrix(tmp_path, values))
    plan = sc_load.compile(
        dataset,
        range(rows),
        batch_size=2,
        prefetch_step=4,
    )
    return plan, values


def test_distributed_rank_iterators_preserve_round_robin_batches(tmp_path: Path) -> None:
    plan, values = _plan(tmp_path)
    config = sc_load.SessionConfig(worker_count=2, io_mode="blocking")
    with plan.open_distributed(2, config) as distributed:
        rank0 = distributed.rank(0)
        rank1 = distributed.rank(1)
        assert rank0.shape == (4, 3)
        assert rank1.shape == (4, 3)
        assert rank0.batch_count == rank1.batch_count == 2
        np.testing.assert_array_equal(rank0.read(), values[[0, 1, 4, 5]])
        np.testing.assert_array_equal(rank1.read(), values[[2, 3, 6, 7]])
        distributed.wait(timeout=5)
        assert distributed.state == "finished"
        assert distributed.finished


def test_distributed_zero_copy_view_is_read_only_and_leased(tmp_path: Path) -> None:
    plan, values = _plan(tmp_path, rows=2)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    with plan.open_distributed(1, config) as distributed:
        iterator = distributed.rank(0, copy=False)
        batch = iterator.next_batch()
        assert batch is not None
        assert not batch.flags.owndata
        assert not batch.flags.writeable
        assert batch.base is not None
        np.testing.assert_array_equal(batch, values)
        with pytest.raises(ValueError):
            batch[0, 0] = -1
        with pytest.raises(ValueError):
            batch.setflags(write=True)
        del batch
        assert iterator.next_batch() is None
        distributed.wait(timeout=5)


@pytest.mark.parametrize(
    "dtype",
    [np.int16, np.int32, np.uint16, np.uint32, np.float32, np.float64],
)
def test_distributed_copy_returns_compact_numpy_owned_batches(
    tmp_path: Path,
    dtype: type[np.generic],
) -> None:
    values = np.arange(12, dtype=np.dtype(dtype)).reshape(4, 3)
    dataset = sc_load.register(_write_matrix(tmp_path, values))
    plan = sc_load.compile(
        dataset,
        range(4),
        output=sc_load.OutputSpec(3, dtype),
        batch_size=2,
        prefetch_step=2,
    )
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    with plan.open_distributed(1, config) as distributed:
        iterator = distributed.rank(0)
        batch = iterator.next_batch()
        assert batch is not None
        assert batch.flags.owndata
        assert batch.flags.c_contiguous
        assert batch.flags.writeable
        assert batch.base is None
        assert batch.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(batch, values[:2])
        batch[0, 0] = 42
        assert batch[0, 0] == 42
        tail = iterator.read()
        assert tail.flags.owndata
        assert tail.flags.c_contiguous
        assert tail.flags.writeable
        assert tail.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(tail, values[2:])
        assert iterator.read().shape == (0, 3)
        assert iterator.batches_yielded == 2
        assert iterator.rows_yielded == 4
        assert iterator.exhausted
        assert iterator.closed
        distributed.wait(timeout=5)


def test_distributed_copy_supports_zero_width_output(tmp_path: Path) -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    dataset = sc_load.register(_write_matrix(tmp_path, values)).with_feature_map([None] * 3)
    plan = sc_load.compile(
        dataset,
        range(2),
        output=sc_load.OutputSpec(0, np.float32),
        batch_size=2,
        prefetch_step=2,
    )
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    with plan.open_distributed(1, config) as distributed:
        iterator = distributed.rank(0)
        batch = iterator.next_batch()
        assert batch is not None
        assert batch.shape == (2, 0)
        assert batch.flags.owndata
        assert batch.flags.c_contiguous
        assert batch.flags.writeable
        assert iterator.next_batch() is None
        distributed.wait(timeout=5)
    with plan.open_distributed(1, config) as distributed:
        output = distributed.rank(0).read()
        assert output.shape == (2, 0)
        assert output.flags.owndata
        assert output.flags.c_contiguous
        assert output.flags.writeable
        distributed.wait(timeout=5)


def test_distributed_validation_and_cancellation(tmp_path: Path) -> None:
    plan, _ = _plan(tmp_path)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    with pytest.raises(ValueError, match="world_size"):
        plan.open_distributed(0, config)
    with pytest.raises(sc_load.ResourceLimitError):
        plan.open_distributed(1, config, maximum_control_bytes=1)

    distributed = plan.open_distributed(1, config)
    first = distributed.rank(0)
    with pytest.raises(ValueError, match="already has an iterator"):
        distributed.rank(0)
    assert first.next_batch() is not None
    first.close()
    with pytest.raises(sc_load.CancelledError):
        distributed.wait(timeout=5)
    distributed.close()


def test_distributed_ranks_validates_even_when_no_ranks_remain(tmp_path: Path) -> None:
    plan, _ = _plan(tmp_path, rows=2)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    distributed = plan.open_distributed(1, config)
    iterator = distributed.ranks()[0]
    with pytest.raises(TypeError, match="copy"):
        distributed.ranks(copy=1)  # type: ignore[arg-type]
    assert iterator.read().shape == (2, 3)
    distributed.wait(timeout=5)
    distributed.close()
    with pytest.raises(ValueError, match="closed"):
        distributed.ranks()


def test_distributed_ranks_rolls_back_partial_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _ = _plan(tmp_path)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    distributed = plan.open_distributed(3, config)
    original = sc_load.DistributedSession._new_iterator
    created: list[sc_load.DistributedIterator] = []

    def fail_second(
        session: sc_load.DistributedSession,
        rank: int,
        copy: bool,
    ) -> sc_load.DistributedIterator:
        if rank == 1:
            raise OSError("injected descriptor exhaustion")
        iterator = original(session, rank, copy)
        created.append(iterator)
        return iterator

    monkeypatch.setattr(sc_load.DistributedSession, "_new_iterator", fail_second)
    with pytest.raises(OSError, match="descriptor exhaustion"):
        distributed.ranks()
    assert distributed.info()["issued_ranks"] == ()
    assert distributed._handles == []
    assert len(created) == 1
    assert created[0].closed
    distributed.close()


def test_distributed_iterator_state_restore_closes_fd_on_failure() -> None:
    read_fd, write_fd = os.pipe()

    class _TransferredFd:
        def detach(self) -> int:
            return read_fd

    iterator = sc_load.DistributedIterator.__new__(sc_load.DistributedIterator)
    try:
        with pytest.raises(TypeError, match="metadata"):
            iterator.__setstate__((_TransferredFd(), 0, None, True))  # type: ignore[arg-type]
        with pytest.raises(OSError):
            os.fstat(read_fd)
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass
        os.close(write_fd)


def test_distributed_iterator_close_releases_every_resource_after_error() -> None:
    read_fd, write_fd = os.pipe()
    metadata = distributed_module._DistributedMetadata(1, 2, 3, 2, 1, "f32")
    iterator = sc_load.DistributedIterator(read_fd, 0, metadata, copy=True)

    class _FailingClient:
        def close(self) -> None:
            raise OSError("injected client close failure")

    iterator._client = _FailingClient()  # type: ignore[assignment]
    iterator._owner_pid = os.getpid()
    try:
        with pytest.raises(OSError, match="injected client close failure"):
            iterator.close()
        assert iterator.closed
        assert iterator._client is None
        assert iterator._descriptor == -1
        with pytest.raises(OSError):
            os.fstat(read_fd)
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass
        os.close(write_fd)


def test_distributed_iterator_rejects_mismatched_descriptor_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd, write_fd = os.pipe()
    metadata = distributed_module._DistributedMetadata(1, 2, 3, 2, 1, "f32")
    iterator = sc_load.DistributedIterator(read_fd, 0, metadata, copy=True)

    class _MismatchedClient:
        rank = 0
        world_size = 1
        n_rows = 2
        n_cols = 4
        batch_size = 2
        batch_count = 1
        dtype = "f32"
        closed = False

        def close(self) -> None:
            self.closed = True

    client = _MismatchedClient()
    monkeypatch.setattr(sc_load._core, "_attach_shared", lambda _fd, _rank: client)
    try:
        with pytest.raises(RuntimeError, match="metadata does not match"):
            iterator.__enter__()
        assert client.closed
        assert iterator.closed
        assert iterator._descriptor == -1
        with pytest.raises(OSError):
            os.fstat(read_fd)
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass
        os.close(write_fd)


def test_distributed_iterator_serializes_concurrent_first_consumers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, values = _plan(tmp_path, rows=8)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    original = sc_load._core._attach_shared
    first_entered = threading.Event()
    release_first = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    def delayed_attach(fd: int, rank: int) -> Any:
        nonlocal call_count
        with call_lock:
            call_index = call_count
            call_count += 1
        if call_index == 0:
            first_entered.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("first attach was not released")
        return original(fd, rank)

    monkeypatch.setattr(sc_load._core, "_attach_shared", delayed_attach)
    with plan.open_distributed(1, config) as distributed:
        iterator = distributed.rank(0)
        batches: list[np.ndarray[Any, Any]] = []
        errors: list[BaseException] = []

        def consume() -> None:
            try:
                batch = iterator.next_batch()
                assert batch is not None
                batches.append(batch)
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=consume)
        second = threading.Thread(target=consume)
        first.start()
        assert first_entered.wait(timeout=5)
        second.start()
        time.sleep(0.05)
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)
        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert call_count == 1
        combined = np.concatenate(batches)
        combined = combined[np.argsort(combined[:, 0])]
        np.testing.assert_array_equal(combined, values[:4])
        np.testing.assert_array_equal(iterator.read(), values[4:])
        distributed.wait(timeout=5)


def test_distributed_iterator_serializes_read_and_next_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, values = _plan(tmp_path, rows=16)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    original = distributed_module._call_core
    read_entered = threading.Event()
    allow_read = threading.Event()

    with plan.open_distributed(1, config) as distributed:
        iterator = distributed.rank(0)
        iterator.__enter__()

        def delayed_call(function: Any, *args: Any, **kwargs: Any) -> Any:
            if getattr(function, "__name__", "") == "read":
                read_entered.set()
                if not allow_read.wait(timeout=5):
                    raise TimeoutError("read was not released")
            return original(function, *args, **kwargs)

        monkeypatch.setattr(distributed_module, "_call_core", delayed_call)
        results: dict[str, Any] = {}
        errors: list[BaseException] = []

        def read_all() -> None:
            try:
                results["read"] = iterator.read()
            except BaseException as error:
                errors.append(error)

        def next_one() -> None:
            try:
                results["next"] = iterator.next_batch()
            except BaseException as error:
                errors.append(error)

        reader = threading.Thread(target=read_all)
        consumer = threading.Thread(target=next_one)
        reader.start()
        assert read_entered.wait(timeout=5)
        consumer.start()
        time.sleep(0.05)
        allow_read.set()
        reader.join(timeout=5)
        consumer.join(timeout=5)
        assert not reader.is_alive()
        assert not consumer.is_alive()
        assert errors == []
        np.testing.assert_array_equal(results["read"], values)
        assert results["next"] is None
        distributed.wait(timeout=5)


def test_distributed_zero_copy_reports_held_ring_slot(tmp_path: Path) -> None:
    values = np.arange(18, dtype=np.float32).reshape(6, 3)
    dataset = sc_load.register(_write_matrix(tmp_path, values))
    plan = sc_load.compile(dataset, range(6), batch_size=2, prefetch_step=2)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    with plan.open_distributed(1, config) as distributed:
        iterator = distributed.rank(0, copy=False)
        first = iterator.next_batch()
        assert first is not None
        second = iterator.next_batch()
        assert second is not None
        with pytest.raises(sc_load.InvalidInputError, match="still holds logical batch"):
            iterator.next_batch()
        del first
        del second


def test_distributed_iterator_close_wakes_a_concurrent_next(tmp_path: Path) -> None:
    plan, _ = _plan(tmp_path, rows=16)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    distributed = plan.open_distributed(2, config)
    iterator = distributed.rank(0)
    assert iterator.next_batch() is not None
    assert iterator.next_batch() is not None
    assert iterator.next_batch() is not None

    errors: list[BaseException] = []

    def wait_for_next() -> None:
        try:
            iterator.next_batch()
        except BaseException as error:
            errors.append(error)

    waiter = threading.Thread(target=wait_for_next)
    waiter.start()
    time.sleep(0.05)
    assert waiter.is_alive()
    iterator.close()
    waiter.join(timeout=5)
    assert not waiter.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], sc_load.CancelledError)
    with pytest.raises(sc_load.CancelledError):
        distributed.wait(timeout=5)
    distributed.close()


@pytest.mark.skipif(not hasattr(mp, "get_context"), reason="multiprocessing unavailable")
def test_distributed_iterator_transfers_to_spawned_processes(tmp_path: Path) -> None:
    plan, values = _plan(tmp_path)
    config = sc_load.SessionConfig(worker_count=2, io_mode="blocking")
    context = mp.get_context("spawn")
    queue = context.Queue()
    with plan.open_distributed(2, config) as distributed:
        handles = distributed.ranks(copy=True)
        processes = [
            context.Process(target=_consume_rank, args=(handle, queue)) for handle in handles
        ]
        for process in processes:
            process.start()
        results = [queue.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        distributed.wait(timeout=5)

    by_rank = {
        rank: (np.asarray(rows), contiguous, writeable)
        for rank, rows, contiguous, writeable in results
    }
    np.testing.assert_array_equal(by_rank[0][0], values[[0, 1, 4, 5]])
    np.testing.assert_array_equal(by_rank[1][0], values[[2, 3, 6, 7]])
    assert by_rank[0][1:] == (True, True)
    assert by_rank[1][1:] == (True, True)


@pytest.mark.skipif(not hasattr(mp, "get_context"), reason="multiprocessing unavailable")
def test_distributed_detects_a_rank_process_that_exits_without_cleanup(tmp_path: Path) -> None:
    plan, _ = _plan(tmp_path)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    context = mp.get_context("spawn")
    attached = context.Event()
    with plan.open_distributed(1, config) as distributed:
        process = context.Process(
            target=_attach_rank_and_exit,
            args=(distributed.rank(0), attached),
        )
        process.start()
        assert attached.wait(timeout=10)
        process.join(timeout=10)
        assert process.exitcode == 0
        with pytest.raises(sc_load.CancelledError):
            distributed.wait(timeout=5)


@pytest.mark.skipif(not hasattr(mp, "get_context"), reason="multiprocessing unavailable")
def test_distributed_reclaims_dead_owner_for_an_empty_rank(tmp_path: Path) -> None:
    plan, values = _plan(tmp_path, rows=2)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    context = mp.get_context("spawn")
    attached = context.Event()
    with plan.open_distributed(2, config) as distributed:
        rank0 = distributed.rank(0)
        empty_rank = distributed.rank(1)
        process = context.Process(
            target=_attach_rank_and_exit,
            args=(empty_rank, attached),
        )
        process.start()
        assert attached.wait(timeout=10)
        process.join(timeout=10)
        assert process.exitcode == 0
        assert empty_rank.read().shape == (0, 3)
        np.testing.assert_array_equal(rank0.read(), values)
        distributed.wait(timeout=5)


@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="fork start method unavailable",
)
def test_closing_an_attached_iterator_after_fork_does_not_cancel_parent(
    tmp_path: Path,
) -> None:
    plan, values = _plan(tmp_path)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    context = mp.get_context("fork")
    with plan.open_distributed(1, config) as distributed:
        iterator = distributed.rank(0)
        iterator.__enter__()
        process = context.Process(
            target=_close_inherited_handles,
            args=(iterator, distributed._runner.inner),
        )
        process.start()
        process.join(timeout=5)
        assert not process.is_alive()
        assert process.exitcode == 0
        np.testing.assert_array_equal(iterator.read(), values)
        distributed.wait(timeout=5)


@pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(),
    reason="fork start method unavailable",
)
def test_closing_inherited_session_does_not_close_child_rank(tmp_path: Path) -> None:
    plan, values = _plan(tmp_path)
    config = sc_load.SessionConfig(worker_count=1, io_mode="blocking")
    context = mp.get_context("fork")
    queue = context.Queue()
    with plan.open_distributed(1, config) as distributed:
        iterator = distributed.rank(0)
        process = context.Process(
            target=_close_inherited_session_then_consume,
            args=(distributed, iterator, queue),
        )
        process.start()
        result = queue.get(timeout=10)
        process.join(timeout=10)
        assert not process.is_alive()
        assert process.exitcode == 0
        assert result[0] == "ok", result
        np.testing.assert_array_equal(np.asarray(result[1]), values)
        distributed.wait(timeout=5)


def test_empty_distributed_plan_finishes_for_every_rank() -> None:
    plan = sc_load.compile(
        [],
        [],
        output=sc_load.OutputSpec(3, np.float32),
        batch_size=2,
        prefetch_step=2,
    )
    with plan.open_distributed(
        3,
        sc_load.SessionConfig(worker_count=1, io_mode="blocking"),
    ) as distributed:
        for iterator in distributed.ranks():
            result = iterator.read()
            assert result.shape == (0, 3)
        distributed.wait(timeout=5)
