from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from scdata import (
    CellIndexPlan,
    DataBankConfig,
    ScheduledAccessConfig,
    ScheduledPrefetchConfig,
    ScDataBank,
    launch_all,
)
from scdata.databank import _PrefetchPlan, _config_to_rust


DATASETS = [
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/00476f9f-ebc1-4b72-b541-32f912ce36ea/full.zarr.zip", "X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0087cde2-967d-4f7c-8e6e-40e4c9ad1891/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/01209dce-3575-4bed-b1df-129f57fbc031/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0129dbd9-a7d3-4f6b-96b9-1da155a93748/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/019c7af2-c827-4454-9970-44d5e39ce068/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/01ad3cd7-3929-4654-84c0-6db05bd5fd59/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/01b84709-139c-4485-98a9-6e14c58fbbf6/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/01c93cf6-b695-4e30-a26e-121ae8b16a9e/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/01ff5cf0-730f-4ddc-b1be-7b407211f544/full.zarr.zip", "X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/024581e3-5375-4e33-8060-d8448694f556/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/02792605-4760-4023-82ad-40fc4458a5db/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/030faa69-ff79-4d85-8630-7c874a114c19/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/03d38670-1444-4001-bc53-9936e61d9b20/full.zarr.zip", "X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0436a180-cb44-47ba-8ffa-807b7a468469/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/06ef6b36-6c9b-4e10-8a94-d0baf274276e/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/07760522-707a-4a1c-8891-dbd1226d6b27/full.zarr.zip", "X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/077b0429-0f47-48e0-879a-39eaae531d42/full.zarr.zip", "X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/07b1d7c8-5c2e-42f7-9246-26f746cd6013/full.zarr.zip", "X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/07dd8c34-4625-480a-8ee4-273a06c3082b/full.zarr.zip", "X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/090da8ea-46e8-40df-bffc-1f78e1538d27/full.zarr.zip", "X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0920bcb8-4b3a-4e9d-a353-56f529fd3b32/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/093d3bfe-6f0f-4ac0-a7a1-829f94d0a49f/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/095940cb-7422-4510-96e2-cbafd961eb88/full.zarr.zip", "X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/09b518f9-da64-44cc-aec8-70a89d55611f/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0a2d7e87-c3c0-4ed2-86df-ae18811fcc16/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0a8e3443-c3e2-4918-84db-0495657d9175/full.zarr.zip", "X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0ae6f031-2f9c-4247-8b26-db320d6efd32/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0b4a15a7-4e9e-4555-9733-2423e5c66469/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0ba636a1-4754-4786-a8be-7ab3cf760fd6/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0bae7ebf-eb54-46a6-be9a-3461cecefa4c/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0bc7235a-ae5a-479d-a487-510435377e55/full.zarr.zip", "raw/X"),
    ("/mnt/shared-storage-user/dnacoding/wangzhongqi/Data/cellxgene/Homo_sapiens/0bf30b02-db97-4560-b9cd-0fe3588892af/full.zarr.zip", "raw/X"),
]

BATCH_SIZE = 128
GENES = 4096
OUTPUT_DTYPE = "u16"
MEMORY_GIB = 256
FILL_WORKERS = 32
FAST_FUSED_WORKERS = 128
IO_WORKERS = 48
FAST_LOAD_SCHEDULER_WORKERS = 128
FAST_REQUEST_PREFETCH_BLOCKS = 8192
PREFETCH_STEP = 512
ACCESS_PREFETCH_STEP = 512
DECODE_AHEAD_STEPS = 512
READY_AHEAD_STEPS = 512
WARMUP_BATCHES = 64
COALESCE_MAX_GAP_BYTES = 1024 * 1024
COALESCE_MAX_WASTE_RATIO = 0.90
COALESCE_MAX_MERGED_LEN = 8 * 1024 * 1024
STREAM_PROGRESS_UPDATE_EVERY = 1024


def progress_kwargs(desc: str, unit: str) -> dict[str, object]:
    return {
        "desc": desc,
        "unit": unit,
        "dynamic_ncols": True,
        "mininterval": 0.5,
    }


def rate(value: int, seconds: float) -> float:
    return value / seconds if seconds > 0 else 0.0


def data_checksum(data) -> int:
    if not data.size:
        return 0
    checksum = int(data.flat[0])
    if data.size > 1:
        checksum ^= int(data.flat[-1]) << 1
    return checksum & ((1 << 64) - 1)


def make_bank() -> ScDataBank:
    cfg = DataBankConfig.make(
        backend="threaded",
        io__threaded__num_workers=IO_WORKERS,
        decode__num_workers=1,
        access__cpu__num_workers=1,
        access__scheduler_shards=1,
        fill__num_workers=FILL_WORKERS,
        access__cache_capacity_bytes=MEMORY_GIB * 1024**3 * 3 // 4,
        access__memory_budget_bytes=MEMORY_GIB * 1024**3,
        fast__enabled=True,
        fast__fused_workers=FAST_FUSED_WORKERS,
        fast__request_prefetch_blocks=FAST_REQUEST_PREFETCH_BLOCKS,
        fast__memory_budget_bytes=MEMORY_GIB * 1024**3,
        fast__response_queue_bytes_soft_limit=MEMORY_GIB * 1024**3 // 2,
        fast__response_queue_bytes_hard_limit=MEMORY_GIB * 1024**3 * 3 // 4,
        fast__load__scheduler_workers=FAST_LOAD_SCHEDULER_WORKERS,
        fast__load__io_workers=IO_WORKERS,
        fast__load__coalesce__max_gap_bytes=COALESCE_MAX_GAP_BYTES,
        fast__load__coalesce__max_waste_ratio=COALESCE_MAX_WASTE_RATIO,
        fast__load__coalesce__max_merged_len=COALESCE_MAX_MERGED_LEN,
    )
    return ScDataBank(cfg)


def make_sequential_plan(counts: list[int]) -> CellIndexPlan:
    total = sum(counts)
    dataset_index = np.empty(total, dtype=np.uint16)
    cell_dtype = np.uint32 if max(counts, default=0) <= np.iinfo(np.uint32).max else np.uint64
    cell_index = np.empty(total, dtype=cell_dtype)

    offset = 0
    for dataset_idx, count in enumerate(
        tqdm(counts, **progress_kwargs("build sequential plan", "dataset"))
    ):
        stop = offset + count
        dataset_index[offset:stop] = dataset_idx
        cell_index[offset:stop] = np.arange(count, dtype=cell_dtype)
        offset = stop
    return CellIndexPlan(dataset_index, cell_index, BATCH_SIZE)


def prefetch_indexed_fast(bank: ScDataBank, ids: list, plan: CellIndexPlan, genes, config):
    rust_plan = _PrefetchPlan.indexed(plan.dataset_index, plan.cell_index, plan.batch_size)
    rust_config = _config_to_rust(config)
    return bank._core().prefetch_cells(
        [dataset_id._rust for dataset_id in ids],
        rust_plan,
        OUTPUT_DTYPE,
        list(genes),
        None,
        rust_config,
    )


def main() -> None:
    print(
        "[0/7] config: sequential, no payload/cache, dtype=u16, "
        "window=512, fill=32, fused=128, load=128, io=48",
        flush=True,
    )

    print("[1/7] opening 32 datasets", flush=True)
    datasets = []
    for path, matrix in tqdm(DATASETS, **progress_kwargs("open datasets", "dataset")):
        datasets.append(launch_all(Path(path))[matrix])

    print("[2/7] reading dataset cell counts", flush=True)
    counts = []
    for ds in tqdm(datasets, **progress_kwargs("count cells", "dataset")):
        counts.append(int(ds.num_cells))
    print(f"      total cells: {sum(counts):,}", flush=True)

    print("[3/7] building sequential CellIndexPlan", flush=True)
    plan = make_sequential_plan(counts)

    print("[4/7] creating ScDataBank and registering datasets", flush=True)
    with tqdm(total=1, **progress_kwargs("create bank", "step")) as progress:
        bank = make_bank()
        progress.update()
    try:
        ids = []
        for ds in tqdm(datasets, **progress_kwargs("register datasets", "dataset")):
            ids.append(bank.register(ds))

        print("[5/7] preparing genes and prefetch config", flush=True)
        with tqdm(total=2, **progress_kwargs("prepare request", "step")) as progress:
            genes = bank.dataset_genes(ids[0])[:GENES]
            progress.update()
            config = ScheduledPrefetchConfig(
                prefetch_step=PREFETCH_STEP,
                access=ScheduledAccessConfig(
                    prefetch_step=ACCESS_PREFETCH_STEP,
                    decode_ahead_steps=DECODE_AHEAD_STEPS,
                    ready_ahead_steps=READY_AHEAD_STEPS,
                ),
                projected_sparse_data_strategy="selected_only",
                fast_mode="force",
            )
            progress.update()
        print(
            f"      genes: {len(genes):,}, batch size: {BATCH_SIZE}, "
            f"warmup batches: {WARMUP_BATCHES}",
            flush=True,
        )

        seen_batches = batches = cells = bytes_out = checksum = 0
        start = None
        print("[6/7] streaming batches", flush=True)
        stream = prefetch_indexed_fast(bank, ids, plan, genes, config)
        pending_progress = 0
        with tqdm(
            total=plan.num_batches,
            **progress_kwargs("sequential batches", "batch"),
        ) as progress:
            for batch_cells, batch_data, _num_genes in stream:
                seen_batches += 1
                pending_progress += 1
                if pending_progress >= STREAM_PROGRESS_UPDATE_EVERY:
                    progress.update(pending_progress)
                    pending_progress = 0
                if seen_batches <= WARMUP_BATCHES:
                    continue
                if start is None:
                    start = time.perf_counter()
                batches += 1
                cells += len(batch_cells)
                bytes_out += batch_data.nbytes
                if batch_data.size:
                    checksum = (checksum + data_checksum(batch_data)) & ((1 << 64) - 1)
            if pending_progress:
                progress.update(pending_progress)
        seconds = time.perf_counter() - start if start is not None else 0.0
    finally:
        bank.close()

    print("[7/7] result", flush=True)
    with tqdm(total=1, **progress_kwargs("print result", "step")) as progress:
        print(f"datasets: {len(DATASETS)}")
        print("order: sequential")
        print(f"dtype: {OUTPUT_DTYPE}")
        print(f"warmup batches: {WARMUP_BATCHES}")
        print(f"seen batches: {seen_batches:,}")
        print(f"cells: {cells:,}")
        print(f"batches: {batches:,}")
        print(f"seconds: {seconds:.3f}")
        print(f"cell/s: {rate(cells, seconds):.2f}")
        print(f"batch/s: {rate(batches, seconds):.2f}")
        print(f"GB/s: {rate(bytes_out, seconds) / 1e9:.2f}")
        print(f"checksum: {checksum}")
        progress.update()


if __name__ == "__main__":
    main()
