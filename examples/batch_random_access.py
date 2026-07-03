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
SEED = 0
MEMORY_GIB = 128
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
        io__threaded__num_workers=48,
        decode__num_workers=1,
        access__cpu__num_workers=1,
        access__scheduler_shards=1,
        fill__num_workers=32,
        access__cache_capacity_bytes=MEMORY_GIB * 1024**3 * 3 // 4,
        access__memory_budget_bytes=MEMORY_GIB * 1024**3,
        fast__enabled=True,
        fast__fused_workers=96,
        fast__request_prefetch_blocks=8192,
        fast__memory_budget_bytes=MEMORY_GIB * 1024**3,
        fast__response_queue_bytes_soft_limit=MEMORY_GIB * 1024**3 // 2,
        fast__response_queue_bytes_hard_limit=MEMORY_GIB * 1024**3 * 3 // 4,
        fast__load__scheduler_workers=96,
        fast__load__io_workers=48,
        fast__load__coalesce__max_gap_bytes=16 * 1024,
        fast__load__coalesce__max_waste_ratio=0.10,
        fast__load__coalesce__max_merged_len=1024 * 1024,
    )
    return ScDataBank(cfg)


def make_random_plan(counts: list[int]) -> CellIndexPlan:
    with tqdm(total=5, **progress_kwargs("build random plan", "step")) as progress:
        counts_array = np.asarray(counts, dtype=np.int64)
        progress.update()
        offsets = np.concatenate(([0], np.cumsum(counts_array)))
        progress.update()
        order = np.arange(int(offsets[-1]), dtype=np.int64)
        progress.update()
        np.random.default_rng(SEED).shuffle(order)
        progress.update()
        dataset_index = np.searchsorted(offsets[1:], order, side="right").astype(
            np.uint16, copy=False
        )
        cell_index = (order - offsets[dataset_index]).astype(np.uint32, copy=False)
        progress.update()
    return CellIndexPlan(dataset_index, cell_index, BATCH_SIZE)


def main() -> None:
    print("[0/7] config: random order, no payload/cache, fused=96, load=96, io=48", flush=True)

    print("[1/7] opening 32 datasets", flush=True)
    datasets = []
    for path, matrix in tqdm(DATASETS, **progress_kwargs("open datasets", "dataset")):
        datasets.append(launch_all(Path(path))[matrix])

    print("[2/7] reading dataset cell counts", flush=True)
    counts = []
    for ds in tqdm(datasets, **progress_kwargs("count cells", "dataset")):
        counts.append(int(ds.num_cells))
    print(f"      total cells: {sum(counts):,}", flush=True)

    print("[3/7] building random CellIndexPlan", flush=True)
    plan = make_random_plan(counts)

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
                prefetch_step=8192,
                access=ScheduledAccessConfig(
                    prefetch_step=8192,
                    decode_ahead_steps=8192,
                    ready_ahead_steps=8192,
                ),
                projected_sparse_data_strategy="selected_only",
                fast_mode="force",
            )
            progress.update()
        print(f"      genes: {len(genes):,}, batch size: {BATCH_SIZE}", flush=True)

        batches = cells = bytes_out = checksum = 0
        print("[6/7] streaming batches", flush=True)
        start = time.perf_counter()
        stream = bank.prefetch_indexed(
            ids,
            plan,
            genes=genes,
            missing="zero",
            dtype=None,
            config=config,
        )
        pending_progress = 0
        with tqdm(total=plan.num_batches, **progress_kwargs("random batches", "batch")) as progress:
            for batch in stream:
                pending_progress += 1
                if pending_progress >= STREAM_PROGRESS_UPDATE_EVERY:
                    progress.update(pending_progress)
                    pending_progress = 0
                batches += 1
                cells += len(batch.cells)
                bytes_out += batch.data.nbytes
                if batch.data.size:
                    checksum = (checksum + data_checksum(batch.data)) & ((1 << 64) - 1)
            if pending_progress:
                progress.update(pending_progress)
        seconds = time.perf_counter() - start
    finally:
        bank.close()

    print("[7/7] result", flush=True)
    with tqdm(total=1, **progress_kwargs("print result", "step")) as progress:
        print(f"datasets: {len(DATASETS)}")
        print("order: random")
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
