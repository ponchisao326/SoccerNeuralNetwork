"""Memory-bounded streaming data access for the processed feature collection.

The processed table (``ml_match_features``) is read through a PyTorch
``IterableDataset`` that pages a MongoDB cursor instead of materialising the data
in a pandas DataFrame. This keeps peak memory flat (one cursor batch + one
shuffle buffer) and lets the same code path scale to far larger collections than
RAM would allow.

Thread/process safety
---------------------
``pymongo`` clients are not safe to share across processes. With multi-worker
``DataLoader`` parallelism each worker runs :meth:`MatchFeatureDataset.__iter__`
in its own process, so the Mongo client is created lazily *inside* ``__iter__``
and closed when the iterator is exhausted. Workers shard the row space disjointly
by ``row_idx % num_workers`` so no row is emitted twice.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pymongo import ASCENDING, MongoClient
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ml.config import NUM_CLASSES, REGRESSION_TARGETS, MongoConfig

logger = logging.getLogger(__name__)

# Categorical columns, in the fixed order the model's embedding layers expect.
CATEGORICAL_FIELDS: Tuple[str, ...] = (
    "home_team_idx",
    "away_team_idx",
    "league_idx",
    "season_idx",
)

# One training example for the multi-task model:
#   (numeric features, categorical indices, outcome label, regression targets,
#    regression mask).
# The mask is a single 0/1 scalar shared by all regression targets, since they
# all originate from the same WhoScored join and are therefore present or absent
# together (see ml.features.build_event_target_index).
Sample = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class MatchFeatureDataset(IterableDataset[Sample]):
    """Streams a temporal slice of the feature collection as tensors.

    Args:
        mongo: Mongo connection settings.
        feature_names: Ordered numeric feature columns (must match the model's
            numeric input width and the scaler's ordering).
        row_range: Half-open ``[start, end)`` interval over the global
            chronological ``row_idx``. This is how the temporal train/val/test
            split is applied: each split passes a disjoint, ordered range.
        mean: Per-feature means for standardisation (fit on the train split only).
        std: Per-feature standard deviations (zeros are replaced by 1).
        shuffle_buffer: Reservoir size for approximate shuffling. 0 streams in
            chronological order (use for validation/test for determinism).
        seed: Base RNG seed for the shuffle buffer (offset per worker).
        page_size: Mongo cursor batch size.
    """

    def __init__(
        self,
        mongo: MongoConfig,
        feature_names: Sequence[str],
        row_range: Tuple[int, int],
        mean: np.ndarray,
        std: np.ndarray,
        shuffle_buffer: int = 0,
        seed: int = 0,
        page_size: int = 1000,
    ) -> None:
        super().__init__()
        self._mongo = mongo
        self._feature_names = list(feature_names)
        self._start, self._end = row_range
        self._mean = mean.astype(np.float32)
        # Guard against division by zero for constant features.
        safe_std = std.astype(np.float32).copy()
        safe_std[safe_std == 0.0] = 1.0
        self._std = safe_std
        self._shuffle_buffer = shuffle_buffer
        self._seed = seed
        self._page_size = page_size

    def _build_filter(self, num_workers: int, worker_id: int) -> Dict[str, object]:
        """Mongo filter for this worker's disjoint shard of the row range."""
        row_filter: Dict[str, object] = {"row_idx": {"$gte": self._start, "$lt": self._end}}
        if num_workers > 1:
            # ``$mod`` shards the contiguous range across workers without overlap.
            row_filter["row_idx"] = {
                "$gte": self._start,
                "$lt": self._end,
            }
            row_filter["$expr"] = {
                "$eq": [{"$mod": ["$row_idx", num_workers]}, worker_id]
            }
        return row_filter

    def _to_sample(self, doc: Dict[str, object]) -> Sample:
        """Convert one Mongo document to the multi-task tensor tuple."""
        raw = np.array(
            [float(doc[name]) for name in self._feature_names],  # type: ignore[arg-type]
            dtype=np.float32,
        )
        standardised = (raw - self._mean) / self._std
        numeric = torch.from_numpy(standardised)
        categorical = torch.tensor(
            [int(doc[field]) for field in CATEGORICAL_FIELDS],  # type: ignore[arg-type]
            dtype=torch.long,
        )
        outcome = torch.tensor(int(doc["label"]), dtype=torch.long)  # type: ignore[arg-type]
        regression = torch.tensor(
            [float(doc.get(name, 0.0)) for name in REGRESSION_TARGETS],  # type: ignore[arg-type]
            dtype=torch.float32,
        )
        mask = torch.tensor(float(doc.get("targets_present", 0)), dtype=torch.float32)
        return numeric, categorical, outcome, regression, mask

    def _raw_stream(self, client: MongoClient, num_workers: int, worker_id: int) -> Iterator[Sample]:
        """Yield standardised samples from this worker's paged cursor."""
        collection = client[self._mongo.db][self._mongo.feature_collection]
        cursor = collection.find(
            self._build_filter(num_workers, worker_id)
        ).sort([("row_idx", ASCENDING)])
        cursor.batch_size(self._page_size)
        for doc in cursor:
            yield self._to_sample(doc)

    def __iter__(self) -> Iterator[Sample]:
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0

        # Each worker owns its own client/connection (pymongo is not fork-safe).
        client: MongoClient = MongoClient(self._mongo.uri)
        try:
            stream = self._raw_stream(client, num_workers, worker_id)
            if self._shuffle_buffer <= 1:
                yield from stream
                return
            # Reservoir-style buffered shuffle: approximates a global shuffle
            # without materialising the dataset. Seed is offset per worker so
            # shards do not draw an identical permutation.
            rng = random.Random(self._seed + worker_id)
            buffer: List[Sample] = []
            for sample in stream:
                buffer.append(sample)
                if len(buffer) >= self._shuffle_buffer:
                    idx = rng.randrange(len(buffer))
                    buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                    yield buffer.pop()
            rng.shuffle(buffer)
            yield from buffer
        finally:
            client.close()


def make_dataloader(
    dataset: MatchFeatureDataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader[Sample]:
    """Wrap a dataset in a DataLoader.

    ``IterableDataset`` does its own (sharded) iteration, so neither shuffling
    nor a sampler is configured here; default collation stacks the per-sample
    tensor triples into batched ``(B, F)``, ``(B, 4)`` and ``(B,)`` tensors.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def compute_normalization_stats(
    mongo: MongoConfig,
    feature_names: Sequence[str],
    row_range: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and population std over a row range.

    Fitting the scaler on the **training range only** prevents validation/test
    statistics from leaking into the normalisation. The computation runs as a
    single server-side ``$group`` aggregation, so no feature data is pulled into
    the client.

    Args:
        mongo: Mongo connection settings.
        feature_names: Numeric columns to standardise.
        row_range: Half-open ``[start, end)`` train range.

    Returns:
        ``(mean, std)`` arrays aligned with ``feature_names``.
    """
    start, end = row_range
    group: Dict[str, object] = {"_id": None}
    for name in feature_names:
        group[f"{name}__avg"] = {"$avg": f"${name}"}
        group[f"{name}__std"] = {"$stdDevPop": f"${name}"}

    pipeline = [
        {"$match": {"row_idx": {"$gte": start, "$lt": end}}},
        {"$group": group},
    ]

    client: MongoClient = MongoClient(mongo.uri)
    try:
        collection = client[mongo.db][mongo.feature_collection]
        result = list(collection.aggregate(pipeline))
    finally:
        client.close()

    if not result:
        raise RuntimeError("No rows found in the training range for normalisation.")

    stats = result[0]
    mean = np.array([float(stats[f"{n}__avg"] or 0.0) for n in feature_names], dtype=np.float32)
    std = np.array([float(stats[f"{n}__std"] or 0.0) for n in feature_names], dtype=np.float32)
    return mean, std


def compute_class_weights(
    mongo: MongoConfig,
    row_range: Tuple[int, int],
) -> torch.Tensor:
    """Compute inverse-frequency class weights over a row range.

    The 1X2 target is imbalanced (draws are the minority outcome, and home wins
    outnumber away wins because of home advantage). Weighting the cross-entropy
    loss by inverse class frequency counteracts this, preventing the network from
    collapsing onto the majority class. Weights are normalised to mean 1 so the
    overall loss scale (and thus a sensible learning rate) is preserved.

    Args:
        mongo: Mongo connection settings.
        row_range: Half-open ``[start, end)`` train range.

    Returns:
        A ``(NUM_CLASSES,)`` float tensor of per-class weights.
    """
    start, end = row_range
    pipeline = [
        {"$match": {"row_idx": {"$gte": start, "$lt": end}}},
        {"$group": {"_id": "$label", "count": {"$sum": 1}}},
    ]
    client: MongoClient = MongoClient(mongo.uri)
    try:
        collection = client[mongo.db][mongo.feature_collection]
        rows = list(collection.aggregate(pipeline))
    finally:
        client.close()

    counts = np.ones(NUM_CLASSES, dtype=np.float64)  # Laplace smoothing avoids div-by-zero.
    for row in rows:
        label = int(row["_id"])
        if 0 <= label < NUM_CLASSES:
            counts[label] = float(row["count"])

    inverse = counts.sum() / (NUM_CLASSES * counts)
    inverse = inverse / inverse.mean()  # normalise to mean 1
    return torch.tensor(inverse, dtype=torch.float32)


def resolve_split_ranges(
    n_rows: int,
    train_fraction: float,
    val_fraction: float,
) -> Dict[str, Tuple[int, int]]:
    """Translate split fractions into half-open ``row_idx`` ranges.

    Because ``row_idx`` is the global chronological order, slicing it by fraction
    yields a strictly temporal split with no shuffling and no leakage.
    """
    train_end = int(n_rows * train_fraction)
    val_end = int(n_rows * (train_fraction + val_fraction))
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_rows),
    }
