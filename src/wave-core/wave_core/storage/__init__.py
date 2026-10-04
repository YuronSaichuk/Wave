"""Postgres database access, feature caching, and migrations."""

from wave_core.storage.cache import (
    CACHE_KEYS,
    MAX_CACHE_SIZE_BYTES_4MIN,
    get_cache_path,
    get_cache_size_bytes,
    load_feature_cache,
    save_feature_cache,
)
from wave_core.storage.db import (
    check_db,
    close_pool,
    get_connection,
    get_pool,
)
from wave_core.storage.features import (
    ANALYSIS_VERSION,
    TrackFeaturesRecord,
    count_analyzed_tracks,
    delete_track_features,
    get_track_features,
    upsert_track_features,
)
from wave_core.storage.jobs import (
    JobRecord,
    JobStatus,
    create_job,
    delete_job,
    get_job_by_id,
    update_job,
)
from wave_core.storage.tracks import (
    TrackMetadata,
    TrackRecord,
    UpsertStatus,
    count_tracks,
    get_track_by_id,
    get_track_by_path,
    upsert_track,
)

__all__ = [
    "ANALYSIS_VERSION",
    "CACHE_KEYS",
    "JobRecord",
    "JobStatus",
    "MAX_CACHE_SIZE_BYTES_4MIN",
    "TrackFeaturesRecord",
    "TrackMetadata",
    "TrackRecord",
    "UpsertStatus",
    "check_db",
    "close_pool",
    "count_analyzed_tracks",
    "count_tracks",
    "create_job",
    "delete_job",
    "delete_track_features",
    "get_cache_path",
    "get_cache_size_bytes",
    "get_connection",
    "get_job_by_id",
    "get_pool",
    "get_track_by_id",
    "get_track_by_path",
    "get_track_features",
    "load_feature_cache",
    "save_feature_cache",
    "update_job",
    "upsert_track",
    "upsert_track_features",
]
