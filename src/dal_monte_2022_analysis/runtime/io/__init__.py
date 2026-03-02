"""Runtime IO adapters for processed-data artifacts."""

from .processed_data import (
    build_processed_pickle_path,
    index_agent_paths,
    index_shared_paths,
    load_pickle_path,
    load_processed_pickle,
    save_pickle_path,
    save_processed_pickle,
    scan_processed_paths,
    scan_processed_paths_for_filename,
)
from .analysis_index import scan_analysis_date_paths, scan_analysis_paths
from .plot_output import (
    normalize_extension,
    save_figure,
)

__all__ = [
    "build_processed_pickle_path",
    "index_agent_paths",
    "index_shared_paths",
    "load_pickle_path",
    "load_processed_pickle",
    "save_pickle_path",
    "save_processed_pickle",
    "scan_processed_paths",
    "scan_processed_paths_for_filename",
    "scan_analysis_paths",
    "scan_analysis_date_paths",
    "normalize_extension",
    "save_figure",
]
