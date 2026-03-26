"""Runtime IO adapters for processed-data artifacts."""

from .processed_data import (
    build_processed_data_path,
    build_processed_out_dir,
    build_processed_output_path,
    build_processed_pickle_path,
    index_agent_paths,
    index_shared_paths,
    list_processed_modalities,
    load_pickle_path,
    load_processed_pickle,
    save_pickle_path,
    save_processed_pickle,
    scan_processed_paths,
    scan_processed_paths_for_filename,
)
from .analysis_index import (
    build_analysis_output_dir,
    scan_analysis_date_paths,
    scan_analysis_paths,
)
from .plot_output import (
    normalize_extension,
    save_figure,
)

__all__ = [
    "build_processed_out_dir",
    "build_processed_data_path",
    "build_processed_output_path",
    "build_processed_pickle_path",
    "index_agent_paths",
    "index_shared_paths",
    "list_processed_modalities",
    "load_pickle_path",
    "load_processed_pickle",
    "save_pickle_path",
    "save_processed_pickle",
    "scan_processed_paths",
    "scan_processed_paths_for_filename",
    "build_analysis_output_dir",
    "scan_analysis_paths",
    "scan_analysis_date_paths",
    "normalize_extension",
    "save_figure",
]
