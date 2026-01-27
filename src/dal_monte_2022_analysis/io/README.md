# io

IO modules handle the translation between files on disk and data objects in memory.
Key pieces:
- `index_dataset.py`: discovers session files based on config patterns.
- `load_mat.py`: consistent `.mat` loader.
- `data_extraction.py`: modality-specific extraction from MATLAB structures.
- `build_dataset.py`: batch extraction pipeline with multiprocessing.
- `clean_dataset.py`: pruning/interpolation pipeline that writes cleaned outputs.

The guiding idea is to keep pipelines explicit and side-effectful (write outputs),
while keeping data logic in `data/`.

Data flow (extraction)
```
configs/dataset.yaml
  -> io/index_dataset.py (find raw .mat files)
    -> io/load_mat.py (load per file)
      -> io/data_extraction.py (extract Position/Pupil/Timeline/ROI)
        -> io/build_dataset.py (serialize pickles)
          -> processed_data_root/date=.../session=.../modality/agent=...pkl
```

Data flow (cleaning)
```
processed pickles (timeline + position + pupil)
  -> data/cleaning.py (prune + interpolate)
    -> io/clean_dataset.py (write cleaned pickles)
      -> processed_data_root/.../modality_cleaned/agent=...pkl
```
