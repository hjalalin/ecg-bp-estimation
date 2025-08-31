ECG‑Only Blood Pressure Estimation
I use the VitalDB open dataset and its official Python package for discovery and loading.
Full dataset and Python library instructions: https://vitaldb.net/dataset/?query=lib#h.ezjaar311v9

Repository Structure
ecg-bp/
├─ README.md
├─ pyproject.toml # or requirements.txt
├─ configs/
│ ├─ features.yaml
│ └─ train_cnn.yaml
├─ data/
│ ├─ raw/ # downloaded waveforms (gitignored)
│ ├─ interim/ # beat tables, features (gitignored)
│ └─ processed/ # model-ready parquet/csv (gitignored)
├─ notebooks/
│ ├─ 00_preview_data.ipynb
│ ├─ 01_feature_baseline.ipynb
│ └─ 02_deep_baseline.ipynb
├─ src/
│ ├─ data/
│ │ ├─ load_vitaldb.py
│ │ └─ beat_labeling.py
│ ├─ features/
│ │ ├─ ecg_clean.py
│ │ ├─ rpeaks_and_intervals.py
│ │ └─ perbeat_features.py
│ ├─ models/
│ │ ├─ train_feature_model.py
│ │ ├─ train_cnn1d.py
│ │ └─ datasets.py
│ └─ viz/
│ ├─ bland_altman.py
│ └─ plots.py
└─ scripts/
├─ 00_download_sample.sh
├─ 10_make_beat_table.py
├─ 20_build_features.py
├─ 30_train_feature_model.py
├─ 40_train_cnn1d.py
└─ 50_eval_and_plots.py