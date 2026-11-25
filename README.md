# Front-pocket activity recognition with two inertial sensors

This project identifies sitting, standing, walking, stair ascent and stair descent from a phone’s accelerometer and gyroscope. Each participant has a personal dictionary of ten-sample motion templates. The experiment compares accelerometer-only, gyroscope-only, raw distance fusion and fusion with per-axis scaling learned from training data, alongside a contiguous three-window smoothing comparison, multi-condition degradation tests and batch compute-time profiling.

## Environment

Tested with Python 3.12+. Run the following from a macOS or Linux terminal:

```sh
git clone https://github.com/Nyvo-io/front-pocket-activity-recognition.git
cd front-pocket-activity-recognition
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Small demo

The included demo contains participant 1’s front-pocket training recordings and Step 1 test recording. Run it without downloading the full dataset:

```sh
python experiment.py --data demo --out demo-results
```

The clean condition has 291 test windows. For raw fusion, accuracy is 0.859107 and macro recall is 0.896037. The demo averages recall over the activities present in this test recording.

## Download the full dataset

Emily Huang and Jukka-Pekka Onnela, *Smartphone Gyroscope and Accelerometer Dataset for Human Activity Recognition* (2020). The demo CSVs are an unmodified subset of the dataset, distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Zenodo dataset record](https://doi.org/10.5281/zenodo.3925679)

```sh
curl -fL --retry 3 https://zenodo.org/api/records/3925679/files/data.zip/content -o data.zip
unzip -q data.zip
```

The extracted directory should contain `data/subject1` through `data/subject4`.

## Reproduce the full experiment

```sh
python experiment.py --data data --out results
python figures.py --out results
python window_context.py --data data --out results
python signals.py --data data --out results
python profiles.py --data data --out results
python sensitivity.py --data data --out results
python bonus_analysis.py --data data --out results
python paper_figures.py
```

Key outputs written to `results/`:
- `predictions.csv` & `scores.csv`: Per-window predictions and aggregate metrics over the 7,632 common test windows under clean, delayed (`0.3 s`), noisy (`0.5σ`) and missing-gyroscope conditions. Clean raw fusion reaches 72.14% macro recall vs. 70.69% for scaled fusion.
- `voting_ablation.csv`: Three-window majority-vote smoothing comparison inspired by the Movelet voting idea (computed strictly within contiguous 5-sample-stride segments so voting never crosses filtered label gaps; note that Huang, Yan and Onnela 2022 vote over ten 1-sample-stride windows starting at `t`), raising clean raw fusion macro recall to 72.47%.
- `lag_sweep.csv` & `noise_sweep.csv`: Gyroscope delay sweep at 0, 1, 2, 3, 5 and 7 samples (7,612 common windows; raw fusion exceeds scaled fusion by 2.08–4.55 percentage points across nonzero lags) and 0–1.5σ multi-level Gaussian noise sweep (5 random seeds per nonzero level).
- `efficiency.csv`: Dictionary memory footprint (`1,256.4 KB` for 6-axis float64 templates) and offline batch-averaged matching compute time (`61.7 μs/window` for raw fusion on Apple M5 at batch size 32; excludes sensor acquisition, window waiting and online output latency).
- `paper_figures/`: All eight publication-ready figures used in the IEEE technical report.

Reference method: [Multistream movelets paper (Sensors 2022)](https://doi.org/10.3390/s22072618)
