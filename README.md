## Create an environment

The project ships with two parallel environments:

### Baseline environment (KNN + SVD only, no SASRec)

```
conda create -n rs_group_v4 python=3.10
conda activate rs_group_v4
pip install --upgrade setuptools wheel pyquery
conda install -c conda-forge scikit-surprise
pip install -r requirements.txt
```

### SASRec-enabled environment (recommended for the full system)

```
conda env create -f environment_sas.yml
conda activate rs_group_v4_sas
```

This adds PyTorch + RecBole on top of the baseline so that **Algo Variant B (SASRec)** is
available for online serving and offline evaluation.

## Train SASRec (one-time, only needed for the SASRec environment)

```
python -m flaskr.tools.train_sasrec
```

This converts `flaskr/static/ml_data/ratings.csv` into a RecBole `.inter` atomic file
(under `flaskr/static/ml_data/sasrec/data/ml-demo/`) and trains a SASRec model on it.
The best checkpoint is saved to `flaskr/static/ml_data/sasrec/saved/SASRec-*.pth` and a
pointer to it is written to `latest_checkpoint.txt`. If the checkpoint is missing,
Algo Variant B falls back gracefully to MF (SVD) and the UI shows a notice.

## Run the project
```
flask --app flaskr run --debug
```

> **Windows note 1**: if Smart App Control / Device Guard blocks the freshly
> generated `flask.exe` shim (error: "应用程序控制策略已阻止此文件"), launch
> Flask through the Python module entry-point instead — it routes through the
> already-trusted `python.exe`:
> ```
> python -m flask --app flaskr run
> ```
>
> **Windows note 2**: avoid `--debug` in the SASRec environment. Flask's debug
> reloader (watchdog) watches every imported `.py` file, including PyTorch's
> lazily-imported modules under `site-packages/torch/...`; the first request
> triggers those lazy imports, watchdog sees them as "file changed" and the
> server enters an infinite restart loop, leaving the browser hanging on
> "Loading". Run without `--debug`, or scope the watcher to your own code:
> ```
> python -m flask --app flaskr run --debug --exclude-patterns "*\site-packages\*"
> ```

## A/B variants (algorithm x UI)
This demo supports stable A/B assignment using cookies.

- UI Variant A: original UI, rating threshold = 10
- UI Variant B: improved UI with onboarding progress, rating threshold = 5, async
  partial refresh, explanation cards, sorting controls, filter toolbar, tabbed
  sections, **a chronological rating timeline**, **diversity / recency rerank
  sliders**, **a SASRec attention-based "Why these picks?" panel**, and a
  **cold-start poster picker** that lets users tap movies they've watched.
- Algo Variant A: user-based k-NN (KNNWithMeans) + genre multi-hot content
- Algo Variant B: **SASRec (sequential Transformer)** + TF-IDF content
  (falls back to MF/SVD if SASRec is unavailable)

You can override assignment in browser for debugging:

```
/?ui_variant=A&algo_variant=A
/?ui_variant=A&algo_variant=B
/?ui_variant=B&algo_variant=A
/?ui_variant=B&algo_variant=B
```

## Event logging
Experiment events are appended to:

```
./flaskr/static/ml_data/ab_events.jsonl
```

Tracked events include page view, UI rendered, genres saved, rating updates,
onboarding completion, like toggles, clean-all actions, sort and filter changes,
**slider changes (diversity/recency)**, **cold-start poster picks**, and
**attention-explanation views**.

## Offline evaluation (Precision@K / Recall@K / NDCG@K)
Run the evaluation tool to compare KNN, MF, multi-hot, TF-IDF, and SASRec:

```
python -m flaskr.tools.eval_tool --k 12 --rating-threshold 4.0
```

Optional quick run on subset of users:

```
python -m flaskr.tools.eval_tool --max-users 50
```

By default, aggregated metrics are saved to:

```
./flaskr/static/ml_data/offline_eval_results.csv
```

If `recbole`/`torch`/the SASRec checkpoint are missing, the SASRec row is skipped
silently and the rest of the algorithms are still evaluated.

## Add the recommendation algorithm
- Classical CF / content recommenders live in [`flaskr/main.py`](flaskr/main.py).
- The SASRec adapter lives in [`flaskr/tools/sasrec_tool.py`](flaskr/tools/sasrec_tool.py)
  and is invoked from `getRecommendationBy` when `algo_variant == 'B'`.
- Training entry-point: [`flaskr/tools/train_sasrec.py`](flaskr/tools/train_sasrec.py).
- SASRec hyperparameters: [`flaskr/static/ml_data/sasrec/sasrec_config.yaml`](flaskr/static/ml_data/sasrec/sasrec_config.yaml).

## About the Dataset
The dataset path is: ./flaskr/static/ml_data/

The ratings.csv file includes the following columns:
- userId: the IDs of users.  
- movieId: the IDs of movies.  
- rating: the rating given by the user to the movie, on a 5-star scale
- timestamp: the time when the user rated the movie, recorded in seconds since the epoch (as returned by time(2) function). A larger timestamp means the rating was made later.

You can use pandas to convert the timestamp to standard date and time. For example, 1717665888 corresponds to 2024-06-06 09:24:48.
```
import pandas as pd
timestamp = 1717665888
dt_str = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
print(dt_str)
```