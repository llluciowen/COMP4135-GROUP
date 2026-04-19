## Create an environment

```
conda create -n lab3
conda activate lab3

```

## Install Python packages 

```
pip install --upgrade setuptools wheel pyquery
conda install -c conda-forge scikit-surprise
pip install -r requirements.txt

```

## Run the project
```
flask --app flaskr run --debug
```

## A/B variants (algorithm x UI)
This demo now supports stable A/B assignment using cookies.

- UI Variant A: original UI, rating threshold = 10
- UI Variant B: improved UI with onboarding progress, rating threshold = 5
- Algo Variant A: user-based k-NN (KNNWithMeans) + genre multi-hot content
- Algo Variant B: MF (SVD) + TF-IDF content

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

Current tracked events include page view, UI rendered, genres saved, rating updates,
onboarding completion, like toggles, and clean-all actions.

## Offline evaluation (Precision@K / Recall@K / NDCG@K)
Run the evaluation tool to compare KNN, MF, multi-hot, and TF-IDF:

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

## Add the recommendation algorithm
You only need to modify the `main.py` file. Its path is as follows:
```
path: /flaskr/main.py
```

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