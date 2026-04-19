import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from surprise import Dataset, KNNWithMeans, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(data_dir):
    ratings = pd.read_csv(data_dir / 'ratings.csv')
    movies = pd.read_csv(data_dir / 'movie_info.csv')
    movies['genres'] = movies['genres'].fillna('').apply(lambda value: value.split('|') if value else [])
    return ratings, movies


def time_split_by_user(ratings_df, min_interactions=8, test_ratio=0.2):
    train_chunks = []
    test_chunks = []

    for _, user_df in ratings_df.groupby('userId'):
        user_df = user_df.sort_values('timestamp')
        if len(user_df) < min_interactions:
            continue

        test_size = max(1, int(len(user_df) * test_ratio))
        test_size = min(test_size, len(user_df) - 1)
        if test_size <= 0:
            continue

        train_chunks.append(user_df.iloc[:-test_size])
        test_chunks.append(user_df.iloc[-test_size:])

    if not train_chunks or not test_chunks:
        return pd.DataFrame(), pd.DataFrame()

    train_df = pd.concat(train_chunks, ignore_index=True)
    test_df = pd.concat(test_chunks, ignore_index=True)
    return train_df, test_df


def get_user_sets(df):
    grouped = df.groupby('userId')['movieId'].apply(set)
    return grouped.to_dict()


def precision_recall_ndcg_at_k(ranked_items, relevant_set, k):
    top_k_items = ranked_items[:k]
    hits = [1 if movie_id in relevant_set else 0 for movie_id in top_k_items]

    precision = sum(hits) / k
    recall = sum(hits) / len(relevant_set)

    dcg = 0.0
    for idx, hit in enumerate(hits):
        if hit:
            dcg += 1.0 / math.log2(idx + 2)

    ideal_hits = [1] * min(len(relevant_set), k)
    idcg = 0.0
    for idx, hit in enumerate(ideal_hits):
        if hit:
            idcg += 1.0 / math.log2(idx + 2)

    ndcg = dcg / idcg if idcg > 0 else 0.0
    return precision, recall, ndcg


def evaluate_cf(train_df, test_df, all_movie_ids, k, rating_threshold, algo_name):
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader=reader)
    trainset = train_data.build_full_trainset()

    if algo_name == 'knn':
        algo = KNNWithMeans(sim_options={'name': 'pearson', 'user_based': True})
    else:
        algo = SVD(n_factors=50, n_epochs=20, random_state=42)

    algo.fit(trainset)

    train_seen_by_user = get_user_sets(train_df)
    test_relevant_by_user = (
        test_df[test_df['rating'] >= rating_threshold]
        .groupby('userId')['movieId']
        .apply(set)
        .to_dict()
    )

    metrics = []
    for user_id, relevant_items in test_relevant_by_user.items():
        if not relevant_items:
            continue

        seen_items = train_seen_by_user.get(user_id, set())
        candidates = [movie_id for movie_id in all_movie_ids if movie_id not in seen_items]
        if not candidates:
            continue

        predictions = [algo.predict(user_id, movie_id) for movie_id in candidates]
        ranked = [pred.iid for pred in sorted(predictions, key=lambda pred: pred.est, reverse=True)]
        metrics.append(precision_recall_ndcg_at_k(ranked, relevant_items, k))

    return metrics


def build_multi_hot_matrix(movies_df):
    feature_list = sorted({genre for genres in movies_df['genres'] for genre in genres if genre})
    matrix_df = movies_df[['movieId']].copy(deep=True)
    for genre in feature_list:
        matrix_df[genre] = movies_df['genres'].apply(lambda genres: 1 if genre in genres else 0)
    matrix = matrix_df[feature_list].to_numpy(dtype=float)
    return matrix_df, matrix, feature_list


def build_tfidf_matrix(movies_df):
    text_df = movies_df[['movieId', 'title', 'overview', 'genres']].copy(deep=True)
    text_df['title'] = text_df['title'].fillna('')
    text_df['overview'] = text_df['overview'].fillna('')
    text_df['genres_text'] = text_df['genres'].apply(lambda values: ' '.join(values))
    text_df['text_features'] = text_df['title'] + ' ' + text_df['overview'] + ' ' + text_df['genres_text']

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = vectorizer.fit_transform(text_df['text_features'])
    return text_df, matrix


def rank_from_scores(movie_ids, scores, excluded_ids, k):
    masked_scores = scores.copy()
    for movie_id in excluded_ids:
        index = movie_ids.get(movie_id)
        if index is not None:
            masked_scores[index] = -np.inf

    ranked_indexes = np.argsort(masked_scores)[::-1]
    ranked_movie_ids = [movie_id for movie_id, _ in sorted(movie_ids.items(), key=lambda item: item[1])]

    results = []
    for idx in ranked_indexes:
        if np.isneginf(masked_scores[idx]):
            continue
        results.append(ranked_movie_ids[idx])
        if len(results) == k:
            break

    return results


def evaluate_content_based(train_df, test_df, movies_df, k, rating_threshold, mode='multi_hot'):
    movie_id_to_index = {int(movie_id): idx for idx, movie_id in enumerate(movies_df['movieId'].tolist())}
    train_seen_by_user = get_user_sets(train_df)
    test_relevant_by_user = (
        test_df[test_df['rating'] >= rating_threshold]
        .groupby('userId')['movieId']
        .apply(set)
        .to_dict()
    )

    if mode == 'multi_hot':
        matrix_df, matrix, feature_list = build_multi_hot_matrix(movies_df)
        movie_vectors = matrix_df[feature_list].to_numpy(dtype=float)
    else:
        _, tfidf_matrix = build_tfidf_matrix(movies_df)

    metrics = []
    for user_id, relevant_items in test_relevant_by_user.items():
        if not relevant_items:
            continue

        user_train = train_df[(train_df['userId'] == user_id) & (train_df['rating'] >= rating_threshold)]
        like_movie_ids = [int(movie_id) for movie_id in user_train['movieId'].tolist() if int(movie_id) in movie_id_to_index]
        if not like_movie_ids:
            continue

        seen_items = train_seen_by_user.get(user_id, set())

        if mode == 'multi_hot':
            like_indexes = [movie_id_to_index[movie_id] for movie_id in like_movie_ids]
            user_profile = np.mean(movie_vectors[like_indexes], axis=0)
            scores = cosine_similarity([user_profile], movie_vectors)[0]
        else:
            like_indexes = [movie_id_to_index[movie_id] for movie_id in like_movie_ids]
            scores = cosine_similarity(tfidf_matrix[like_indexes], tfidf_matrix).mean(axis=0)
            scores = np.asarray(scores).reshape(-1)

        ranked = rank_from_scores(movie_id_to_index, scores, seen_items, k)
        metrics.append(precision_recall_ndcg_at_k(ranked, relevant_items, k))

    return metrics


def summarize_metrics(metrics):
    if not metrics:
        return 0.0, 0.0, 0.0, 0

    metrics_array = np.array(metrics)
    return (
        float(np.mean(metrics_array[:, 0])),
        float(np.mean(metrics_array[:, 1])),
        float(np.mean(metrics_array[:, 2])),
        int(len(metrics)),
    )


def maybe_limit_users(train_df, test_df, max_users):
    if max_users <= 0:
        return train_df, test_df

    candidate_users = sorted(test_df['userId'].unique().tolist())[:max_users]
    train_limited = train_df[train_df['userId'].isin(candidate_users)].copy(deep=True)
    test_limited = test_df[test_df['userId'].isin(candidate_users)].copy(deep=True)
    return train_limited, test_limited


def evaluate_all(data_dir, k, rating_threshold, min_interactions, test_ratio, max_users, output_path):
    ratings_df, movies_df = load_data(data_dir)
    train_df, test_df = time_split_by_user(ratings_df, min_interactions=min_interactions, test_ratio=test_ratio)

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError('No train/test data created. Lower min_interactions or test_ratio.')

    train_df, test_df = maybe_limit_users(train_df, test_df, max_users)
    all_movie_ids = sorted(movies_df['movieId'].astype(int).unique().tolist())

    evaluations = []

    for algorithm_name in ['knn', 'mf_svd']:
        cf_metrics = evaluate_cf(train_df, test_df, all_movie_ids, k, rating_threshold, algorithm_name)
        precision, recall, ndcg, users = summarize_metrics(cf_metrics)
        evaluations.append({
            'algorithm': algorithm_name,
            'precision_at_k': precision,
            'recall_at_k': recall,
            'ndcg_at_k': ndcg,
            'users_evaluated': users,
        })

    for algorithm_name, mode in [('content_multi_hot', 'multi_hot'), ('content_tfidf', 'tfidf')]:
        content_metrics = evaluate_content_based(train_df, test_df, movies_df, k, rating_threshold, mode=mode)
        precision, recall, ndcg, users = summarize_metrics(content_metrics)
        evaluations.append({
            'algorithm': algorithm_name,
            'precision_at_k': precision,
            'recall_at_k': recall,
            'ndcg_at_k': ndcg,
            'users_evaluated': users,
        })

    result_df = pd.DataFrame(evaluations)
    result_df = result_df.sort_values(by=['ndcg_at_k'], ascending=False).reset_index(drop=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)

    return result_df


def parse_args():
    parser = argparse.ArgumentParser(description='Offline evaluation for recommender variants.')
    parser.add_argument('--k', type=int, default=12, help='Top-K for ranking metrics.')
    parser.add_argument('--rating-threshold', type=float, default=4.0, help='Relevant-item rating threshold.')
    parser.add_argument('--min-interactions', type=int, default=8, help='Minimum ratings required per user.')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Hold-out ratio per user (time split).')
    parser.add_argument('--max-users', type=int, default=0, help='Limit users for quick experiments. 0 means all.')
    parser.add_argument(
        '--output',
        type=str,
        default='flaskr/static/ml_data/offline_eval_results.csv',
        help='CSV path for saving aggregated results.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / 'flaskr' / 'static' / 'ml_data'
    output_path = project_root / args.output if args.output else None

    results = evaluate_all(
        data_dir=data_dir,
        k=args.k,
        rating_threshold=args.rating_threshold,
        min_interactions=args.min_interactions,
        test_ratio=args.test_ratio,
        max_users=args.max_users,
        output_path=output_path,
    )

    print(results.to_string(index=False))
    if output_path:
        print(f'\nSaved result CSV to: {output_path}')


if __name__ == '__main__':
    main()
