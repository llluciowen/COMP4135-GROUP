from flask import (
    Blueprint, current_app, make_response, render_template, request
)
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from .tools.data_tool import *

from surprise import Reader
from surprise import KNNWithMeans, SVD
from surprise import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

bp = Blueprint('main', __name__, url_prefix='/')

movies, genres, rates = loadData()

EXPERIMENT_COOKIE_MAX_AGE = 60 * 60 * 24 * 90
DEFAULT_USER_ID = 611
TOP_K = 12

UI_VARIANTS = ('A', 'B')
ALGO_VARIANTS = ('A', 'B')

RATING_THRESHOLD_BY_UI = {
    'A': 10,
    'B': 5,
}

UI_VARIANT_LABELS = {
    'A': 'Control UI',
    'B': 'Improved UI',
}

ALGO_VARIANT_LABELS = {
    'A': 'User-based k-NN + Multi-hot',
    'B': 'MF (SVD) + TF-IDF',
}

VALID_EVENT_NAMES = {
    'page_view',
    'ui_rendered',
    'genres_saved',
    'rating_updated',
    'onboarding_completed',
    'like_toggled',
    'clean_all',
}


def split_cookie_values(raw_cookie):
    if not raw_cookie:
        return []
    return [value for value in raw_cookie.split(',') if value != '']


def coerce_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_variant(raw_value, allowed_variants):
    if not raw_value:
        return None
    candidate = str(raw_value).upper()
    if candidate in allowed_variants:
        return candidate
    return None


def stable_bucket(seed, experiment_name, variants):
    hash_input = f"{seed}:{experiment_name}"
    digest = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    return variants[int(digest, 16) % len(variants)]


def get_or_create_user_seed():
    user_seed = request.cookies.get('ab_seed')
    if user_seed:
        return user_seed, False

    fingerprint = f"{request.headers.get('User-Agent', '')}|{request.remote_addr}|{datetime.now(timezone.utc).isoformat()}"
    user_seed = hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:16]
    return user_seed, True


def get_or_assign_variant(cookie_name, query_variant, seed, variants):
    requested_variant = normalize_variant(query_variant, variants)
    if requested_variant:
        return requested_variant, True

    cookie_variant = normalize_variant(request.cookies.get(cookie_name), variants)
    if cookie_variant:
        return cookie_variant, False

    return stable_bucket(seed, cookie_name, variants), True


def log_experiment_event(event_name, payload):
    try:
        path = f"{current_app.root_path}/static/ml_data/ab_events.jsonl"
        event = {
            'event': event_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        event.update(payload)
        with open(path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(event, ensure_ascii=True) + '\n')
    except OSError:
        # Logging should never break recommendation rendering.
        return


@bp.route('/event', methods=('POST',))
def event():
    payload = request.get_json(silent=True) or {}
    event_name = str(payload.get('event', '')).strip()

    if event_name not in VALID_EVENT_NAMES:
        return {'status': 'ignored', 'reason': 'invalid_event'}, 202

    ui_variant = normalize_variant(request.cookies.get('ui_variant'), UI_VARIANTS)
    algo_variant = normalize_variant(request.cookies.get('algo_variant'), ALGO_VARIANTS)
    user_seed = request.cookies.get('ab_seed')

    event_payload = {
        'ui_variant': ui_variant,
        'algo_variant': algo_variant,
        'ab_seed': user_seed,
        'movie_id': coerce_int(payload.get('movie_id')),
        'rating': coerce_float(payload.get('rating')),
        'likes_count': coerce_int(payload.get('likes_count')),
        'genres_count': coerce_int(payload.get('genres_count')),
        'total_ratings': coerce_int(payload.get('total_ratings')),
        'rating_threshold': coerce_int(payload.get('rating_threshold')),
        'liked': payload.get('liked'),
    }

    log_experiment_event(event_name, event_payload)
    return {'status': 'ok'}, 200


@bp.route('/', methods=('GET', 'POST'))
def index():
    default_genres = genres.to_dict('records')

    user_genres = split_cookie_values(request.cookies.get('user_genres'))
    user_rates = split_cookie_values(request.cookies.get('user_rates'))
    user_likes = split_cookie_values(request.cookies.get('user_likes'))

    user_seed, should_set_seed_cookie = get_or_create_user_seed()
    ui_variant, should_set_ui_cookie = get_or_assign_variant('ui_variant', request.args.get('ui_variant'), user_seed,
                                                            UI_VARIANTS)
    algo_variant, should_set_algo_cookie = get_or_assign_variant('algo_variant', request.args.get('algo_variant'),
                                                                user_seed, ALGO_VARIANTS)
    rating_threshold = RATING_THRESHOLD_BY_UI.get(ui_variant, 10)

    default_genres_movies = getMoviesByGenres(user_genres)[:10]
    recommendations_movies, recommendations_message = getRecommendationBy(user_rates, algo_variant=algo_variant)

    liked_movie_ids = [int(raw_id) for raw_id in user_likes if raw_id.isdigit()]
    likes_similar_movies, likes_similar_message = getLikedSimilarBy(liked_movie_ids, algo_variant=algo_variant)
    likes_movies = getUserLikesBy(user_likes)

    log_experiment_event('page_view', {
        'ui_variant': ui_variant,
        'algo_variant': algo_variant,
        'genres_count': len(user_genres),
        'ratings_count': len(user_rates),
        'likes_count': len(user_likes),
        'rating_threshold': rating_threshold,
        'onboarding_completed': len(user_rates) >= rating_threshold,
    })

    response = make_response(render_template('index.html',
                                             genres=default_genres,
                                             user_genres=user_genres,
                                             user_rates=user_rates,
                                             user_likes=user_likes,
                                             default_genres_movies=default_genres_movies,
                                             recommendations=recommendations_movies,
                                             recommendations_message=recommendations_message,
                                             likes_similars=likes_similar_movies,
                                             likes_similar_message=likes_similar_message,
                                             likes=likes_movies,
                                             ui_variant=ui_variant,
                                             ui_variant_label=UI_VARIANT_LABELS[ui_variant],
                                             algo_variant=algo_variant,
                                             algo_variant_label=ALGO_VARIANT_LABELS[algo_variant],
                                             rating_threshold=rating_threshold,
                                             ))

    if should_set_seed_cookie:
        response.set_cookie('ab_seed', user_seed, max_age=EXPERIMENT_COOKIE_MAX_AGE, samesite='Lax')
    if should_set_ui_cookie:
        response.set_cookie('ui_variant', ui_variant, max_age=EXPERIMENT_COOKIE_MAX_AGE, samesite='Lax')
    if should_set_algo_cookie:
        response.set_cookie('algo_variant', algo_variant, max_age=EXPERIMENT_COOKIE_MAX_AGE, samesite='Lax')

    return response


def getUserLikesBy(user_likes):
    results = []

    valid_likes = [int(movie_id) for movie_id in user_likes if str(movie_id).isdigit()]

    if len(valid_likes) > 0:
        mask = movies['movieId'].isin(valid_likes)
        results = movies.loc[mask]

        original_orders = pd.DataFrame()
        for _id in valid_likes:
            movie = results.loc[results['movieId'] == int(_id)]
            if len(original_orders) == 0:
                original_orders = movie
            else:
                original_orders = pd.concat([movie, original_orders])
        results = original_orders

    if len(results) > 0:
        return results.to_dict('records')
    return results

def is_genre_match(movie_genres, interested_genres):
    return bool(set(movie_genres).intersection(set(interested_genres)))

def getMoviesByGenres(user_genres):
    results = []
    valid_genre_ids = [int(genre_id) for genre_id in user_genres if str(genre_id).isdigit()]

    if len(valid_genre_ids) > 0:
        genres_mask = genres['id'].isin(valid_genre_ids)
        user_genres = [1 if has is True else 0 for has in genres_mask]
        user_genres_df = pd.DataFrame(user_genres,columns=['value'])
        user_genres_df = pd.concat([user_genres_df, genres['name']], axis=1)
        interested_genres = user_genres_df[user_genres_df['value'] == 1]['name'].tolist()
        results = movies[movies['genres'].apply(lambda x: is_genre_match(x, interested_genres))]

    if len(results) > 0:
        return results.to_dict('records')
    return results


def infer_user_id(user_rates_df):
    if len(user_rates_df) == 0:
        return DEFAULT_USER_ID
    return int(user_rates_df.iloc[0]['userId'])


def getRecommendationBy(user_rates, algo_variant='A', top_k=TOP_K):
    results = pd.DataFrame()
    if len(user_rates) > 0:
        reader = Reader(rating_scale=(1, 5))
        try:
            user_rates_df = ratesFromUser(user_rates)
        except (IndexError, ValueError):
            return [], 'No recommendations.'
        training_rates = pd.concat([rates, user_rates_df], ignore_index=True)
        training_data = Dataset.load_from_df(training_rates[['userId', 'movieId', 'rating']], reader=reader)
        trainset = training_data.build_full_trainset()

        if algo_variant == 'B':
            algo = SVD(n_factors=50, n_epochs=20, random_state=42)
            algorithm_name = 'Matrix Factorization (SVD)'
        else:
            algo = KNNWithMeans(sim_options={'name': 'pearson', 'user_based': True})
            algorithm_name = 'User-based k-NN with means'

        algo.fit(trainset)

        user_id = infer_user_id(user_rates_df)
        rated_movie_ids = set(user_rates_df[user_rates_df['userId'] == user_id]['movieId'].tolist())
        predictions = [algo.predict(user_id, movie_id) for movie_id in movies['movieId'].unique() if
                       movie_id not in rated_movie_ids]
        top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_k]

        top_movie_ids = [int(pred.iid) for pred in top_predictions]
        score_map = {int(pred.iid): round(float(pred.est), 3) for pred in top_predictions}
        order_map = {movie_id: index for index, movie_id in enumerate(top_movie_ids)}

        results = movies[movies['movieId'].isin(top_movie_ids)].copy(deep=True)
        results['score'] = results['movieId'].map(score_map)
        results['rank'] = results['movieId'].map(order_map)
        results = results.sort_values(by=['rank']).drop(columns=['rank'])

    if len(results) > 0:
        return results.to_dict('records'), f"These movies are recommended by {algorithm_name}."
    return [], "No recommendations."


def getLikedSimilarBy(user_likes, algo_variant='A'):
    results = pd.DataFrame()
    if len(user_likes) > 0:
        if algo_variant == 'B':
            results = generate_tfidf_recommendation_results(user_likes, TOP_K)
            message = 'The movies are similar to your likes using TF-IDF content vectors.'
        else:
            item_rep_matrix, item_rep_vector, feature_list = item_representation_based_movie_genres(movies)
            user_profile = build_user_profile(user_likes, item_rep_vector, feature_list)
            results = generate_recommendation_results(user_profile, item_rep_matrix, item_rep_vector, TOP_K,
                                                     exclude_movie_ids=user_likes)
            message = 'The movies are similar to your likes using genre multi-hot vectors.'

    if len(results) > 0:
        return results.to_dict('records'), message
    return [], 'No similar movies found.'


def generate_tfidf_recommendation_results(user_likes, k=12):
    content_movies = movies.copy(deep=True)
    content_movies['title'] = content_movies['title'].fillna('')
    content_movies['overview'] = content_movies['overview'].fillna('')
    content_movies['genres_text'] = content_movies['genres'].apply(
        lambda values: ' '.join(values) if isinstance(values, list) else '')
    content_movies['text_features'] = content_movies['title'] + ' ' + content_movies['overview'] + ' ' + content_movies[
        'genres_text']

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(content_movies['text_features'])
    liked_mask = content_movies['movieId'].isin(user_likes)

    if liked_mask.sum() == 0:
        return pd.DataFrame()

    similarity_matrix = cosine_similarity(tfidf_matrix[liked_mask.values], tfidf_matrix)
    content_movies['similarity'] = similarity_matrix.mean(axis=0)

    rec_result = content_movies[~content_movies['movieId'].isin(user_likes)]
    rec_result = rec_result.sort_values(by=['similarity'], ascending=False)[:k]
    rec_result = rec_result.drop(columns=['genres_text', 'text_features'])
    return rec_result


# Step 1: Representing items with multi-hot vectors
def item_representation_based_movie_genres(movies_df):
    movies_with_genres = movies_df.copy(deep=True)
    genre_list = []
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            movies_with_genres.at[index, genre] = 1
            if genre not in genre_list:
                genre_list.append(genre)

    movies_with_genres = movies_with_genres.fillna(0)

    movies_genre_matrix = movies_with_genres[genre_list].to_numpy()
    
    return movies_genre_matrix, movies_with_genres, genre_list

# Step 2: Building user profile
def build_user_profile(movieIds, item_rep_vector, feature_list, weighted=True, normalized=True):
    user_movie_rating_df = item_rep_vector[item_rep_vector['movieId'].isin(movieIds)]
    user_movie_df = user_movie_rating_df[feature_list].mean()
    user_profile = user_movie_df.T
    
    if normalized and sum(user_profile.values) > 0:
        user_profile = user_profile / sum(user_profile.values)
        
    return user_profile


# Step 3: Predicting user preference for items
def generate_recommendation_results(user_profile, item_rep_matrix, movies_data, k=12, exclude_movie_ids=None):
    u_v = user_profile.values
    u_v_matrix = [u_v]
    recommendation_table = cosine_similarity(u_v_matrix, item_rep_matrix)
    recommendation_table_df = movies_data.copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]

    if exclude_movie_ids:
        recommendation_table_df = recommendation_table_df[~recommendation_table_df['movieId'].isin(exclude_movie_ids)]

    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[:k]
    return rec_result
