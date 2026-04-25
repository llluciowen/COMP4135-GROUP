from flask import (
    Blueprint, current_app, make_response, render_template, request
)
import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

from .tools.data_tool import *
from .tools import sasrec_tool

# Lazy / optional import of scikit-surprise so the SASRec-only env can boot even if
# `surprise` cannot load (e.g. a Windows app-control policy blocking its compiled
# .pyd files). We expose `Reader`, `KNNWithMeans`, `SVD`, `Dataset` if available.
try:
    from surprise import Reader, KNNWithMeans, SVD, Dataset
    _SURPRISE_AVAILABLE = True
    _SURPRISE_ERROR = None
except Exception as _exc:  # pragma: no cover - depends on environment
    _SURPRISE_AVAILABLE = False
    _SURPRISE_ERROR = str(_exc)
    Reader = KNNWithMeans = SVD = Dataset = None  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

bp = Blueprint('main', __name__, url_prefix='/')

movies, genres, rates = loadData()

# Load full ratings (with timestamp) for sequence construction.
_RATINGS_WITH_TIME_PATH = os.path.join(os.path.dirname(__file__), 'static', 'ml_data', 'ratings.csv')
try:
    rates_with_time = pd.read_csv(_RATINGS_WITH_TIME_PATH)
except Exception:
    rates_with_time = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])

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
    'B': 'SASRec (Sequential Transformer) + TF-IDF',
}

VALID_EVENT_NAMES = {
    'page_view',
    'ui_rendered',
    'genres_saved',
    'rating_updated',
    'onboarding_completed',
    'like_toggled',
    'clean_all',
    'async_refresh',
    'sort_changed',
    'rating_modal_saved',
    'filter_changed',
    'slider_changed',
    'cold_start_seed_selected',
    'attention_explanation_viewed',
    'onboarding_variant_picked',
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


def get_active_algo_variant():
    cookie_variant = normalize_variant(request.cookies.get('algo_variant'), ALGO_VARIANTS)
    if cookie_variant:
        return cookie_variant
    return ALGO_VARIANTS[0]


def build_explain_cards(user_genres, user_rates, user_likes, ui_variant, algo_variant):
    selected_genre_names = []
    if len(user_genres) > 0:
        selected_ids = {int(genre_id) for genre_id in user_genres if str(genre_id).isdigit()}
        selected_genre_names = genres[genres['id'].isin(selected_ids)]['name'].tolist()

    if algo_variant == 'B':
        if sasrec_tool.is_available():
            recommendation_logic = (
                'Primary ranking uses a SASRec sequential Transformer that conditions on the time-ordered '
                'sequence of your ratings; similar-item lists use TF-IDF text vectors.'
            )
        else:
            recommendation_logic = (
                'SASRec checkpoint not loaded yet, falling back to MF (SVD). '
                'Run `python -m flaskr.tools.train_sasrec` to enable the sequential model. '
                f'({sasrec_tool.status_message()})'
            )
    else:
        recommendation_logic = 'Primary ranking uses user-based k-NN, and similar-items use genre multi-hot vectors.'

    genre_summary = ', '.join(selected_genre_names[:5]) if selected_genre_names else 'No genres selected yet'

    return [
        {
            'title': 'Why You See These Results',
            'body': recommendation_logic,
        },
        {
            'title': 'Profile Snapshot',
            'body': f"Genres: {len(user_genres)} | Ratings: {len(user_rates)} | Likes: {len(user_likes)}",
        },
        {
            'title': 'Current Focus Genres',
            'body': genre_summary,
        },
        {
            'title': 'Experiment Context',
            'body': f"UI {ui_variant} with algorithm variant {algo_variant}.",
        },
    ]


def compute_recommendation_payload(user_genres, user_rates, user_likes, ui_variant, algo_variant,
                                   diversity=0.0, recency=0.0):
    default_genres_movies = getMoviesByGenres(user_genres)[:10]
    recommendations_movies, recommendations_message = getRecommendationBy(user_rates, algo_variant=algo_variant)

    if algo_variant == 'B' and (diversity > 0.0 or recency > 0.0) and recommendations_movies:
        recommendations_movies = sasrec_tool.diversity_rerank(
            recommendations_movies, diversity=diversity, recency=recency
        )

    liked_movie_ids = [int(raw_id) for raw_id in user_likes if str(raw_id).isdigit()]
    likes_similar_movies, likes_similar_message = getLikedSimilarBy(liked_movie_ids, algo_variant=algo_variant)
    likes_movies = getUserLikesBy(user_likes)
    timeline = build_rating_timeline(user_rates)

    return {
        'default_genres_movies': default_genres_movies,
        'recommendations': recommendations_movies,
        'recommendations_message': recommendations_message,
        'likes_similars': likes_similar_movies,
        'likes_similar_message': likes_similar_message,
        'likes': likes_movies,
        'explain_cards': build_explain_cards(user_genres, user_rates, user_likes, ui_variant, algo_variant),
        'rating_timeline': timeline,
        'sasrec_available': bool(sasrec_tool.is_available() and algo_variant == 'B'),
    }


def build_rating_timeline(user_rates):
    """Return a chronological list of {movieId, title, year, cover_url, rating, position}
    for the current session, used by UI-B's timeline visualization.
    """
    if not user_rates:
        return []
    try:
        rates_df = ratesFromUser(user_rates)
    except (IndexError, ValueError):
        return []
    rates_df = rates_df.copy()
    # Append a tiebreaker so the order matches insertion order from the cookie
    rates_df['order'] = range(len(rates_df))
    rates_df = rates_df.sort_values('order')
    timeline = []
    for pos, row in enumerate(rates_df.itertuples(index=False), start=1):
        movie_id = int(row.movieId)
        meta = movies[movies['movieId'] == movie_id]
        if len(meta) == 0:
            continue
        meta = meta.iloc[0]
        timeline.append({
            'position': pos,
            'movieId': movie_id,
            'title': str(meta.get('title', '')),
            'year': None if pd.isna(meta.get('year')) else int(meta.get('year')),
            'cover_url': str(meta.get('cover_url', '')),
            'rating': float(row.rating),
        })
    return timeline


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
        'section': payload.get('section'),
        'sort_key': payload.get('sort_key'),
        'filter_key': payload.get('filter_key'),
        'filter_value': payload.get('filter_value'),
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


@bp.route('/api/recommendations', methods=('POST',))
def recommendations_api():
    payload = request.get_json(silent=True) or {}

    raw_genres = payload.get('user_genres', split_cookie_values(request.cookies.get('user_genres')))
    raw_rates = payload.get('user_rates', split_cookie_values(request.cookies.get('user_rates')))
    raw_likes = payload.get('user_likes', split_cookie_values(request.cookies.get('user_likes')))

    user_genres = [str(value) for value in raw_genres if str(value) != '']
    user_rates = [str(value) for value in raw_rates if str(value) != '']
    user_likes = [str(value) for value in raw_likes if str(value) != '']

    ui_variant = normalize_variant(payload.get('ui_variant'), UI_VARIANTS)
    if ui_variant is None:
        ui_variant = normalize_variant(request.cookies.get('ui_variant'), UI_VARIANTS) or UI_VARIANTS[0]

    algo_variant = normalize_variant(payload.get('algo_variant'), ALGO_VARIANTS)
    if algo_variant is None:
        algo_variant = get_active_algo_variant()

    diversity = max(0.0, min(1.0, coerce_float(payload.get('diversity'), 0.0) or 0.0))
    recency = max(0.0, min(1.0, coerce_float(payload.get('recency'), 0.0) or 0.0))

    result_payload = compute_recommendation_payload(
        user_genres, user_rates, user_likes, ui_variant, algo_variant,
        diversity=diversity, recency=recency,
    )
    result_payload['rating_threshold'] = RATING_THRESHOLD_BY_UI.get(ui_variant, 10)

    log_experiment_event('async_refresh', {
        'ui_variant': ui_variant,
        'algo_variant': algo_variant,
        'genres_count': len(user_genres),
        'ratings_count': len(user_rates),
        'likes_count': len(user_likes),
        'rating_threshold': result_payload['rating_threshold'],
        'diversity': diversity,
        'recency': recency,
    })

    return result_payload, 200


@bp.route('/api/sasrec_explain', methods=('POST',))
def sasrec_explain_api():
    payload = request.get_json(silent=True) or {}
    raw_rates = payload.get('user_rates', split_cookie_values(request.cookies.get('user_rates')))
    user_rates = [str(value) for value in raw_rates if str(value) != '']
    if not user_rates:
        return {'history': [], 'available': sasrec_tool.is_available()}, 200

    try:
        rates_df = ratesFromUser(user_rates)
    except (IndexError, ValueError):
        return {'history': [], 'available': sasrec_tool.is_available()}, 200

    sequence = [int(mid) for mid in rates_df['movieId'].tolist()]
    top_n = max(1, min(5, coerce_int(payload.get('top_n'), 3) or 3))

    evidence_ids = sasrec_tool.get_attention_evidence(sequence, top_n=top_n)
    history = []
    for movie_id in evidence_ids:
        meta = movies[movies['movieId'] == int(movie_id)]
        if len(meta) == 0:
            continue
        meta = meta.iloc[0]
        history.append({
            'movieId': int(movie_id),
            'title': str(meta.get('title', '')),
            'year': None if pd.isna(meta.get('year')) else int(meta.get('year')),
            'cover_url': str(meta.get('cover_url', '')),
        })

    log_experiment_event('attention_explanation_viewed', {
        'history_size': len(sequence),
        'evidence_size': len(history),
    })

    return {'history': history, 'available': sasrec_tool.is_available()}, 200


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

    page_payload = compute_recommendation_payload(user_genres, user_rates, user_likes, ui_variant, algo_variant)

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
                                             default_genres_movies=page_payload['default_genres_movies'],
                                             recommendations=page_payload['recommendations'],
                                             recommendations_message=page_payload['recommendations_message'],
                                             likes_similars=page_payload['likes_similars'],
                                             likes_similar_message=page_payload['likes_similar_message'],
                                             likes=page_payload['likes'],
                                             explain_cards=page_payload['explain_cards'],
                                             rating_timeline=page_payload['rating_timeline'],
                                             sasrec_available=page_payload['sasrec_available'],
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


def _recommend_without_surprise(user_rates_df, top_k):
    """Genre-affinity + popularity fallback when scikit-surprise is unavailable."""
    user_id = infer_user_id(user_rates_df)
    rated_ids = set(user_rates_df[user_rates_df['userId'] == user_id]['movieId'].tolist())

    # User's positive history (rating >= 4) drives genre affinity, all rated rows fall back.
    pos_df = user_rates_df[(user_rates_df['userId'] == user_id) & (user_rates_df['rating'] >= 4)]
    if len(pos_df) == 0:
        pos_df = user_rates_df[user_rates_df['userId'] == user_id]
    liked_ids = set(pos_df['movieId'].tolist())

    genre_weights = {}
    if liked_ids:
        liked_movies = movies[movies['movieId'].isin(liked_ids)]
        for genre_list in liked_movies['genres'].dropna():
            for g in genre_list:
                genre_weights[g] = genre_weights.get(g, 0) + 1
        # Normalize.
        total = sum(genre_weights.values()) or 1
        genre_weights = {g: w / total for g, w in genre_weights.items()}

    # Global popularity stats from the loaded `rates` table.
    pop_stats = rates.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    pop_stats.columns = ['movieId', 'avg_rating', 'rating_count']
    global_avg = float(rates['rating'].mean()) if len(rates) > 0 else 3.5
    # Bayesian-shrunk rating to dampen low-count noise.
    m = 20.0
    pop_stats['shrunk'] = (
        (pop_stats['rating_count'] * pop_stats['avg_rating'] + m * global_avg)
        / (pop_stats['rating_count'] + m)
    )

    candidates = movies[~movies['movieId'].isin(rated_ids)].copy()
    candidates = candidates.merge(pop_stats[['movieId', 'shrunk']], on='movieId', how='left')
    candidates['shrunk'] = candidates['shrunk'].fillna(global_avg)

    def _genre_score(genre_list):
        if not isinstance(genre_list, list) or not genre_weights:
            return 0.0
        return float(sum(genre_weights.get(g, 0.0) for g in genre_list))

    candidates['genre_aff'] = candidates['genres'].apply(_genre_score)
    # Final score: genre affinity dominates, popularity breaks ties.
    candidates['score'] = candidates['genre_aff'] * 4.0 + (candidates['shrunk'] - global_avg)

    top = candidates.sort_values(by=['score', 'shrunk'], ascending=False).head(top_k).copy()
    if len(top) == 0:
        return [], 'No recommendations.'
    top['score'] = top['score'].round(3)
    top = top.drop(columns=['genre_aff', 'shrunk'])
    return top.to_dict('records'), 'These movies are recommended by genre-affinity scoring with Bayesian-smoothed popularity.'


def getRecommendationBy(user_rates, algo_variant='A', top_k=TOP_K):
    if len(user_rates) == 0:
        return [], "No recommendations."

    try:
        user_rates_df = ratesFromUser(user_rates)
    except (IndexError, ValueError):
        return [], 'No recommendations.'

    if algo_variant == 'B' and sasrec_tool.is_available():
        return _recommend_with_sasrec(user_rates_df, top_k)

    if not _SURPRISE_AVAILABLE:
        # Fallback recommender that does not require scikit-surprise.
        results, fb_msg = _recommend_without_surprise(user_rates_df, top_k)
        return results, fb_msg

    # Algo-A path (or fallback if SASRec unavailable): collaborative filtering with Surprise.
    reader = Reader(rating_scale=(1, 5))
    training_rates = pd.concat([rates, user_rates_df], ignore_index=True)
    training_data = Dataset.load_from_df(training_rates[['userId', 'movieId', 'rating']], reader=reader)
    trainset = training_data.build_full_trainset()

    if algo_variant == 'B':
        algo = SVD(n_factors=50, n_epochs=20, random_state=42)
        algorithm_name = f"Matrix Factorization (SVD) — fallback. {sasrec_tool.status_message()}"
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


def _recommend_with_sasrec(user_rates_df, top_k):
    """Build a chronological item sequence for the active session and rank with SASRec."""
    user_id = infer_user_id(user_rates_df)
    history_movie_ids = [int(mid) for mid in user_rates_df['movieId'].tolist()]

    # Augment with the user's historical ratings from ratings.csv (if any) sorted by timestamp,
    # so cold-start sessions still benefit from the larger context. Session ratings always go last.
    if len(rates_with_time) and user_id in rates_with_time['userId'].values:
        existing = (
            rates_with_time[rates_with_time['userId'] == user_id]
            .sort_values('timestamp')['movieId']
            .astype(int)
            .tolist()
        )
        # Avoid duplicate items between historical and session inputs
        seen = set(history_movie_ids)
        sequence = [m for m in existing if m not in seen] + history_movie_ids\
            if existing else history_movie_ids
    else:
        sequence = history_movie_ids

    ranked = sasrec_tool.recommend_for_sequence(
        sequence,
        top_k=top_k,
        exclude_movie_ids=history_movie_ids,
    )
    if not ranked:
        return [], "SASRec returned no candidates (sequence may be empty in the trained vocabulary)."

    score_map = {mid: round(score, 4) for mid, score in ranked}
    top_movie_ids = [mid for mid, _ in ranked]
    order_map = {mid: idx for idx, mid in enumerate(top_movie_ids)}

    results = movies[movies['movieId'].isin(top_movie_ids)].copy(deep=True)
    results['score'] = results['movieId'].map(score_map)
    results['rank'] = results['movieId'].map(order_map)
    results = results.sort_values(by=['rank']).drop(columns=['rank'])

    if len(results) == 0:
        return [], "SASRec ranked items not found in movie metadata."
    return results.to_dict('records'), 'These movies are recommended by SASRec, conditioned on your time-ordered ratings.'


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
