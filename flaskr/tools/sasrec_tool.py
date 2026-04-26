"""Runtime adapter that loads a trained SASRec model and serves recommendations
for a given user history (sequence of movieIds) plus attention-style explanations.

Usage:
    from flaskr.tools.sasrec_tool import is_available, recommend_for_sequence, get_attention_evidence

The module is intentionally lazy — torch/recbole are only imported on first use,
so the rest of the demo continues to run without these heavier dependencies.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SASREC_DIR = PROJECT_ROOT / 'flaskr' / 'static' / 'ml_data' / 'sasrec'
LATEST_POINTER = SASREC_DIR / 'latest_checkpoint.txt'

_state_lock = threading.Lock()
_state = {
    'loaded': False,
    'load_error': None,
    'model': None,
    'dataset': None,
    'config': None,
    'device': None,
    'token_to_id': None,    # mapping movieId-string -> internal int
    'id_to_token': None,    # internal int -> movieId-string
    'max_seq_len': 50,
}


def _resolve_checkpoint() -> Path | None:
    if LATEST_POINTER.exists():
        candidate = Path(LATEST_POINTER.read_text(encoding='utf-8').strip())
        if candidate.exists():
            return candidate
    saved_dir = SASREC_DIR / 'saved'
    if saved_dir.exists():
        candidates = sorted(saved_dir.glob('SASRec-*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return None


def _ensure_loaded() -> bool:
    """Load the saved SASRec model on first use. Returns True on success."""
    if _state['loaded']:
        return True
    if _state['load_error'] is not None:
        return False

    with _state_lock:
        if _state['loaded']:
            return True
        if _state['load_error'] is not None:
            return False

        ckpt = _resolve_checkpoint()
        if ckpt is None:
            _state['load_error'] = (
                'No SASRec checkpoint found. Run `python -m flaskr.tools.train_sasrec` first.'
            )
            return False

        try:
            import torch  # noqa: F401
            from recbole.quick_start import load_data_and_model
        except ImportError as exc:
            _state['load_error'] = (
                f'PyTorch / RecBole not installed in this environment ({exc}). '
                'Install via environment_sas.yml.'
            )
            return False

        try:
            config, model, dataset, _train, _valid, _test = load_data_and_model(model_file=str(ckpt))
        except Exception as exc:
            _state['load_error'] = f'Failed to load SASRec checkpoint {ckpt}: {exc}'
            return False

        model.eval()
        item_field = config['ITEM_ID_FIELD']
        token_to_id = {str(token): int(idx) for token, idx in dataset.field2token_id[item_field].items()}
        id_to_token = {int(idx): str(token) for token, idx in token_to_id.items()}

        _state.update({
            'loaded': True,
            'model': model,
            'dataset': dataset,
            'config': config,
            'device': config['device'],
            'token_to_id': token_to_id,
            'id_to_token': id_to_token,
            'max_seq_len': int(config['MAX_ITEM_LIST_LENGTH']),
        })
        return True


def is_available() -> bool:
    """Return True iff a SASRec checkpoint can be loaded right now."""
    return _ensure_loaded()


def status_message() -> str:
    if _state['loaded']:
        return 'SASRec ready.'
    return _state['load_error'] or 'SASRec not loaded yet.'


def _map_external_to_internal(movie_ids: Iterable[int]) -> List[int]:
    token_to_id = _state['token_to_id'] or {}
    out: List[int] = []
    for mid in movie_ids:
        token = str(int(mid))
        idx = token_to_id.get(token)
        if idx is not None and idx > 0:  # 0 is RecBole's [PAD]
            out.append(idx)
    return out


def _build_interaction(internal_seq: List[int]):
    """Construct a RecBole Interaction containing a single padded sequence."""
    import torch
    from recbole.data.interaction import Interaction

    max_len = _state['max_seq_len']
    if len(internal_seq) > max_len:
        internal_seq = internal_seq[-max_len:]
    seq_len = len(internal_seq)
    padded = internal_seq + [0] * (max_len - seq_len)

    config = _state['config']
    item_seq_field = config['ITEM_LIST_LENGTH_FIELD'] if False else 'item_id_list'  # default name
    # RecBole names the sequence field as f"{ITEM_ID_FIELD}_list"
    item_id_field = config['ITEM_ID_FIELD']
    seq_field = f"{item_id_field}_list"
    seq_len_field = config['ITEM_LIST_LENGTH_FIELD']

    interaction = Interaction({
        seq_field: torch.tensor([padded], dtype=torch.long),
        seq_len_field: torch.tensor([seq_len], dtype=torch.long),
    }).to(_state['device'])
    return interaction


def recommend_for_sequence(
    movie_ids_chronological: Iterable[int],
    top_k: int = 12,
    exclude_movie_ids: Iterable[int] | None = None,
) -> List[Tuple[int, float]]:
    """Score all items conditioned on the given chronological history.

    Returns: list of (movieId, score) sorted by score descending, length <= top_k.
    Items already in history (and exclude_movie_ids) are filtered out.
    Returns [] if SASRec cannot be loaded or the sequence has no known items.
    """
    if not _ensure_loaded():
        return []

    internal_seq = _map_external_to_internal(movie_ids_chronological)
    if not internal_seq:
        return []

    import torch

    interaction = _build_interaction(internal_seq)
    with torch.no_grad():
        scores = _state['model'].full_sort_predict(interaction)  # shape: [1, n_items]
    scores = scores.detach().cpu().flatten()

    # Mask padding + already-seen items
    seen_internal = set(internal_seq)
    if exclude_movie_ids:
        seen_internal.update(_map_external_to_internal(exclude_movie_ids))
    scores[0] = float('-inf')  # PAD token
    for idx in seen_internal:
        if 0 <= idx < scores.shape[0]:
            scores[idx] = float('-inf')

    top_n = min(int(top_k), int((scores > float('-inf')).sum().item()))
    if top_n <= 0:
        return []
    top_scores, top_indices = torch.topk(scores, k=top_n)

    id_to_token = _state['id_to_token']
    results: List[Tuple[int, float]] = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        token = id_to_token.get(int(idx))
        if token is None:
            continue
        try:
            results.append((int(token), float(score)))
        except (TypeError, ValueError):
            continue
    return results


def get_attention_evidence(
    movie_ids_chronological: Iterable[int],
    top_n: int = 3,
) -> List[int]:
    """Return up to top_n movieIds from the user's history that the SASRec
    last-step attention attended to most. Falls back to "most recent N" if
    attention weights cannot be extracted.
    """
    if not _ensure_loaded():
        return list(movie_ids_chronological)[-top_n:]

    history = list(movie_ids_chronological)
    internal_seq = _map_external_to_internal(history)
    if not internal_seq:
        return history[-top_n:]

    import torch

    model = _state['model']
    config = _state['config']
    max_len = _state['max_seq_len']
    truncated_history = history[-len(internal_seq):][-max_len:]
    truncated_internal = internal_seq[-max_len:]

    seq_len = len(truncated_internal)
    padded = truncated_internal + [0] * (max_len - seq_len)
    item_seq = torch.tensor([padded], dtype=torch.long, device=_state['device'])
    item_seq_len = torch.tensor([seq_len], dtype=torch.long, device=_state['device'])

    weights = None
    try:
        # Reproduce SASRec.forward up to the encoded sequence and grab the last block's attention.
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = model.position_embedding(position_ids)
        item_emb = model.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = model.LayerNorm(input_emb)
        input_emb = model.dropout(input_emb)
        extended_attention_mask = model.get_attention_mask(item_seq)

        # output_all_encoded_layers=True so RecBole's TransformerEncoder returns every layer.
        # We need the *attention* matrix of the last layer. RecBole's MultiHeadAttention does not
        # expose it by default, so we re-run the last MHA manually to grab the attention probs.
        hidden = input_emb
        for layer in model.trm_encoder.layer[:-1]:
            hidden = layer(hidden, extended_attention_mask)
        last_layer = model.trm_encoder.layer[-1]
        mha = last_layer.multi_head_attention

        # Replicate the internals of RecBole's MultiHeadAttention.forward to capture attention probs.
        mixed_q = mha.query(hidden)
        mixed_k = mha.key(hidden)
        q = mha.transpose_for_scores(mixed_q)
        k = mha.transpose_for_scores(mixed_k)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / (mha.attention_head_size ** 0.5)
        attention_scores = attention_scores + extended_attention_mask
        attention_probs = torch.softmax(attention_scores, dim=-1)  # [1, heads, L, L]

        last_step = seq_len - 1  # position attending *out*
        per_head = attention_probs[0, :, last_step, :seq_len]  # [heads, L]
        weights = per_head.mean(dim=0).detach().cpu().tolist()
    except Exception:
        weights = None

    if weights is None or len(weights) != seq_len:
        return truncated_history[-top_n:]

    # Pair weights with the user-facing movieIds and pick the largest, excluding the last step itself.
    pairs = list(zip(truncated_history, weights))
    if len(pairs) > 1:
        pairs_excluding_last = pairs[:-1]
    else:
        pairs_excluding_last = pairs
    pairs_excluding_last.sort(key=lambda x: x[1], reverse=True)
    picked = [int(mid) for mid, _w in pairs_excluding_last[:top_n]]
    return picked
