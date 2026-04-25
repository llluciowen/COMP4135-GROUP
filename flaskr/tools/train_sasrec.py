"""Train a SASRec model on the demo ratings.csv and save a checkpoint usable by sasrec_tool.

Usage (from Demo_materials/):
    python -m flaskr.tools.train_sasrec
    python -m flaskr.tools.train_sasrec --epochs 10  # quick experiment

Outputs:
    flaskr/static/ml_data/sasrec/data/ml-demo/ml-demo.inter   (RecBole atomic file)
    flaskr/static/ml_data/sasrec/saved/SASRec-*.pth          (best checkpoint)
    flaskr/static/ml_data/sasrec/latest_checkpoint.txt        (path of best checkpoint)
"""
import argparse
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SASREC_DIR = PROJECT_ROOT / 'flaskr' / 'static' / 'ml_data' / 'sasrec'
DATA_DIR = SASREC_DIR / 'data' / 'ml-demo'
SAVED_DIR = SASREC_DIR / 'saved'
CONFIG_PATH = SASREC_DIR / 'sasrec_config.yaml'
RATINGS_CSV = PROJECT_ROOT / 'flaskr' / 'static' / 'ml_data' / 'ratings.csv'
LATEST_POINTER = SASREC_DIR / 'latest_checkpoint.txt'


def build_atomic_inter():
    """Convert ratings.csv to RecBole atomic .inter format."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(RATINGS_CSV)
    ratings = ratings[['userId', 'movieId', 'rating', 'timestamp']]
    ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = ratings.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    out_path = DATA_DIR / 'ml-demo.inter'
    header = 'user_id:token\titem_id:token\trating:float\ttimestamp:float\n'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(header)
        for row in ratings.itertuples(index=False):
            f.write(f"{row.user_id}\t{row.item_id}\t{row.rating}\t{row.timestamp}\n")
    print(f"[train_sasrec] Wrote {len(ratings):,} interactions to {out_path}")
    return out_path


def train(epochs_override=None):
    # Lazy import so the rest of the demo doesn't require torch/recbole
    from logging import getLogger
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.sequential_recommender import SASRec
    from recbole.trainer import Trainer
    from recbole.utils import init_seed, init_logger

    config_dict = {'checkpoint_dir': str(SAVED_DIR)}
    if epochs_override is not None:
        config_dict['epochs'] = int(epochs_override)

    config = Config(
        model='SASRec',
        dataset='ml-demo',
        config_file_list=[str(CONFIG_PATH)],
        config_dict=config_dict,
    )
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = SASRec(config, train_data.dataset).to(config['device'])
    logger.info(model)

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=True)
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)

    print('\n=== SASRec validation result ===')
    print(best_valid_result)
    print('\n=== SASRec test result ===')
    print(test_result)

    # Persist pointer to best checkpoint so sasrec_tool can find it
    best_ckpt = getattr(trainer, 'saved_model_file', None)
    if best_ckpt and os.path.exists(best_ckpt):
        LATEST_POINTER.write_text(best_ckpt, encoding='utf-8')
        print(f"[train_sasrec] Best checkpoint: {best_ckpt}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train SASRec on the demo ratings.')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs.')
    parser.add_argument('--skip-build', action='store_true', help='Skip rebuilding the .inter file.')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_build:
        build_atomic_inter()
    train(epochs_override=args.epochs)


if __name__ == '__main__':
    main()
