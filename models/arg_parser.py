import argparse


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        type=str,
        default='data/10K_20K_full_en.csv',
        help='Path to directory with training data.'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='cache/tfidf_pca_nn_full',
        help='Path to trained model.'
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train model from scratch.'
    )

    return parser
