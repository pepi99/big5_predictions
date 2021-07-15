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

    parser.add_argument(
        '--use_bpe',
        action='store_true',
        help='Use BPE before the tf idf.'
    )

    parser.add_argument(
        '--use_bert',
        action='store_true',
        help='Use bert pretrained model.'
    )

    parser.add_argument(
        '--clean_text',
        action='store_true',
        help='Remove links, mentions etc before training.'
    )

    # =========== TRAINING ARGUMENTS ============

    parser.add_argument(
        '--batch_size',
        type=int,
        default='16'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default='10',
    )

    return parser
