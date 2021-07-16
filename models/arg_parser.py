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
        default='cache/hf_model',
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
        '--pretrained_model',
        type=str,
        default='bertweet-base',
        help='Name of the hugging face model.'
    )

    parser.add_argument(
        '--long',
        action='store_true',
        help='Use linear transformers.'
    )

    parser.add_argument(
        '--clean_text',
        action='store_true',
        help='Remove links, mentions etc before training.'
    )

    parser.add_argument(
        '--save_tokenized_path',
        type=str,
        default=None,
        help='Save tokenized sequences.'
    )

    parser.add_argument(
        '--load_tokenized_path',
        type=str,
        default=None,
        help='Load tokenized sequence.'
    )

    # =========== TRAINING ARGUMENTS ============

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16
    )

    parser.add_argument(
        '--subbatch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--max_len',
        type=int,
        default=128,
        help='Maximum length for input sequences.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
    )

    return parser
