import pandas as pd
import argparse
import re


def main(args):
    print('Loading dataframe')
    df = pd.read_csv(args.data_path)
    print('Calculating bad tweets...')
    df['bad_tweets'] = df['input_text'].apply(lambda x: x.count('… https://t.co'))
    print('Calculating total tweets...')
    df['total_tweets'] = df['input_text'].apply(lambda x: x.count('https://t.co'))

    print(df['bad_tweets'].sum() / df['total_tweets'].sum())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        default='data/200_samples.csv',
        type=str
    )

    arguments = parser.parse_args()

    main(args=arguments)
