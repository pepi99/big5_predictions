import pandas as pd
import re


class DataLoader:
    def __init__(self, input_file):
        self.input_file = input_file

    def parse_input(self):
        df = pd.read_csv(self.input_file)[:3500]
        y = df[['big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
                'big5_neuroticism']].to_numpy()

        input_texts = df[['input_text']].astype(str).to_numpy()
        X = [re.sub(r'http\S+', '', text[0]) for text in input_texts]  # remove links
        return X, y
