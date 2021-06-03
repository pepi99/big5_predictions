import pandas as pd
import re
from models.db_connector import Connector


class DataLoader:
    def __init__(self):
        self.connector = Connector()

    def parse_input(self):
        # df = pd.read_csv(self.input_file)
        db_query = '''SELECT big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' '''
        df = self.connector.query(db_query)
        df.to_csv('../data/big5_data.csv')
        y = df[['big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
                'big5_neuroticism']].to_numpy()

        input_texts = df[['input_text']].astype(str).to_numpy()
        X = [re.sub(r'http\S+', '', text[0]) for text in input_texts]  # remove links
        return X, y
