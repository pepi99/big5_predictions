import pandas as pd
import re
from db_connector import Connector
from langdetect import detect

class DataLoader:
    def __init__(self):
        self.connector = Connector()

    def parse_input(self):
        # df = pd.read_csv(self.input_file)
        db_query = '''SELECT big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' and input_text not like '%𝐇𝐞’𝐬 𝐜𝐮𝐦𝐦𝐢𝐧𝐠 𝐟𝐨𝐫 𝐛𝐞𝐬𝐭 ‘𝐃𝐢𝐜𝐤 𝐓𝐚𝐤𝐞𝐫’ 𝟐𝟎𝟐𝟏%' LIMIT 12000 '''
        df = self.connector.query(db_query)
        print('Shape of non-filtered df: ', df.shape)
        print('Filtering out nonenglish texts...')
        df = df[df.input_text.apply(detect).eq('en')]
        print('Non english texts filtered!')
        #df.to_csv('../data/big5_data.csv')
        y = df[['big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
                'big5_neuroticism']].to_numpy()

        input_texts = df[['input_text']].astype(str).to_numpy()
        X = [re.sub(r'http\S+', '', text[0]) for text in input_texts]  # remove links
        return X, y
