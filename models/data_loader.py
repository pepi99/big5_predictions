import pandas as pd
import re
from db_connector import Connector
from langdetect import detect
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import PorterStemmer

class DataLoader:
    def __init__(self):
        self.connector = Connector()

    def parse_input(self):
        # df = pd.read_csv(self.input_file)
        db_query = '''SELECT big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' and input_text not like '%𝐇𝐞’𝐬 𝐜𝐮𝐦𝐦𝐢𝐧𝐠 𝐟𝐨𝐫 𝐛𝐞𝐬𝐭 ‘𝐃𝐢𝐜𝐤 𝐓𝐚𝐤𝐞𝐫’ 𝟐𝟎𝟐𝟏%' LIMIT 1000 '''
        df = self.connector.query(db_query)
        #df = pd.read_csv('../data/10K_en_full.csv')
        print('Shape of non-filtered df: ', df.shape)
        print('Filtering out nonenglish texts...')
        df = df[df.input_text.apply(detect).eq('en')]
        #df = df[df.input_text.apply(lambda x: len(x.split()) >= 20000)]
        print('Non english texts filtered!')
        #df.to_csv('../data/10K_en_full.csv')
        #print('df saved!')
        #df.to_csv('../data/big5_data.csv')
        y = df[['big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
                'big5_neuroticism']].to_numpy()

        input_texts = df[['input_text']].astype(str).to_numpy()
        X = [re.sub(r'http\S+', '', text[0]) for text in input_texts]  # remove links
        X = self.tokenize_text(X)
        return X, y
    def tokenize_item(self, item):
        tokens = word_tokenize(item)
        stems = []
        for token in tokens:
            stems.append(PorterStemmer().stem(token))
        return stems
    def tokenize_text(self, text):
        res = []
        for txt in tqdm(text):
            res.append(' '.join(self.tokenize_item(txt.lower())))
        return res
