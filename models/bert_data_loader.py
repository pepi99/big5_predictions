import re
from db_connector import Connector
from langdetect import detect
from nltk.stem import WordNetLemmatizer, PorterStemmer
from tqdm import tqdm
from nltk import word_tokenize

tqdm.pandas()


class DataLoader:
    def __init__(self):
        self.connector = Connector()

    def parse_input(self):
        db_query = '''SELECT big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' LIMIT 500 '''
        df = self.connector.query(db_query)
        df = df[df.input_text.apply(detect).eq('en')]
        #df = df[df.input_text.apply(lambda x: len(x.split()) <= 500)]
        df['input_text'] = df['input_text'].astype(str).apply(lambda x: self.splitter(x))  # apply the spliter
        df = df.explode('input_text')  # enlarge the df
        df['input_text'] = df['input_text'].apply(lambda x: ' '.join(x))  # join
        # df.to_csv('../data/big5_data.csv')
        y = df[['big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
                'big5_neuroticism']].to_numpy()

        input_texts = df[['input_text']].astype(str).to_numpy()
        X = [re.sub(r'http\S+', '', text[0]) for text in input_texts]  # remove links
        print('Tokenizing...')
        X = self.tokenize_text(X)
        print('Tokenizing finished!')
        print('Data loaded!')
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

    def splitter(self, input_text, n=500):
        values = input_text.split()
        return [values[i:i + n] for i in range(0, len(values), n)]

