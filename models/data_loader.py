import pandas as pd
import re
from langdetect import detect
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

tqdm.pandas()


class DataLoader:
    def __init__(self):
        self.connector = None

    def detect_bad(self, text):
        lan = None
        try:
            lan = detect(text)
        except:
            lan = 'err'
            print('Found a bad text...')
        return lan

    def parse_input(self, path, clean_text=True):
        #db_query = '''SELECT big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' '''
        #df = self.connector.query(db_query)
        #print('Shape of non-filtered df: ', df.shape)
        #print('Length filtering...')
        #df = df[df.input_text.progress_apply(lambda x: len(x.split()) in range(4000, 5000))]
        #print('Filtering non-english...')
        #df = df[df.input_text.progress_apply(self.detect_bad).eq('en')]
        #print('Done!')
        #print('Shape of the DF: ', df.shape)
        #df.to_csv('../data/4K_5K_full_en.csv')
        #print('df saved!')
        df = pd.read_csv(path)
        print('Shape of the read df is: ', df.shape)
        df['input_text'] = df['input_text'].str.lower()

        if clean_text:
            df['input_text'] = df['input_text'].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
            df['input_text'] = df['input_text'].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
            df['input_text'] = df['input_text'].apply(lambda x: re.sub(r'{link}', '', x))
            df['input_text'] = df['input_text'].apply(lambda x: re.sub(r"\[video\]", '', x))
            df['input_text'] = df['input_text'].apply(lambda x: re.sub(r'&[a-z]+;', '', x))
            df['input_text'] = df['input_text'].apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', x))
            df['input_text'] = df['input_text'].apply(lambda x: re.sub(r'@mention', '', x))

        y = df[['big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
                'big5_neuroticism']].to_numpy()

        input_texts = df[['input_text']].astype(str).to_numpy()
        X = input_texts.flatten()
        #X = [re.sub(r'http\S+', '', text[0]) for text in input_texts]  # remove links
        #X = self.tokenize_text(X)
        return X, y

    @staticmethod
    def tokenize_item(item):
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

    def count_words(self):
        #db_query = '''SELECT big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' '''
        #df = self.connector.query(db_query)
        #df = df[df.input_text.apply(self.detect_bad).eq('en')]
        df = pd.read_csv('../data/300_inf_full_en.csv')
        d = {}
        f1 = df[df.input_text.apply(lambda x: len(x.split()) in range(300, 1000))].shape[0]
        f2 = df[df.input_text.apply(lambda x: len(x.split()) in range(1000, 2000))].shape[0]
        f3 = df[df.input_text.apply(lambda x: len(x.split()) in range(2000, 3000))].shape[0]
        f4 = df[df.input_text.apply(lambda x: len(x.split()) in range(3000, 4000))].shape[0]
        f5 = df[df.input_text.apply(lambda x: len(x.split()) in range(4000, 5000))].shape[0]
        f6 = df[df.input_text.apply(lambda x: len(x.split()) in range(5000, 10000))].shape[0]
        f7 = df[df.input_text.apply(lambda x: len(x.split()) in range(10000, 15000))].shape[0]
        f8 = df[df.input_text.apply(lambda x: len(x.split()) in range(15000, 25000))].shape[0]
        f9 = df[df.input_text.apply(lambda x: len(x.split()) in range(25000, 35000))].shape[0]
        f10 = df[df.input_text.apply(lambda x: len(x.split()) in range(35000, 50000))].shape[0]
        f11 = df[df.input_text.apply(lambda x: len(x.split()) in range(50000, 100000))].shape[0]

        d['300-1000'] = f1
        d['1000-2000'] = f2
        d['2000-3000'] = f3
        d['3000-4000'] = f4
        d['4000-5000'] = f5
        d['5000-10000'] = f6
        d['10000-15000'] = f7
        d['15000-25000'] = f8
        d['25000-35000'] = f9
        d['35000-50000'] = f10
        d['50000-100000'] = f11
        print('Lengths are : ', d)

    def plot_distribution(self):
        #db_query = '''SELECT big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' LIMIT 10000 '''
        #df = self.connector.query(db_query)
        #df = df[df.input_text.apply(detect).eq('en')]
        df = pd.read_csv('../data/300_inf_full_en.csv')
        input_texts = df[['input_text']].astype(str).to_numpy()
        X = input_texts.flatten()
        lengths = [len(x.split()) for x in X]
           
        plt.hist(lengths, color='blue', bins=100, edgecolor='black')
        plt.savefig('../visualization/word_distribution.png')
        #plt.show()
