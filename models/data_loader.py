import pandas as pd
import re
from db_connector import Connector
from langdetect import detect
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from language_model import LanModel

tqdm.pandas()
class DataLoader:
    def __init__(self):
        self.connector = Connector()
        self.lanmodel = LanModel()
    def detect_bad(self, text):
        lan = None
        try:
            lan = detect(text)
        except:
            lan = 'err'
            print('Found a bad text...')
        return lan

    def parse_input(self):
        #db_query = '''SELECT id, big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' '''
        db_query = '''SELECT id, big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' UNION ALL (SELECT id, big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM aaa_new_data_set where input_text IS NOT NULL and input_text <> '')
 '''
        df = self.connector.query(db_query)
        print('Shape of non-filtered df: ', df.shape)
        print('Length filtering...')
        df = df[df.input_text.progress_apply(lambda x: len(x.split()) in range(300, 150000))]
        df['input_text'] = df['input_text'].apply(lambda text: ''.join(x for x in text if x.isprintable())) # Otherwise model can't read text
        #print('Filtering out non-english...')
        #df = df[df.input_text.progress_apply(self.detect_bad).eq('en')]
        #print('Done!')
        #print('Shape of the DF: ', df.shape)
        #df.to_csv('../data/4K_5K_full_en.csv')
        #print('df saved!')
        #df = pd.read_csv('../data/300_inf_full_en.csv')
        #print('Original english filtered df shape: ', df.shape)
        print('Saving non english texts for further use (to import them in the DB)...')
        df_non_english = df[df.input_text.apply(self.lanmodel.is_english).ne(True)]
        print('Non english texts saved! Non English df shape is: ', df_non_english.shape)
        print('Now saving English texts for analysis...')
        df = df[df.input_text.apply(self.lanmodel.is_english)]
        print('Filtering finished!')
        print('New filtered english df shape: ', df.shape)
        #df.to_csv('../data/300_10K_full_en_new.csv')
        #print('Dataframe saved!')
        #df_non_english = df[df.input_text.apply(detect).ne('en')]


        self.insert_nonenglish(df_non_english) 


        #df['input_text'] = df['input_text'].str.lower()
        #df['input_text'] = df['input_text'].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
        #df['input_text'] = df['input_text'].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
        #df['input_text'] = df['input_text'].apply(lambda x: re.sub(r'{link}', '', x))
        #df['input_text'] = df['input_text'].apply(lambda x: re.sub(r"\[video\]", '', x))
        #df['input_text'] = df['input_text'].apply(lambda x: re.sub(r'&[a-z]+;', '', x))
        #df['input_text'] = df['input_text'].apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', x))
        #df['input_text'] = df['input_text'].apply(lambda x: re.sub(r'@mention', '', x))

        y = df[['id', 'big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
                'big5_neuroticism']].to_numpy()
    
        df = self.preprocess_df(df)
        input_texts = df[['input_text']].astype(str).to_numpy()
        X = input_texts.flatten()
        #X = [re.sub(r'http\S+', '', text[0]) for text in input_texts]  # remove links
        #X = self.tokenize_text(X)
        return X, y
    def insert_nonenglish(self, df):
        df_non_english = df[df.input_text.apply(lambda x: len(x.split()) >= 300)]
        y = df_non_english[['id', 'big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
                            'big5_neuroticism']].to_numpy()

        df_non_english = self.preprocess_df(df)
        input_texts = df_non_english[['input_text']].astype(str).to_numpy()
        X = input_texts.flatten()
        ids = y[:, 0]
        y = y[:, 1:]
        self.connector.insert_analysis(ids=ids, X=X, non_english=True)
        return

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
    def preprocess_df(self, df):
        df_preprocessed = df.copy(deep=True)
        df_preprocessed['input_text'] = df_preprocessed['input_text'].str.lower()
        df_preprocessed['input_text'] = df_preprocessed['input_text'].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
        df_preprocessed['input_text'] = df_preprocessed['input_text'].apply(
            lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
        df_preprocessed['input_text'] = df_preprocessed['input_text'].apply(lambda x: re.sub(r'{link}', '', x))
        df_preprocessed['input_text'] = df_preprocessed['input_text'].apply(lambda x: re.sub(r"\[video\]", '', x))
        df_preprocessed['input_text'] = df_preprocessed['input_text'].apply(lambda x: re.sub(r'&[a-z]+;', '', x))
        df_preprocessed['input_text'] = df_preprocessed['input_text'].apply(
            lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', x))
        df_preprocessed['input_text'] = df_preprocessed['input_text'].apply(lambda x: re.sub(r'@mention', '', x))
        return df_preprocessed

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

#d = DataLoader()
#rint(d.count_words())
#d.plot_distribution()
