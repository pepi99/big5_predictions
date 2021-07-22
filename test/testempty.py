import psycopg2
import re
import pandas as pd
from langdetect import detect

class Connector:
    def __init__(self):
        host = '134.213.113.101'
        user = 'nlp'
        password = 'p2021nlp-psql'
        # driver = 'SQL+Server'
        db = 'nlp-data'
        port = '5432'
        self.connection = psycopg2.connect(
            database=db,
            user=user,
            password=password,
            host=host,
            port=port
        )

    def query(self, db_query):
        print('Querying database...')
        df = pd.read_sql_query(db_query, self.connection)
        #         df = pd.read_csv('../input/big5-data-encsv/big5_data_en.csv')[:200]
        print(df.shape)
        print('Database queried!')
        return df


class DataLoader:
    def __init__(self):
        self.connector = Connector()

    def parse_input(self):
        # df = pd.read_csv(self.input_file)
        db_query = '''SELECT big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' '''
        df = self.connector.query(db_query)
        # df = df[df.input_text.apply(detect).eq('en')]
        # if not os.path.exists('data'):
        #     os.makedirs('data')
        #         df.to_csv('data/big5_data.csv')
        y = df[['big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
                'big5_neuroticism']].to_numpy()

        input_texts = df[['input_text']].astype(str).to_numpy()
        X = [re.sub(r'http\S+', '', text[0]) for text in input_texts]  # remove links
        return X, y

#dl = DataLoader()
#X, y = dl.parse_input()
#error_texts = ''
#counter = 0
#for text in X:
#    try:
#        language = detect(text)
#    except:
#        counter += 1
#        print('Caught error: ', counter)
#        language = 'error'
#        error_texts += '*'* 200 + text
#tf = open('errors.txt', 'w')
#tf.write(error_texts)
#print('Finished!')
data = open('errors.txt', 'r').read().split('*' * 200)
print(len(data))
row = data[1]
print(row)
print(detect(row))
        
