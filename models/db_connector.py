import pandas as pd
import psycopg2


# query = '''SELECT big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, input_text  FROM data_personality_analiser_nlp where input_text IS NOT NULL and input_text <> '' '''

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
        self.cursor = self.connection.cursor()

    def query(self, db_query):
        print('Quierying database...')
        df = pd.read_sql_query(db_query, self.connection)
        # df = pd.read_csv('../data/big5_data_en.csv')
        print(df.shape)
        print('Database queried!')
        return df
    def insert(self, input_texts, y, yhat):
        for input_text, y_i, yhat_i in zip(input_texts, y, yhat):
            big5_openness = int(y_i[0])
            big5_conscientiousness = int(y_i[1])
            big5_extraversion = int(y_i[2])
            big5_agreeableness = int(y_i[3])
            big5_neuroticism = int(y_i[4])

            big5_openness_p = int(yhat_i[0])
            big5_conscientiousness_p = int(yhat_i[1])
            big5_extraversion_p = int(yhat_i[2])
            big5_agreeableness_p = int(yhat_i[3])
            big5_neuroticism_p = int(yhat_i[4])
            self.cursor.execute('INSERT INTO validation_texts (input_text, big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, big5_openness_p, big5_conscientiousness_p, big5_extraversion_p, big5_agreeableness_p, big5_neuroticism_p) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', (input_text, big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, big5_openness_p, big5_conscientiousness_p, big5_extraversion_p, big5_agreeableness_p, big5_neuroticism_p))
            self.connection.commit()
        self.cursor.close()
        self.connection.close()

