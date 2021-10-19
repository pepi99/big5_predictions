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
            self.cursor.execute('INSERT INTO validation_texts_v3 (input_text, big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, big5_openness_p, big5_conscientiousness_p, big5_extraversion_p, big5_agreeableness_p, big5_neuroticism_p) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', (input_text, big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism, big5_openness_p, big5_conscientiousness_p, big5_extraversion_p, big5_agreeableness_p, big5_neuroticism_p))
            self.connection.commit()
        self.cursor.close()
        self.connection.close()
    def insert_analysis(self, ids, X, y_pred=False, non_english=False):
        if non_english:
            for id, xi in zip(ids, X):
                id_int = int(id)
                n_words = len(xi.split())
                self.cursor.execute('INSERT INTO data_analysis_v3 (id, word_count, non_blank_English, is_used_for_test, big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',(id_int, n_words, '0', '0', -1, -1, -1, -1, -1))
                self.connection.commit()
        else:
            if type(y_pred) != bool:
                for id, xi, yi_pred in zip(ids, X, y_pred):
                    id_int = int(id)
                    n_words = len(xi.split())
                    big5_openness_p = int(yi_pred[0])
                    big5_conscientiousness_p = int(yi_pred[1])
                    big5_extraversion_p = int(yi_pred[2])
                    big5_agreeableness_p = int(yi_pred[3])
                    big5_neuroticism_p = int(yi_pred[4])
                    self.cursor.execute('INSERT INTO data_analysis_v3 (id, word_count, non_blank_english, is_used_for_test, big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',(id_int, n_words, '1', '1', big5_openness_p,big5_conscientiousness_p, big5_extraversion_p, big5_agreeableness_p, big5_neuroticism_p))
                    self.connection.commit()
            else:
                for id, xi in zip(ids, X):
                    id_int = int(id)
                    n_words = len(xi.split())
                    self.cursor.execute('INSERT INTO data_analysis_v3 (id, word_count, non_blank_English, is_used_for_test, big5_openness, big5_conscientiousness, big5_extraversion, big5_agreeableness, big5_neuroticism) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',(id_int, n_words, '1', '0', -1, -1, -1, -1, -1))
                    self.connection.commit()
