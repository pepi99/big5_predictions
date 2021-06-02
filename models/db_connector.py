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

    def query(self, db_query):
        print('Quierying database...')
        df = pd.read_sql_query(db_query, self.connection)
        print('Database queried!')
        return df
