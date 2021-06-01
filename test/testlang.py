from langdetect import detect
from googletrans import Translator
import pandas as pd

detector = Translator()


def new_detect(x):
    return detector.detect(x).lang


dec_lan = detector.detect('Hello nice to meet you')

df = pd.read_csv('../data/twitter_data_en.csv')
# print(df[detector.detect(df['input_text']).lang == 'en'])
# print(df[detector.detect(df['input_text']]))
# df_new = df[df.input_text.apply(detect).eq('en')]
print(df.shape)
# df_new.to_csv('../data/twitter_data_en.csv')
