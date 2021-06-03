from langdetect import detect
# from googletrans import Translator
import pandas as pd

# detector = Translator()


# def new_detect(x):
#     return detector.detect(x).lang


# dec_lan = detector.detect('Hello nice to meet you')

df = pd.read_csv('../data/big5_data.csv')
print(df)
# print(df[detector.detect(df['input_text']).lang == 'en'])
# print(df[detector.detect(df['input_text']]))
df_new = df[df.input_text.apply(detect).eq('en')]
df_new.to_csv('../data/big5_data_en.csv')
# print(df_new)
# print(df_new.shape)
# df_new.to_csv('../data/twitter_data_en.csv')
