from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

model = SentenceTransformer('bert-base-nli-mean-tokens')


def cosine_similarity(sentence_embeddings, ind_a, ind_b):
    s = sentence_embeddings
    return np.dot(s[ind_a], s[ind_b]) / (np.linalg.norm(s[ind_a]) * np.linalg.norm(s[ind_b]))


df = pd.read_csv('twitter_data.csv').iloc[:10000, :]  # Don't take full data
column_names_df = pd.read_csv('column_names.csv', header=None)
column_names = column_names_df.iloc[:, 3].array.to_numpy()
df.columns = column_names

Y = df[['big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
        'big5_neuroticism']].to_numpy()

input_texts = df[['input_text']].astype(str).to_numpy()
# print(input_texts)
input_texts_filter = [re.sub(r'http\S+', '', text[0]) for text in tqdm(input_texts)]
# print(input_texts_filter)
encodings = model.encode(input_texts_filter)
print(Y)
print(encodings)
np.savetxt('input_data/Y.txt', Y)
np.savetxt('input_data/X.txt', encodings)
# text_files = df.index.array.to_numpy()
# text_files = [str(n) for n in text_files]
# documents = [open('large_data/' + f + '.txt').read() for f in text_files]
#
#
# #
# s0 = "our president is a good leader he will not fail"
# s1 = "our president is not a good leader he will fail"
# s2 = "our president is a good leader"
# s3 = "our president will succeed"

# sentences = [s0, s1, s2, s3]
# sentence_embeddings = model.encode(s0)
# print(sentence_embeddings[0].shape)
# print(sentence_embeddings[1].shape)
# print(sentence_embeddings[2].shape)
# print(sentence_embeddings[3].shape)

# print(f"{s0} <--> {s1}: {cosine_similarity(sentence_embeddings, 0, 1)}")
# print(f"{s0} <--> {s2}: {cosine_similarity(sentence_embeddings, 0, 2)}")
# print(f"{s0} <--> {s3}: {cosine_similarity(sentence_embeddings, 0, 3)}")
