### Very slow

import torch
from transformers import LongformerModel, LongformerTokenizer
import pandas as pd
import re
import numpy as np
from tqdm import tqdm

model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')


SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
print('SAMPLE TEXT LENGTH: ', len(SAMPLE_TEXT.split()))

def encode(text):
    input_ids = torch.tensor(tokenizer.encode(text, truncation=True)).unsqueeze(0)  # batch of size 1
    # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long,
                                device=input_ids.device)  # initialize to local attention
    global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long,
                                        device=input_ids.device)  # initialize to global attention to be deactivated for all tokens
    global_attention_mask[:, [1, 4, 21, ]] = 1  # Set global attention to random tokens for the sake of this example
    # Usually, set global attention based on the task. For example,
    # classification: the <s> token
    # QA: question tokens
    # LM: potentially on the beginning of sentences and paragraphs
    outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    return outputs


df = pd.read_csv('../twitter_data.csv')
column_names_df = pd.read_csv('../column_names.csv', header=None)
column_names = column_names_df.iloc[:, 3].array.to_numpy()
df.columns = column_names

Y = df[['big5_openness', 'big5_conscientiousness', 'big5_extraversion', 'big5_agreeableness',
        'big5_neuroticism']].to_numpy()

input_texts = df[['input_text']].astype(str).to_numpy()
input_texts_filter = [re.sub(r'http\S+', '', text[0]) for text in input_texts]

encodings = []
for input_text in tqdm(input_texts_filter):
    # txt = ' '.join(input_text.split()[:4096])
    outputs = encode(input_text)
    sequence_output = outputs.last_hidden_state
    pooled_output = outputs.pooler_output
    pooled_output = pooled_output.detach().numpy()
    encodings.append(pooled_output)
    # print('Sequence output: ', sequence_output.shape)
    # print('Pooled output: ', pooled_output.shape)
encodings = np.array(encodings)
print('Encodings shape is: ', encodings.shape)
