import transformers as t
import torch
import torch.nn as nn

import tqdm
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
import pickle


class BertWrapper:
    def __init__(
            self,
            pretrained_model="vinai/bertweet-base",
            batch_size=32,
            num_epochs=10,
            args=None
    ):
        #pretrained_model = 'prajjwal1/bert-tiny'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.name = pretrained_model.split('/')[-1]
        self.model = t.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=5).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.criterion = nn.MSELoss()
        self.max_len = args.max_len

        self.tokenizer = t.AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)

        self.batch_size = batch_size
        self.epochs = num_epochs
        self.validation_steps = 400
        self.best_model_path = None
        self.args = args

    def fit(self, x, y):
        x = self._get_subtexts(x)

        # implicit shuffling in train_test_split
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=1)
        best_val_loss = np.inf

        for epoch in range(self.epochs):
            # Training
            train_loss = []
            pbar = tqdm.tqdm(range(0, len(x_train), self.batch_size), leave=False)

            for idx in pbar:
                self.optimizer.zero_grad()

                x_batch = x_train[idx: idx + self.batch_size]
                y_batch = torch.FloatTensor(y_train[idx: idx + self.batch_size]).to(self.device)

                y_predicted = self._get_prediction(x_batch)
                loss = self.criterion(y_predicted, y_batch)

                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
                pbar.set_description(
                    f"Epoch {epoch} of {self.epochs}: loss {loss.item():.4f}"
                )
                wandb.log({"Train loss": loss.item()})

                if idx % (self.validation_steps * self.batch_size) == 0:
                    # Validation
                    val_loss = []
                    with torch.no_grad():
                        for val_idx in tqdm.tqdm(range(0, len(x_val), self.batch_size), desc='Validation'):
                            x_batch = x_val[val_idx: val_idx + self.batch_size]

                            y_batch = torch.FloatTensor(
                                y_val[val_idx: val_idx + self.batch_size]
                            ).to(self.device)

                            y_predicted = self._get_prediction(x_batch)
                            loss = self.criterion(y_predicted, y_batch)

                            val_loss.append(loss.item())

                    final_loss = sum(val_loss) / len(val_loss)
                    print("Validation loss:", final_loss)
                    wandb.log({"Validation loss": final_loss})

                    if final_loss < best_val_loss:
                        print("Saving best mdoel weights...")
                        self.best_model_path = 'cache/{}'.format(self.name)
                        self.save(self.best_model_path)

                        best_val_loss = final_loss

    def predict(self, x_test):
        x = self._get_subtexts(x_test, training=False)

        with torch.no_grad():
            return self._get_prediction(x, training=False).cpu().numpy()

    def load(self, path):
        self.model = t.AutoModelForSequenceClassification.from_pretrained(path, num_labels=5).to(self.device)
        print("Model loaded from", path)

    def save(self, path):
        self.model.save_pretrained(path)
        print("Model saved to", path)

    def _get_subtexts(self, x, training=True):
        if self.args.load_tokenized_path is not None and training:
            # Loading pretokenized training set

            with open(self.args.load_tokenized_path, "rb") as fp:
                sequences = pickle.load(fp)
                print("Load previously tokenized sentences")
        else:
            # Tokenize training set from scratch

            sequences = self._get_tokenized_sequences(x)

            if self.args.save_tokenized_path is not None and training:
                # Saving tokenized sequences
                with open(self.args.save_tokenized_path, "wb") as fp:
                    pickle.dump(sequences, fp)

                print("Tokenized sequences was saved to", self.args.save_tokenized_path)

        x = self._get_reshaped_sequences(sequences)

        return x

    def _get_prediction(self, x_batch, training=True):
        predictions = []

        for sample in x_batch:
            if training:
                subbatch_size = min(sample.shape[0], self.args.subbatch_size)
                sample_indexes = np.random.choice(np.arange(subbatch_size), size=subbatch_size, replace=False)
                subbatch = sample[sample_indexes]

                sample = subbatch

            current_prediction = self.model(sample.to(self.device), return_dict=True)['logits']
            predictions.append(current_prediction.mean(axis=0))

        predictions = torch.stack(predictions)

        return predictions

    def _get_reshaped_sequences(self, sequences):
        new_x = []

        for input_ids in sequences:
            # reshape sequences to final batches
            num_elements = input_ids.shape[1] // self.max_len

            new_x.append(
                input_ids[0, :input_ids.shape[1] - input_ids.shape[1] % self.max_len].reshape(
                    (num_elements, self.max_len)
                )
            )

        return new_x

    def _get_tokenized_sequences(self, x):
        sequences = []

        for idx in tqdm.tqdm(range(x.shape[0]), desc='Text tokenization'):
            tokenized = self.tokenizer(x[idx], return_tensors='pt')

            input_ids = tokenized['input_ids']

            sequences.append(input_ids)

        return sequences
