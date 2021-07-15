import transformers as t
import torch
import torch.nn as nn

import tqdm
from sklearn.model_selection import train_test_split
import wandb
import numpy as np


MAX_SEQ_LEN = 128


class BertWrapper:
    def __init__(
            self,
            pretrained_model="vinai/bertweet-base",
            batch_size=32,
            num_epochs=10
    ):
        #pretrained_model = 'prajjwal1/bert-tiny'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.name = pretrained_model.split('/')[-1]
        self.model = t.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, num_labels=5).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

        self.tokenizer = t.AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)

        self.batch_size = batch_size
        self.epochs = num_epochs
        self.validation_steps = 200
        self.best_model_path = None

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

                if idx % self.validation_steps == 0:
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

                    if final_loss < best_val_loss:
                        print("Saving best mdoel weights...")
                        self.best_model_path = 'cache/{}_{}.pt'.format(self.name, idx + epoch)
                        self.save(self.best_model_path)

    def predict(self, x_test):
        x = self._get_subtexts(x_test)

        with torch.no_grad():
            return self._get_prediction(x).numpy()

    def load(self, path):
        self.model = t.AutoModelForSequenceClassification.from_pretrained(path, num_labels=5).to(self.device)
        print("Model saved to", path)

    def save(self, path):
        self.model.save_pretrained(path)

    def _get_subtexts(self, x):
        new_x = []

        for idx in tqdm.tqdm(range(x.shape[0]), desc='Text tokenization'):
            tokenized = self.tokenizer(x[idx], return_tensors='pt')

            elem = tokenized['input_ids']
            num_elements = elem.shape[1] // MAX_SEQ_LEN

            new_x.append(
                elem[0, :elem.shape[1] - elem.shape[1] % MAX_SEQ_LEN].reshape(
                    (num_elements, MAX_SEQ_LEN)
                )
            )

        x = new_x

        return x

    def _get_prediction(self, x_batch):
        predictions = []

        for sample in x_batch:
            current_prediction = self.model(sample.to(self.device), return_dict=True)['logits']
            predictions.append(current_prediction.mean(axis=0))

        predictions = torch.stack(predictions)

        return predictions
