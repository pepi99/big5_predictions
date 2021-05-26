import unittest

from models.bert_model import BertModel


class TestModels(unittest.TestCase):
    def test_bert(self):
        model = BertModel()
        texts = ['abc', 'abcd.', 'abcd efgh']
        texts_embeddings = model.encode(texts)
        print(texts_embeddings)
        self.assertEqual(3, len(texts_embeddings))
