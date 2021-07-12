class TokenizerWrapper:
    def __init__(self, tokenizer):
        """
        Adapter from hugging face tokenizer interface to sklearn tokenizer interface
        :param tokenizer:
        """
        self.tokenizer = tokenizer

    def __call__(self, sequence):
        return self.tokenizer.tokenize(sequence)
