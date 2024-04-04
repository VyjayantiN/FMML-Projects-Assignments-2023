import typing
import numpy as np


class BayesianMulticlassModel:
    """
    A multi-class bayesian classifier from encoded text tokens
    """

    def __init__(self, num_classes, num_tokens):
        self.counts = np.zeros(shape=(num_classes, num_tokens))
        self.token_to_index = {}

    def fit(self, x_train: typing.Iterable[np.ndarray], y_train: typing.Iterable[int]):
        for x, y in zip(x_train, y_train):
            for token in x:
                if token not in self.token_to_index:
                    self.token_to_index[token] = len(self.token_to_index)
                self.counts[y, self.token_to_index[token]] += 1

    def predict(self, counts_vector):
        class_frequencies = np.sum(self.counts, axis=1)
        word_frequencies = np.sum(self.counts, axis=0)

        prior = class_frequencies / np.sum(class_frequencies)  # p(label)
        likelihood = self.counts / np.expand_dims(
            class_frequencies, axis=1
        )  # p(word|label)
        evidence = word_frequencies / np.sum(word_frequencies)  # p(word)

        likelihood = np.multiply(likelihood, counts_vector)
        prior = np.expand_dims(prior, axis=1)

        posterior_marginal = prior * likelihood / evidence + 0.00001
        posterior_joint = np.sum(np.log(posterior_marginal), axis=1)
        return np.flip(np.argsort(posterior_joint))