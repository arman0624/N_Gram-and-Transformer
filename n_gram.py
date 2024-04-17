"""
n-gram language model
"""

import os
import argparse
from typing import List, Any
from collections import Counter
import numpy as np
from load_data import load_data
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer


def get_args():
    parser = argparse.ArgumentParser(description='n-gram model')
    parser.add_argument('-t', '--tokenization_level', type=str, default='character',
                        help="At what level to tokenize the input data")
    parser.add_argument('-n', '--n', type=int, default=1,
                        help="The value of n to use for the n-gram model")

    parser.add_argument('-e', '--experiment_name', type=str, default='testing',
                        help="What should we name our experiment?")
    parser.add_argument('-s', '--num_samples', type=int, default=10,
                        help="How many samples should we get from our model??")
    parser.add_argument('-x', '--max_steps', type=int, default=40,
                        help="What should the maximum output length of our samples be?")
    parser.add_argument('-sm', '--smoothing_tech', type=str, default=None,
                        help="What smoothing technique should we use?")

    args = parser.parse_args()
    return args


class NGramLM:
    name = "n_gram"

    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab_counts = Counter()
        self.vocab = []
        self.smoothing_tech = None

    def learn(self, training_data: List[List[Any]]):
        temp = []
        for sentence in training_data:
            sentence = ['<BOS>'] * (self.n - 1) + sentence
            temp.extend(sentence)
            if self.n != 1:
                for k in range(self.n - 1):
                    for i in range(len(sentence) - (self.n - k) + 1):
                        ngram = tuple(sentence[i:i + (self.n - k)])
                        context = ngram[:-1]
                        self.ngram_counts[ngram] += 1
                        self.context_counts[context] += 1
            else:
                for i in range(len(sentence) - self.n + 1):
                    ngram = tuple(sentence[i:i + self.n])
                    context = ngram[:-1]
                    self.ngram_counts[ngram] += 1
                    self.context_counts[context] += 1
        self.vocab = list(set(temp))
        self.vocab_counts = Counter(temp)

    def log_probability(self, model_input: List[Any], base=np.e):
        model_input = ['<BOS>'] * (self.n - 1) + model_input
        log_prob = 0.0
        for i in range(len(model_input) - self.n + 1):
            prob = 0.0
            ngram = tuple(model_input[i:i + self.n])
            context = ngram[:-1]
            ngram_count = self.ngram_counts[ngram]
            context_count = self.context_counts[context]
            if self.smoothing_tech == "Linear_Interpolation":
                for j in range(self.n, 0, -1):
                    temp = ngram[:-j]
                    if j != 1:
                        contemp = temp[:-1]
                        ngram_count = self.ngram_counts[temp]
                        context_count = self.context_counts[contemp]
                        if ngram_count != 0 and context_count != 0:
                            prob += (ngram_count / context_count)
                    else:
                        prob += (self.vocab_counts.get(temp, 1) / len(self.vocab))
            else:
                if ngram_count != 0 and context_count != 0:
                    prob = ngram_count / context_count
                else:
                    prob = 1e-10
            log_prob += np.log(prob) / np.log(base)
        return log_prob


def main():
    # Get key arguments
    args = get_args()

    # Get the data for language-modeling and WER computation
    tokenization_level = args.tokenization_level
    # Initialize and "train" the n-gram model
    n = args.n
    model_type = "n_gram"
    train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data = load_data(tokenization_level, model_type)
    model = NGramLM(n)
    model.smoothing_tech = args.smoothing_tech
    model.learn(train_data)

    # Evaluate model perplexity
    val_perplexity = evaluate_perplexity(model, val_data)
    print(f'Model perplexity on the val set: {val_perplexity}')
    dev_perplexity = evaluate_perplexity(model, dev_data)
    print(f'Model perplexity on the dev set: {dev_perplexity}')
    test_perplexity = evaluate_perplexity(model, test_data)
    print(f'Model perplexity on the test set: {test_perplexity}')

    # Evaluate model WER
    experiment_name = args.experiment_name
    dev_wer_savepath = os.path.join('results', f'{experiment_name}_n_gram_dev_wer_predictions.csv')
    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath)
    dev_wer = compute_wer('data/wer_data/dev_ground_truths.csv', dev_wer_savepath)
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join('results', f'{experiment_name}_n_gram_test_wer_predictions.csv')
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath)


if __name__ == "__main__":
    main()
