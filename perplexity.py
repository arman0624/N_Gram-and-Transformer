from typing import List, Any, Tuple
import torch
from tqdm import tqdm

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def evaluate_perplexity(model: Any, data: Any):
    """
    Function for computing perplexity on the provided dataset.

    Inputs:
        model (Any): An n-gram or Transformer model.
        data (Any):  Data in the form suitable for the input model. For the n-gram model,
                     data should be of type List[List[Any]]. For the transformer, the data
                     should by of type torch.utils.data.DataLoader.
    """
    m = num_tokens_in_corpus(model, data)
    l = corpus_log_probability(model, data)
    return 2 ** (- l / m)


def num_tokens_in_corpus(model, data):
    """
    Helper function returning the number of tokens in the corpus
    """
    if type(data) == list:
        total = sum([len(dp) for dp in data])
    else:
        padding_idx = model.padding_idx

        total = 0
        input_seq = data['input_sequences']
        target_seq = data['target_sequences']
        for i in range(len(input_seq)):
            input_tokens = input_seq[i]
            target_tokens = target_seq[i]
            total += torch.sum(target_tokens != padding_idx).item()

    return total


def corpus_log_probability(model, data):
    """
    Helper function computing the total log-probability of the input corpus
    """
    log_p = 0
    if type(data) == list:
        for datapoint in data:
            log_p += model.log_probability(datapoint, base=2)
    else:
        input_seq = data['input_sequences']
        target_seq = data['target_sequences']
        for i in tqdm(range(len(input_seq))):
            input_tokens = input_seq[i].to(DEVICE).reshape(1, -1)
            target_tokens = target_seq[i].to(DEVICE).reshape(1, -1)
            log_p += model.log_probability(input_tokens.long(), target_tokens.long(), base=2)

    return log_p
