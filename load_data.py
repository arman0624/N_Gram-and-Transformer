import csv
import json
import os
import random

import torch
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
import re
from tokenizers import Tokenizer

from typing import List, Any, Tuple, Dict
import pickle


def dump_lookup_table(item2id):
    print("Dumping lookup table..., size:", len(item2id))
    with open("item2id.pkl", "wb") as f:
        pickle.dump(item2id, f)
    id2item = {v: k for k, v in item2id.items()}
    with open("id2item.pkl", "wb") as f:
        pickle.dump(id2item, f)
    print("Done.")


def load_lookup_table():
    print("Loading lookup table...")
    with open("item2id.pkl", "rb") as f:
        item2id = pickle.load(f)
    with open("id2item.pkl", "rb") as f:
        id2item = pickle.load(f)
    print("Done.")
    return item2id, id2item


def load_text_file(file_path: str) -> List[str]:
    """Reads a text file and returns a list of lines."""
    print(f"Loading text file from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]


def load_csv_file(file_path: str):
    data = {}
    print(f"Loading CSV file from {file_path}...")
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'id':
                continue
            id = row[0]
            data[id] = {
                'sentence': row[1],
            }
    return data


def load_json_file(file_path: str) -> Any:
    """Reads a JSON file and returns the parsed JSON."""
    print(f"Loading JSON file from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def tokenize(text: str, level: str) -> List[str]:
    """Tokenizes the input text based on the specified level (word or character)."""
    if level == 'word':
        # Simple word tokenizer, consider using a more robust method for complex texts
        return re.findall(r'\b\w+\b', text.lower())
    elif level == 'character':
        return list(text.lower())
    else:
        raise ValueError("Unsupported tokenization level. Choose 'word' or 'character'.")


def build_vocab(data, tokenization_level) -> Dict[str, int]:
    system_labels = [
        '<bos>',
        '<eos>',
        '<pad>',
        '<unk>'
    ]
    word2idx = {}
    for label in system_labels:
        word2idx[label] = len(word2idx)

    for line in tqdm(data):
        for token in line:
            if token not in word2idx:
                word2idx[token] = len(word2idx)

    print(f"Vocab size: {len(word2idx)}, dumping to file...")
    dump_lookup_table(word2idx)
    print("Done.")
    return word2idx


def text_to_indices(data, vocab: Dict[str, int], to_idx=True):
    if not to_idx:
        return data
    res = []
    for line in data:
        res.append([vocab.get(item, vocab['<unk>']) for item in line])
    return res


def load_data(tokenization_level: str, model_type: str, split=0.2):
    # Load raw data
    base_path = 'data/lm_data'
    train_data_raw = load_text_file(os.path.join(base_path, 'treebank-sentences-train.txt'))
    dev_data_raw = load_text_file(os.path.join(base_path, 'treebank-sentences-dev.txt'))
    test_data_raw = load_text_file(os.path.join(base_path, 'treebank-sentences-test.txt'))

    # Load WER data
    base_wer_path = 'data/wer_data'
    dev_ground_truths = load_csv_file(os.path.join(base_wer_path, 'dev_ground_truths.csv'))
    dev_sentences = load_json_file(os.path.join(base_wer_path, 'dev_sentences.json'))
    test_sentences = load_json_file(os.path.join(base_wer_path, 'revised_test_sentences.json'))

    if model_type == 'n_gram':
        if tokenization_level == 'subword':
            return n_gram_subword_processing(base_path, dev_data_raw, dev_ground_truths, dev_sentences, split,
                                             test_data_raw, test_sentences, train_data_raw)
        elif tokenization_level == 'word':
            return word_and_character_processing(dev_data_raw, dev_ground_truths, dev_sentences, split, test_data_raw,
                                                 test_sentences, tokenization_level, train_data_raw, to_idx=False)
        elif tokenization_level == 'character':
            return word_and_character_processing(dev_data_raw, dev_ground_truths, dev_sentences, split, test_data_raw,
                                                 test_sentences, tokenization_level, train_data_raw, to_idx=False)
    elif model_type == 'transformer':
        train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data = word_and_character_processing(dev_data_raw, dev_ground_truths, dev_sentences, split, test_data_raw,
                                             test_sentences, tokenization_level, train_data_raw)
        train_data = transformer_post_process(train_data)
        val_data = transformer_post_process(val_data)
        dev_data = transformer_post_process(dev_data)
        test_data = transformer_post_process(test_data)
        return train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data
    else:
        raise ValueError("Unsupported model_type. Choose 'n_gram' or 'transformer'.")


def transformer_post_process(train_data):
    input_sequences = []
    target_sequences = []
    for item in train_data:
        input_seq = item[:-1]
        target_seq = item[1:]
        input_sequences.append(torch.tensor(input_seq, dtype=torch.long))
        target_sequences.append(torch.tensor(target_seq, dtype=torch.long))

    return {
        'input_sequences': input_sequences,
        'target_sequences': target_sequences
    }


def word_and_character_processing(dev_data_raw, dev_ground_truths, dev_sentences, split, test_data_raw, test_sentences,
                                  tokenization_level, train_data_raw, to_idx=True):
    if tokenization_level == 'word':
        train_data_raw = [re.findall(r'\b\w+\b', line.lower()) for line in train_data_raw]
        dev_data_raw = [re.findall(r'\b\w+\b', line.lower()) for line in dev_data_raw]
        test_data_raw = [re.findall(r'\b\w+\b', line.lower()) for line in test_data_raw]
    elif tokenization_level == 'character':
        train_data_raw = [list(line.lower()) for line in train_data_raw]
        dev_data_raw = [list(line.lower()) for line in dev_data_raw]
        test_data_raw = [list(line.lower()) for line in test_data_raw]

    vocab = build_vocab(train_data_raw, tokenization_level)
    train_data_preprocessed = text_to_indices(train_data_raw, vocab, to_idx=to_idx)
    dev_data_preprocessed = text_to_indices(dev_data_raw, vocab, to_idx=to_idx)
    test_data_preprocessed = text_to_indices(test_data_raw, vocab, to_idx=to_idx)
    dev_wer_data = {}
    for doc_id in dev_ground_truths:
        if tokenization_level == 'word':
            tokens = [re.findall(r'\b\w+\b', line.lower()) for line in dev_sentences[doc_id]['sentences']]
        elif tokenization_level == 'character':
            tokens = [list(line.lower()) for line in dev_sentences[doc_id]['sentences']]
        else:
            raise ValueError("Unsupported tokenization level. Choose 'word' or 'character'.")
        tokens = text_to_indices(tokens, vocab, to_idx=to_idx)
        dev_wer_data[doc_id] = {
            'ground_truth': dev_ground_truths[doc_id]['sentence'],
            # 'ground_truth_tokens': tokens[dev_sentences[doc_id]['sentences'].index(dev_ground_truths[doc_id]['sentence'])],
            'sentences': dev_sentences[doc_id]['sentences'],
            'tokens': {
                dev_sentences[doc_id]['sentences'][i]: tokens[i] for i in range(len(tokens))
            },
            'acoustic_scores': {
                dev_sentences[doc_id]['sentences'][i]: dev_sentences[doc_id]['acoustic_scores'][i] for i in
                range(len(tokens))
            }
        }

    test_wer_data = {}
    for doc_id in test_sentences:
        if tokenization_level == 'word':
            tokens = [re.findall(r'\b\w+\b', line.lower()) for line in test_sentences[doc_id]['sentences']]
        elif tokenization_level == 'character':
            tokens = [list(line.lower()) for line in test_sentences[doc_id]['sentences']]
        else:
            raise ValueError("Unsupported tokenization level. Choose 'word' or 'character'.")
        tokens = text_to_indices(tokens, vocab, to_idx=to_idx)
        test_wer_data[doc_id] = {
            'sentences': test_sentences[doc_id]['sentences'],
            'tokens': {
                test_sentences[doc_id]['sentences'][i]: tokens[i] for i in range(len(tokens))
            },
            'acoustic_scores': {
                test_sentences[doc_id]['sentences'][i]: test_sentences[doc_id]['acoustic_scores'][i] for i in
                range(len(tokens))
            }
        }
    random.shuffle(train_data_preprocessed)
    train_data, val_data = train_data_preprocessed[
                           :int(len(train_data_preprocessed) * (1 - split))], train_data_preprocessed[
                                                                              int(len(train_data_preprocessed) * (
                                                                                      1 - split)):]
    dev_data = dev_data_preprocessed
    test_data = test_data_preprocessed
    print(
        f"Train size: {len(train_data)}, Val size: {len(val_data)}, Dev size: {len(dev_data)}, Test size: {len(test_data)}")
    return train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data


def n_gram_subword_processing(base_path, dev_data_raw, dev_ground_truths, dev_sentences, split, test_data_raw,
                              test_sentences, train_data_raw):
    print("Training subword-level tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    print(f'Using: {tokenizer.model}')
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", '<BOS>', '<EOS>'])
    # tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files=[
        os.path.join(base_path, 'treebank-sentences-train.txt'),
    ], trainer=trainer)
    print(f"Training complete. Vocab size: {tokenizer.get_vocab_size()}")
    print("Tokenizing data...")
    train_data_tokenized = tokenizer.encode_batch(train_data_raw)
    train_data_tokenized = [item.tokens for item in train_data_tokenized]
    dev_data_tokenized = tokenizer.encode_batch(dev_data_raw)
    dev_data_tokenized = [item.tokens for item in dev_data_tokenized]
    test_data_tokenized = tokenizer.encode_batch(test_data_raw)
    test_data_tokenized = [item.tokens for item in test_data_tokenized]
    print("Tokenization complete.")
    dev_wer_data = {}
    for doc_id in dev_ground_truths:
        tokens = tokenizer.encode_batch(dev_sentences[doc_id]['sentences'])
        tokens = [item.tokens for item in tokens]
        dev_wer_data[doc_id] = {
            'ground_truth': dev_ground_truths[doc_id]['sentence'],
            'sentences': dev_sentences[doc_id]['sentences'],
            'tokens': {
                dev_sentences[doc_id]['sentences'][i]: tokens[i] for i in range(len(tokens))
            },
            'acoustic_scores': {
                dev_sentences[doc_id]['sentences'][i]: dev_sentences[doc_id]['acoustic_scores'][i] for i in
                range(len(tokens))
            }
        }
    test_wer_data = {}
    for doc_id in test_sentences:
        tokens = tokenizer.encode_batch(test_sentences[doc_id]['sentences'])
        tokens = [item.tokens for item in tokens]

        test_wer_data[doc_id] = {
            'sentences': test_sentences[doc_id]['sentences'],
            'tokens': {
                test_sentences[doc_id]['sentences'][i]: tokens[i] for i in range(len(tokens))
            },
            'acoustic_scores': {
                test_sentences[doc_id]['sentences'][i]: test_sentences[doc_id]['acoustic_scores'][i] for i in
                range(len(tokens))
            }
        }
    random.shuffle(train_data_tokenized)
    train_data, val_data = train_data_tokenized[
                           :int(len(train_data_tokenized) * (1 - split))], train_data_tokenized[
                                                                           int(len(train_data_tokenized) * (
                                                                                   1 - split)):]
    dev_data = dev_data_tokenized
    test_data = test_data_tokenized
    print(
        f"Train size: {len(train_data)}, Val size: {len(val_data)}, Dev size: {len(dev_data)}, Test size: {len(test_data)}")
    return train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data
