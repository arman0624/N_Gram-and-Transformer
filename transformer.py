import os
import sys
import argparse
from typing import Dict, List
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from load_data import load_data, load_lookup_table
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("DEVICE: ", DEVICE)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("DEVICE: ", DEVICE)
    x = torch.ones(1, device=DEVICE)
    print(x)
else:
    print("MPS device not found.")


class CharacterLevelTransformer(nn.Module):
    name = "transformer"

    def __init__(self, num_layers: int, hidden_dim: int, num_heads: int,
                 ff_dim: int, dropout: float, vocab_size: int):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.padding_idx = vocab_size - 1
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=self.padding_idx)
        self.pos_embed = PositionalEncoding(hidden_dim, dropout)
        self.decoder = Decoder(num_layers, hidden_dim, num_heads, ff_dim, dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def log_probability(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor, base=np.e):
        """
        Computes the log-probabilities for the inputs in the given minibatch.

        Input:
            input_tokens (torch.Tensor): A tensor of shape (B, T), where B is the 
                                         batch-size and T is the input length. 
            target_tokens (torch.Tensor): A tensor of shape (B, T). For a given (i, j),
                                          target_tokens[i, j] should be the token following
                                          input_tokens[i, j]
        Output (torch.Tensor): A tensor of shape (B,) containing the log-probability for each
                               example in the minibatch
        """
        input_tokens = input_tokens.to(DEVICE)
        target_tokens = target_tokens.to(DEVICE)
        with torch.no_grad():
            output = self.forward(input_tokens)
            log_probs = F.log_softmax(output, dim=-1)
            log_probs = log_probs / math.log(base)
            log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
            log_probs = log_probs.masked_fill(target_tokens == self.padding_idx, 0)
            log_probs = log_probs.sum(dim=1)
            return log_probs

    def forward(self, model_input):
        # Perform the embedding
        embeds = self.embed(model_input) * math.sqrt(self.hidden_dim)
        embeds = self.pos_embed(embeds)

        # Pass through the decoder
        mask = construct_self_attn_mask(model_input)
        decoder_output = self.decoder(embeds, mask)
        output = self.lm_head(decoder_output)
        return output


def construct_self_attn_mask(x: torch.Tensor):
    """
    The output to this function should be a mask of shape
    (1, T, T). Indices that a token can attend to should be
    set to true.
    """

    T = x.size(1)
    all_ones = torch.ones(T, T).to(x.device)
    mask = torch.triu(all_ones, diagonal=1).bool()
    mask = ~mask
    mask = mask.unsqueeze(0)
    return mask.to(x.device)


class Decoder(nn.Module):

    def __init__(self, num_layers, hidden_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(num_heads, hidden_dim, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class TransformerBlock(nn.Module):

    def __init__(self, num_heads, hidden_dim, ff_dim, dropout):
        super().__init__()

        # Attention block
        self.attn_block = MultiHeadAttention(num_heads, hidden_dim, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Feedforward block
        self.mlp_block = TransformerMLP(hidden_dim, ff_dim, dropout)
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        # apply attention
        attn_output = self.attn_block(x, mask)
        attn_output = self.attn_dropout(attn_output)
        attn_output = self.attn_norm(x + attn_output)

        # apply feedforward
        ff_output = self.mlp_block(attn_output)
        ff_output = self.mlp_dropout(ff_output)
        ff_output = self.mlp_norm(attn_output + ff_output)

        return ff_output


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        self.h = num_heads
        self.qkv_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, query, key, value, mask):
        """
        There are three errors in this function to fix.
        """
        dot_products = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.qkv_dim)
        dot_products = dot_products.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(dot_products, dim=-1))
        return torch.matmul(attn, value)

    def forward(self, x, mask):

        mask = mask.unsqueeze(1)
        B = x.size(0)
        query = self.q_proj(x).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)
        key = self.k_proj(x).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)
        value = self.v_proj(x).view(B, -1, self.h, self.qkv_dim).transpose(1, 2)

        x = self.attention(query, key, value, mask)

        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.qkv_dim)
        return self.out_proj(x)


class TransformerMLP(nn.Module):

    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_dim, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encodings = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (- math.log(10000) / hidden_dim))
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        positional_encodings[:, 1::2] = torch.cos(position * div_term)
        positional_encodings = positional_encodings.unsqueeze(0)

        self.register_buffer('positional_encodings', positional_encodings, persistent=False)

    def forward(self, x):
        positional_encodings = self.positional_encodings[:, :x.size(1), :]
        x = x + positional_encodings
        return self.dropout(x)


from torch.utils.data import Dataset, DataLoader


class CharacterDataset(Dataset):
    def __init__(self, input_data, target_data):
        """
        Args:
            input_data (List[List[int]]): List of input sequences, where each sequence is a list of token indices.
            target_data (List[List[int]]): List of target sequences, where each sequence is a list of token indices.
        """
        assert len(input_data) == len(target_data), "Input and target data must have the same number of sequences"
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_seq = self.input_data[idx].clone().detach()
        target_seq = self.target_data[idx].clone().detach()
        return input_seq, target_seq


def create_data_loader(input_data, target_data, batch_size=32, shuffle=True):
    dataset = CharacterDataset(input_data, target_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    print("Data loader shape: ", len(data_loader))
    return data_loader


from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # Assuming 0 is your padding index
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)  # Assuming 0 is your padding index
    return padded_inputs, padded_targets


def train(model, train_data, val_data, dev_wer_data, loss_fct, optimizer, max_epochs=30):
    """
    Training loop for the transformer model. You may change the header as you see fit.
    """
    train_input_data = train_data['input_sequences']
    train_target_data = train_data['target_sequences']
    val_input_data = val_data['input_sequences']
    val_target_data = val_data['target_sequences']
    train_data_loader = create_data_loader(train_input_data, train_target_data, batch_size=32, shuffle=True)
    val_data_loader = create_data_loader(val_input_data, val_target_data, batch_size=32, shuffle=False)
    for epoch in range(max_epochs):
        print("Epoch: ", epoch)
        model.train()

        for inputs, targets in tqdm(train_data_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            output = model(inputs)

            # reshape
            output = output.view(-1, output.size(-1))

            # flatten the targets
            targets = targets.view(-1)

            loss = loss_fct(output, targets)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_data_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                output = model(inputs)
                output = output.view(-1, output.size(-1))
                targets = targets.view(-1)
                val_loss += loss_fct(output, targets).item()
        print(f'Epoch {epoch} val loss: {val_loss / len(val_data_loader)}')


def get_args():
    """
    You may freely add new command line arguments to this function.
    """
    parser = argparse.ArgumentParser(description='Transformer model')
    parser.add_argument('--num_layers', type=int, default=6,
                        help="How many transformer blocks to use")
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help="What is the transformer hidden dimension")
    parser.add_argument('--num_heads', type=int, default=8,
                        help="How many heads to use for Multihead Attention")
    parser.add_argument('--ff_dim', type=int, default=2048,
                        help="What is the intermediate dimension for the feedforward layer")
    parser.add_argument('--dropout_p', type=int, default=0.1,
                        help="The dropout probability to use")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument('--experiment_name', type=str, default='testing_')
    parser.add_argument('--num_samples', type=int, default=10,
                        help="How many samples should we get from our model??")
    parser.add_argument('--max_steps', type=int, default=40,
                        help="What should the maximum output length be?")

    args = parser.parse_args()
    return args


def main():
    # Get key arguments
    args = get_args()

    # Get the data
    tokenization_level = "character"
    model_type = "transformer"
    train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data = load_data(tokenization_level, model_type)

    # Initialize the transformer and train
    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    num_heads = args.num_heads
    ff_dim = args.ff_dim
    dropout_p = args.dropout_p
    id2item, item2id = load_lookup_table()
    vocab_size = len(item2id)
    model = CharacterLevelTransformer(num_layers, hidden_dim, num_heads, ff_dim,
                                      dropout_p, vocab_size).to(DEVICE)
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fct = nn.CrossEntropyLoss(ignore_index=vocab_size - 1)
    max_epochs = 10
    str_data = f"num_layers: {num_layers}, hidden_dim: {hidden_dim}, num_heads: {num_heads}, ff_dim: {ff_dim}, dropout_p: {dropout_p}, learning_rate: {learning_rate}, num_epochs: {max_epochs}"
    train(model, train_data, val_data, dev_wer_data, loss_fct, optimizer, max_epochs)

    # Evaluate model perplexity
    model.eval()
    val_perplexity = evaluate_perplexity(model, val_data)
    print(f'Model perplexity on the val set: {val_perplexity}')
    dev_perplexity = evaluate_perplexity(model, dev_data)
    print(f'Model perplexity on the dev set: {dev_perplexity}')
    test_perplexity = evaluate_perplexity(model, test_data)
    print(f'Model perplexity on the test set: {test_perplexity}')

    # Evaluate model WER
    experiment_name = args.experiment_name
    dev_wer_savepath = os.path.join('results', f'{experiment_name}transformer_dev_wer_predictions.csv')
    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath)
    dev_wer = compute_wer('data/wer_data/dev_ground_truths.csv', dev_wer_savepath)
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join('results', f'{experiment_name}transformer_test_wer_predictions.csv')
    print(str_data)
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath)


if __name__ == "__main__":
    main()
