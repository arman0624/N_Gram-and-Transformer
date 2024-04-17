## N_Gram and Transformer Model

I've built a word, subword, and character-level n-gram language model (LM), and a Transformer LM. 
I then evaluated them on a language modeling task and on an automatic speech recognition
(ASR) transcription reranking task.

A language model is a probability distribution over sequences of tokens.
I trained these language models on a WSJ corpus of about 1M words (from the Penn
treebank). These sentences have been speechified, for example translating “$” to “dollars”, and tokenized
for you. The WSJ dataset is split into training data (80%), dev data (10%) and test data (10%).

A set of ASR candidate transcriptions of given spoken sentences are included, taken from the HUB data set. 
Each spoken sentence has multiple candidate transcriptions and each candidate transcription is
accompanied by a pre-computed acoustic score, which represents the degree to which an acoustic model 
matched that transcription. The trained language model takes the acoustic scores to rank candidate sentences 
and choose the likeliest transcription. The HUB dataset has been filtered such that the candidate 
list always includes the correct transcription and never includes words unseen in the WSJ training data.

Perplexity: The perplexity of the WSJ test sentences is calculated. This number can be useful for
evaluation, but can be misleading. For instance, if there is a bug (and the probabilities do not sum
to 1), we get wonderful perplexities (but potentially terrible word error rates). If there are no 
bugs, higher order n-gram models and more sophisticated smoothing techniques will result in lower
perplexities. The WSJ test data includes unknown words (relatively to the training set). The language models 
treat all entirely unseen words as if they were a single UNK token. This means that, for example, a good
unigram model will actually assign a larger probability to each unknown word than to a known but rare
word - this is because the aggregate probability of the UNK event is large, even though each specific
unknown word itself may be rare. The models WSJ perplexity score will not strictly speaking be the
perplexity of the exact test sentences, but the UNK-ed test sentences (a lower number).

Word Error Rate (WER): The WER on the HUB recognition task is calculated. This implementation takes the 
candidate transcriptions and score each one with the language model, and combines those scores with the 
pre-computed acoustic scores. The best-scoring candidates are compared against the correct answers, and 
the WER is computed. The lower the WER score, the better it is. Note that the candidates are only so bad to 
begin with (the lists are pre-pruned n-best lists). As the complexity of the language model increases, 
it may be that lower perplexity will translate into a better WER, but not always.

# Run n-gram model

```shell
python n_gram.py --n 3 -t character -sm Linear_Interpolation 
```

# Run Transformer

```shell
python transformer.py
```

## Example commands

### Environment

It's highly recommended to use a virtual environment (e.g. conda, venv) for this assignment.

Example of virtual environment creation using conda:
```
conda create -n env_name python=3.10
conda activate env_name
python -m pip install -r requirements.txt
```

### Train and predict commands

```
python3 n_gram.py --n=10 --experiment_name=character --tokenization_level=character --smoothing_tech=Linear_Interpolation
python3 n_gram.py --n=10 --experiment_name=subword --tokenization_level=subword --smoothing_tech=Linear_Interpolation
python transformer.py --num_layers=4 --hidden_dim=512 --experiment_name=transformer
```

### Commands to run unittests

Ensure that your code passes the unittests before submitting it.
The commands can be run from the root directory of the project.
```
pytest tests/test_n_gram.py
pytest tests/test_transformer.py
```
