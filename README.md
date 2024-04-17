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
