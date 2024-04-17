from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.pre_tokenizers import Whitespace

# Initialize a basic word-level tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]",))
tokenizer.pre_tokenizer = Whitespace()

# Example usage
example = "Tokenization is essential for NLP."
output = tokenizer.encode(example)
print(output.tokens)


