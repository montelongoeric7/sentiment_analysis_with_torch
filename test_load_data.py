
from utils import load_data

train_iter, test_iter, tokenizer, vocab = load_data()

print("Vocabulary size:", len(vocab))
print("First 10 tokens in the vocabulary:", vocab.get_itos()[:10])


sample_text = next(iter(train_iter))[1]
tokens = tokenizer(sample_text)
print("\nSample text:", sample_text)
print("Tokenized text:", tokens)
print("Token indices:", [vocab[token] for token in tokens])
