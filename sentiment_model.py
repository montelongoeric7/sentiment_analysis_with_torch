import torch 
import torch.nn as nn
import torch.nn.functional as F



class SentimentTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, output_dim, max_len=512, dropout=0.1):
        super(SentimentTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
        self.transformer = nn.Transformer(embedding_dim, num_heads, num_layers, dropout=dropout)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
    
    
    def forward(self, x):
            seq_length = x.size(1)
            positional_encodings = self.positional_encoding[:, :seq_length, :]
            embedded = self.embedding(x) + positional_encodings
            embedded = self.dropout(embedded)
            embedded = embedded.permute(1, 0, 2)
            src_mask = self._generate_square_subsequent_mask(seq_length).to(x.device)
            transformer_output = self.transformer(embedded, embedded, src_mask=src_mask)
            pooled_output = transformer_output.mean(dim=0)
            output = self.fc(pooled_output)
            
            return output

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask    