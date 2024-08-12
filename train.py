import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentiment_model import SentimentTransformer
from utils import load_data
import time



embedding_dim =128
num_heads = 8
num_layers = 4
output_dim = 2
batch_size=64
num_epochs = 3
learning_rate = 0.001




device  = torch.device('cude' if torch.cuda.is_available() else 'cpu')


train_iter, test_iter, tokenizer, vocab = load_data()

def collate_fn(batch):
    max_seq_len = 512 
    labels = torch.tensor([label for label, _ in batch])

    
    labels = torch.where(labels > 1, torch.tensor(1), labels) 

    texts = [torch.tensor([vocab[token] for token in tokenizer(text)[:max_seq_len]]) for _, text in batch]
    texts = nn.utils.rnn.pad_sequence(texts, padding_value=vocab['<pad>'], batch_first=True)
    return texts, labels




train_loader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = SentimentTransformer(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    output_dim=output_dim
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train(model, loader, criterion, optimizer):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return epoch_loss / len(loader), accuracy


def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return epoch_loss / len(loader), accuracy


best_valid_loss = float('inf')

for epoch in range(num_epochs):
    start_time = time.time()

    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    valid_loss, valid_acc = evaluate(model, test_loader, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc:.2f}%')

    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'models/sentiment_transformer_best.pth')