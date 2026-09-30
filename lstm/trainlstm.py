import re
import pandas as pd
import torch
import torch.nn as nn

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("IMDB_dataset.csv")

print("Dataset shape:", data.shape)
print(data.head())


# ==========================================
# 2. LABEL ENCODING
# ==========================================

data["sentiment"] = data["sentiment"].map({
    "negative": 0,
    "positive": 1
})


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

train_reviews, test_reviews, train_labels, test_labels = train_test_split(
    data["review"],
    data["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=data["sentiment"]
)

print("\nTraining samples:", len(train_reviews))
print("Testing samples:", len(test_reviews))


# ==========================================
# 4. TOKENIZATION
# ==========================================

def tokenize(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.split()


# ==========================================
# 5. BUILD VOCABULARY
# ==========================================

counter = Counter()

for review in train_reviews:
    tokens = tokenize(review)
    counter.update(tokens)


word_to_id = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word in counter:
    word_to_id[word] = len(word_to_id)


print("\nVocabulary size:", len(word_to_id))


# ==========================================
# 6. ENCODE REVIEW
# ==========================================

def encode_review(review):
    tokens = tokenize(review)

    return [
        word_to_id.get(
            token,
            word_to_id["<UNK>"]
        )
        for token in tokens
    ]


# ==========================================
# 7. DATASET CLASS
# ==========================================

class IMDBDataset(Dataset):

    def __init__(self, reviews, labels, max_length=200):

        self.reviews = reviews.tolist()
        self.labels = labels.tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):

        review = encode_review(
            self.reviews[index]
        )

        # Truncate long reviews
        review = review[:self.max_length]

        # Pad short reviews
        if len(review) < self.max_length:

            review += [
                word_to_id["<PAD>"]
            ] * (
                self.max_length - len(review)
            )

        review = torch.tensor(
            review,
            dtype=torch.long
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return review, label


# ==========================================
# 8. CREATE DATASETS
# ==========================================

train_dataset = IMDBDataset(
    train_reviews,
    train_labels,
    max_length=200
)

test_dataset = IMDBDataset(
    test_reviews,
    test_labels,
    max_length=200
)


# ==========================================
# 9. CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ==========================================
# 10. LSTM MODEL
# ==========================================

class SentimentLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_size=128
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        # Token IDs → embeddings
        x = self.embedding(x)

        # Embeddings → LSTM
        output, (hidden, cell) = self.lstm(x)

        # Final hidden state
        final_hidden = hidden[-1]

        # Classification
        output = self.fc(final_hidden)

        return output.squeeze(1)


# ==========================================
# 11. DEVICE
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:", device)


# ==========================================
# 12. CREATE MODEL
# ==========================================

model = SentimentLSTM(
    vocab_size=len(word_to_id),
    embedding_dim=128,
    hidden_size=128
)

model = model.to(device)


# ==========================================
# 13. LOSS FUNCTION
# ==========================================

loss_function = nn.BCEWithLogitsLoss()


# ==========================================
# 14. OPTIMIZER
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 15. TRAINING
# ==========================================

epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for reviews, labels in train_loader:

        reviews = reviews.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(reviews)

        # Calculate loss
        loss = loss_function(
            outputs,
            labels
        )

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Convert logits → probabilities
        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Loss: {total_loss / len(train_loader):.4f} "
        f"Accuracy: {accuracy:.4f}"
    )


# ==========================================
# 16. TESTING
# ==========================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for reviews, labels in test_loader:

        reviews = reviews.to(device)
        labels = labels.to(device)

        outputs = model(reviews)

        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


test_accuracy = correct / total

print(
    "\nTest Accuracy:",
    test_accuracy
)


# ==========================================
# 17. CUSTOM PREDICTION
# ==========================================

def predict_sentiment(review):

    model.eval()

    encoded = encode_review(review)

    encoded = encoded[:200]

    if len(encoded) < 200:

        encoded += [
            word_to_id["<PAD>"]
        ] * (
            200 - len(encoded)
        )

    tensor = torch.tensor(
        [encoded],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(
            output
        ).item()

    if probability >= 0.5:

        sentiment = "Positive"

    else:

        sentiment = "Negative"

    confidence = (
        probability
        if probability >= 0.5
        else 1 - probability
    )

    return sentiment, confidence


# ==========================================
# 18. SAMPLE PREDICTIONS
# ==========================================

reviews = [

    "This movie was absolutely amazing. "
    "I loved every minute of it.",

    "The movie was boring and a complete "
    "waste of time.",

    "Fantastic acting and an excellent story."

]


print("\nSample predictions:")

for review in reviews:

    sentiment, confidence = predict_sentiment(
        review
    )

    print("\nReview:", review)
    print("Prediction:", sentiment)
    print(
        "Confidence:",
        round(confidence * 100, 2),
        "%"
    )


# ==========================================
# 19. SAVE MODEL
# ==========================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "word_to_id": word_to_id
    },
    "lstm_imdb.pth"
)

print("\nModel saved as lstm_imdb.pth")