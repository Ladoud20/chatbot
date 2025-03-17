import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


medical_texts = {
    "Diabetes Typ 2 Hypertonie Herzinsuffizienz": [14, 25],
    "Asthma bronchiale COPD Lungenfibrose": [17, 22],
    "Gastroenteritis Morbus Crohn Colitis ulcerosa": [15, 28],
    "Schlaganfall TIA Hirnblutung Aneurysma": [12, 16, 28],
    "Herzinfarkt KHK Angina pectoris Herzinsuffizienz": [11, 15, 31],
    "Zerebrale Ischämie Chronische Bronchitis": [18],
    # No segmentation
    "Chronische Bronchitis": [],
    "Zerebrale Ischämie": [],
    "Hypertensive Krise": [],
    "Leberzirrhose Alkoholabusus": [],
    "Malignes Melanom": [],
    "Periphere arterielle Verschlusskrankheit": [],
    "Diabetische Retinopathie": [],
}

model_path = "../models/gerMedBert"

# Step 1: Load German Sentence Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)


# Step 2: Data Preparation
class SentenceTextSegmentationDataset(Dataset):
    def __init__(self, medical_texts, context_size=5, max_length=10):
        self.data = []
        self.labels = []
        self.max_length = max_length

        for text, segmentation_points in medical_texts.items():
            text_tokens = list(text)  # Keep text as list of characters

            for i, char in enumerate(text_tokens):
                if char == " ":  # Only consider spaces
                    left_context = "".join(text_tokens[max(0, i - context_size): i])
                    right_context = "".join(text_tokens[i + 1: min(len(text), i + 1 + context_size)])
                    context_window = left_context + " " + right_context

                    # Tokenize with BERT
                    encoding = tokenizer(
                        context_window,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    )

                    self.data.append((encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)))
                    self.labels.append(1 if i in segmentation_points else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, attention_mask = self.data[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids.long(), attention_mask.long(), label

# Create dataset & dataloader
dataset = SentenceTextSegmentationDataset(medical_texts)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Step 3: Define BERT Model
class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.sentence = AutoModel.from_pretrained(model_path)
        self.fc = nn.Linear(self.sentence.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.sentence(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Use CLS token embedding
        x = self.fc(cls_embedding)
        return self.sigmoid(x).squeeze()

# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegmentationModel().to(device)

# Step 4: Training Setup
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Step 5: Training Loop
def train_sentence_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, batch_y in dataloader:
            input_ids, attention_mask, batch_y = (
                input_ids.to(device),
                attention_mask.to(device),
                batch_y.to(device),
            )

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# Train the model
train_sentence_model(model, dataloader, criterion, optimizer, epochs=10)

# Step 6: Prediction Function
def predict_segmentation(text, model, context_size=5, max_length=10):
    model.eval()
    segmentations = []
    text_tokens = list(text)

    for i, char in enumerate(text_tokens):
        if char == " ":
            left_context = "".join(text_tokens[max(0, i - context_size): i])
            right_context = "".join(text_tokens[i + 1: min(len(text), i + 1 + context_size)])
            context_window = left_context + " " + right_context

            encoding = tokenizer(
                context_window,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].squeeze(0).unsqueeze(0).to(device)
            attention_mask = encoding["attention_mask"].squeeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(input_ids=input_ids, attention_mask=attention_mask).item()

            if prediction > 0.5:
                segmentations.append(i)

    return segmentations

# Example usage
test_text = "Diabetes Typ 2 Hypertonie Herzinsuffizienz"
predictions = predict_segmentation(test_text, model)
print("Segmentations at indices:", predictions)

