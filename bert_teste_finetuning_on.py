import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import re

# Caminho do CSV com os dados
caminho_csv = "C:/TCC2/instagram/usernames_com_nomes_1000.csv"

# Carregar os dados
df = pd.read_csv(caminho_csv)

# Normalizar os nomes removendo espaços
def normalizar_nome(name):
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)  # Remove caracteres inválidos, incluindo espaços
    return name

df["Name"] = df["Name"].astype(str).apply(normalizar_nome)
df["Username"] = df["Username"].astype(str).str.strip()

# Criar rótulos (1 para match, 0 para não-match) – Aqui podemos usar Levenshtein como referência
df["Label"] = df.apply(lambda row: 1 if row["Name"] in row["Username"] else 0, axis=1)

# Separar os dados em treino (80%) e teste (20%)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df[["Name", "Username"]].values.tolist(), df["Label"].values.tolist(), test_size=0.2, random_state=42
)

# Carregar o Tokenizador BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Criar Dataset Customizado para BERT
class UserProfileDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        name, username = self.texts[idx]
        inputs = tokenizer(name, username, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Criar DataLoader para treino e teste
train_dataset = UserProfileDataset(train_texts, train_labels)
test_dataset = UserProfileDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Carregar o Modelo BERT pré-treinado para classificação
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# Definir otimizador e função de perda
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Treinar o Modelo BERT
def train_model(model, train_loader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"🔵 Época {epoch + 1} - Perda Média: {total_loss / len(train_loader)}")

# Rodar Treinamento
train_model(model, train_loader, optimizer, criterion, epochs=3)

# Após o treinamento do modelo
model.save_pretrained("meu_modelo_treinado")
tokenizer.save_pretrained("meu_modelo_treinado")

print("✅ Modelo treinado salvo com sucesso!")

# Avaliação do Modelo
def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    print(f"✅ Acurácia: {acc:.4f}, F1-Score: {f1:.4f}")

# Rodar Avaliação
evaluate_model(model, test_loader)
