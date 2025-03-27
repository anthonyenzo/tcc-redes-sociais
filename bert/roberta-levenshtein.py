import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import re
from Levenshtein import distance  # Importando Levenshtein para calcular a distância

# Caminho do CSV com os dados
caminho_csv = "C:/TCC2/data/usernames_com_nomes_1000.csv"

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

# Carregar o Tokenizador Roberta
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Criar Dataset Customizado para Roberta
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

# Carregar o Modelo Roberta pré-treinado para classificação
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

# Definir otimizador e função de perda
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Treinar o Modelo Roberta com avaliação por época
def train_model(model, train_loader, test_loader, optimizer, criterion, epochs=30):
    for epoch in range(epochs):
        model.train()
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

        # Avaliação por época
        model.eval()
        predictions, true_labels = [], []
        levenshtein_distances = []  # Lista para armazenar as distâncias de Levenshtein
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
                
                # Calcular distâncias de Levenshtein
                for i in range(len(batch["input_ids"])):
                    name = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
                    username = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True)
                    dist = distance(name, username)
                    levenshtein_distances.append(dist)

        acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        avg_levenshtein_distance = sum(levenshtein_distances) / len(levenshtein_distances)  # Média das distâncias de Levenshtein
        print(f"🔵 Época {epoch + 1} - Loss: {total_loss / len(train_loader):.4f} | Acurácia: {acc:.4f}, F1-Score: {f1:.4f}, Média Distância Levenshtein: {avg_levenshtein_distance:.4f}")

# Rodar Treinamento
train_model(model, train_loader, test_loader, optimizer, criterion, epochs=30)

# Salvar o modelo e tokenizador
model.save_pretrained("modelo_roberta_treinado")
tokenizer.save_pretrained("modelo_roberta_treinado")

print("✅ Modelo RoBERTa treinado salvo com sucesso!")