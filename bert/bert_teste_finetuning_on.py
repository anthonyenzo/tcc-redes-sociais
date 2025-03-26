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

# Criar rótulos
df["Label"] = df.apply(lambda row: 1 if row["Name"] in row["Username"] else 0, axis=1)

# Separar dados
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df[["Name", "Username"]].values.tolist(), df["Label"].values.tolist(), test_size=0.2, random_state=42
)

# Tokenizador BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset personalizado
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

# Dataloaders
train_loader = DataLoader(UserProfileDataset(train_texts, train_labels), batch_size=8, shuffle=True)
test_loader = DataLoader(UserProfileDataset(test_texts, test_labels), batch_size=8, shuffle=False)

# Modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# Otimizador e perda
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Função de avaliação interna
def avaliar(model, dataloader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=mask)
            pred = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(pred)
            truths.extend(labels.cpu().numpy())
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds)
    return acc, f1

# Treinamento com early stopping
def treinar(model, train_loader, test_loader, optimizer, criterion, epochs=30, paciencia=5):
    melhor_perda = float("inf")
    sem_melhora = 0

    for epoca in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        perda_media = total_loss / len(train_loader)
        acc, f1 = avaliar(model, test_loader)

        print(f"🟦 Época {epoca} - Loss: {perda_media:.4f} | Acurácia: {acc:.4f}, F1-Score: {f1:.4f}")

        # Early stopping
        if perda_media < melhor_perda:
            melhor_perda = perda_media
            sem_melhora = 0
            model.save_pretrained("modelo_30epocas_treinado")
            tokenizer.save_pretrained("modelo_30epocas_treinado")
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                print("🛑 Early stopping ativado.")
                break

# Executar
treinar(model, train_loader, test_loader, optimizer, criterion, epochs=30)

print("✅ Modelo treinado salvo com sucesso (early stopping ou conclusão).")
