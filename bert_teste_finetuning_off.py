import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# Carregar o modelo BERT pré-treinado (sem fine-tuning)
model = SentenceTransformer("paraphrase-mpnet-base-v2")

# Caminho do CSV com os dados
caminho_csv = "C:/TCC2/instagram/usernames_com_nomes_1000.csv"

# Carregar os dados
df = pd.read_csv(caminho_csv)
df["Name"] = df["Name"].astype(str).str.strip()
df["Username"] = df["Username"].astype(str).str.strip()

# Ajustar o threshold de similaridade
threshold = 0.88  # Ajuste conforme necessário
tamanho_amostra = 50  # Cada amostra terá 50 perfis

# Criar listas para armazenar os resultados por grupo
resultados_corretos = []
falsos_positivos = []
grupos = []

# Processar os perfis em blocos de 50
for i in range(0, len(df), tamanho_amostra):
    grupo_df = df.iloc[i:i + tamanho_amostra]
    acertos = 0
    erros = 0
    
    for _, row in grupo_df.iterrows():
        nome = row["Name"]
        username = row["Username"]
        
        # Gerar embeddings com o BERT pré-treinado
        embedding_nome = model.encode(nome, convert_to_tensor=True)
        embedding_username = model.encode(username, convert_to_tensor=True)
        
        # Calcular similaridade coseno
        similarity = util.pytorch_cos_sim(embedding_nome, embedding_username).item()
        
        if similarity >= threshold:
            acertos += 1  # Similaridade acima do threshold → Match correto
        else:
            erros += 1  # Similaridade abaixo do threshold → Falso positivo
    
    resultados_corretos.append(acertos)
    falsos_positivos.append(erros)
    grupos.append(f"A{len(grupos) + 1}")

# Criar DataFrame com os resultados
df_resultados = pd.DataFrame({
    "Amostra": grupos,
    "Resultados Corretos": resultados_corretos,
    "Falsos Positivos": falsos_positivos
})

# Gerar gráfico de barras agrupado por amostra
plt.figure(figsize=(12, 5))
x = np.arange(len(grupos))
largura = 0.4

plt.bar(x - largura/2, df_resultados["Resultados Corretos"],
        width=largura, color="blue", label="Resultados Corretos")
plt.bar(x + largura/2, df_resultados["Falsos Positivos"],
        width=largura, color="red", label="Falsos Positivos")

# Configuração do gráfico
plt.xlabel("Amostras (50 perfis cada)")
plt.ylabel("Quantidade")
plt.title("Resultados Corretos vs Falsos Positivos por Amostra (BERT Sem Fine-Tuning)")
plt.xticks(x, grupos, rotation=45)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()
