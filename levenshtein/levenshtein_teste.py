import pandas as pd
import matplotlib.pyplot as plt
from Levenshtein import distance
import numpy as np

# Caminho do CSV gerado anteriormente
caminho_csv = "C:/TCC2/instagram/usernames_com_nomes_1000.csv"

# Carregar os dados do CSV
df = pd.read_csv(caminho_csv)

# Garantir que os dados são strings e não possuem espaços extras
df["Name"] = df["Name"].astype(str).str.strip()
df["Username"] = df["Username"].astype(str).str.strip()

# Parâmetros
limiar_levenshtein = 8  # Definição da similaridade
tamanho_amostra = 50  # Cada amostra terá 50 perfis

# Criar listas para armazenar os resultados por grupo de 50 perfis
resultados_corretos = []
falsos_positivos = []
grupos = []  # Número do grupo de amostras

# Processar os perfis em blocos de 50
for i in range(0, len(df), tamanho_amostra):
    grupo_df = df.iloc[i:i + tamanho_amostra]  # Pegamos 50 perfis por vez
    acertos = 0
    erros = 0

    for _, row in grupo_df.iterrows():
        nome = row["Name"]
        username = row["Username"]

        # Calcular a distância de Levenshtein
        dist = distance(nome, username)

        if dist <= limiar_levenshtein:
            acertos += 1  # Match correto
        else:
            erros += 1  # Falso positivo

    # Adicionamos os resultados do grupo atual
    resultados_corretos.append(acertos)
    falsos_positivos.append(erros)
    grupos.append(f"A{len(grupos) + 1}")  # Nomeamos cada grupo

# Criar DataFrame com os resultados
df_resultados = pd.DataFrame({
    "Amostra": grupos,
    "Resultados Corretos": resultados_corretos,
    "Falsos Positivos": falsos_positivos
})

# Gerar gráfico de barras agrupado por amostra
plt.figure(figsize=(12, 5))
x = np.arange(len(grupos))  # Posições das barras
largura = 0.4  # Largura das barras

# Criamos as barras lado a lado
plt.bar(x - largura/2, df_resultados["Resultados Corretos"],
        width=largura, color="blue", label="Resultados Corretos")
plt.bar(x + largura/2, df_resultados["Falsos Positivos"],
        width=largura, color="red", label="Falsos Positivos")

# Configuração do gráfico
plt.xlabel("Amostras (50 perfis cada)")
plt.ylabel("Quantidade")
plt.title("Resultados Corretos vs Falsos Positivos por Amostra")
plt.xticks(x, grupos, rotation=45)  # Rotaciona os rótulos do eixo X
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Mostrar gráfico
plt.show()
