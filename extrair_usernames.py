import ijson
import pandas as pd

# Caminho do arquivo JSON
caminho_json = "C:/TCC2/instagram/posts.json"

# Número máximo de usernames para teste (ajustável)
max_registros = 500  

# Lista para armazenar os usernames
usernames = []

try:
    with open(caminho_json, "r", encoding="utf-8") as json_file:
        print("🔄 Processando JSON sem carregar tudo na memória...")

        # Usando ijson para iterar sobre os objetos dentro do array JSON
        for registro in ijson.items(json_file, "item"):
            if isinstance(registro, dict) and "account" in registro:
                if "handle" in registro["account"]:
                    usernames.append(registro["account"]["handle"])

            if len(usernames) >= max_registros:  # Para após atingir o limite desejado
                break

except FileNotFoundError:
    print("❌ ERRO: O arquivo JSON não foi encontrado!")
except Exception as e:
    print(f"❌ ERRO INESPERADO: {e}")

# Salvar usernames extraídos
if usernames:
    df_usernames = pd.DataFrame(usernames, columns=["Username"])
    caminho_csv = "C:/TCC2/instagram/usernames_teste.csv"
    df_usernames.to_csv(caminho_csv, index=False, encoding="utf-8")
    print(f"✅ Os primeiros {max_registros} usernames foram extraídos e salvos em {caminho_csv}!")
else:
    print("⚠ Nenhum username encontrado! Verifique a estrutura do JSON.")
