import ijson
import pandas as pd

# Caminho do arquivo JSON
caminho_json = "C:/TCC2/instagram/posts.json"

# Número máximo de registros a serem extraídos
max_registros = 1000  

# Usamos um conjunto (set) para garantir perfis únicos
dados = set()  

try:
    with open(caminho_json, "r", encoding="utf-8") as json_file:
        print(f"🔄 Extraindo até {max_registros} perfis únicos do JSON...")

        # Usando ijson para iterar sobre os objetos dentro do array JSON
        for registro in ijson.items(json_file, "item"):
            if isinstance(registro, dict) and "account" in registro:
                nome = registro["account"].get("name", "N/A").strip()  
                username = registro["account"].get("handle", "N/A").strip()

                # Adiciona ao conjunto para garantir que sejam únicos
                dados.add((nome, username))

            # Para a extração ao atingir o limite de 1000 perfis
            if len(dados) >= max_registros:
                break

except FileNotFoundError:
    print("❌ ERRO: O arquivo JSON não foi encontrado!")
except Exception as e:
    print(f"❌ ERRO INESPERADO: {e}")

# Converter para DataFrame e salvar no CSV
if dados:
    df = pd.DataFrame(list(dados), columns=["Name", "Username"])

    # Salvar CSV final com os 1000 perfis únicos
    caminho_csv = "C:/TCC2/instagram/usernames_com_nomes_1000.csv"
    df.to_csv(caminho_csv, index=False, encoding="utf-8")

    print(f"✅ Os primeiros {max_registros} perfis únicos foram extraídos e salvos em {caminho_csv}!")
else:
    print("⚠ Nenhum dado encontrado! Verifique a estrutura do JSON.")
