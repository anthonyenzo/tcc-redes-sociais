import bson
import json

# Caminho do arquivo BSON
caminho_bson = "C:/TCC2/instagram/posts.bson"

# Ler o arquivo BSON
with open(caminho_bson, "rb") as file:
    data = bson.decode_all(file.read())  # Decodifica BSON para lista de dicionários

# Função para converter ObjectId para String
def converter_objectid(obj):
    if isinstance(obj, bson.ObjectId):
        return str(obj)  # Converte ObjectId para string
    raise TypeError(f"Tipo {type(obj)} não é serializável para JSON")

# Salvar como JSON convertendo ObjectId para string
caminho_json = "C:/TCC2/instagram/posts.json"
with open(caminho_json, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=4, ensure_ascii=False, default=converter_objectid)

print(f"Arquivo convertido e salvo como JSON: {caminho_json}")
