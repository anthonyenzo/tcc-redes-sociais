# Ler apenas os primeiros bytes do arquivo para verificar estrutura
with open("C:/TCC2/instagram/posts.json", "r", encoding="utf-8") as json_file:
    preview = json_file.read(5000)  # Lê apenas os primeiros 2000 bytes
    print(preview)
