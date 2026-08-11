import shutil
from pathlib import Path


def filtrar_e_copiar_fotos(pasta_origem, pasta_destino, termo_busca):
    # Converte os caminhos para objetos Path
    origem = Path(pasta_origem)
    destino = Path(pasta_destino)

    # Cria a pasta de destino caso ela ainda não exista
    destino.mkdir(parents=True, exist_ok=True)

    contador = 0

    # Percorre todos os arquivos que terminam com .jpeg ou .jpg (case insensitive)
    for arquivo in origem.iterdir():
        if arquivo.is_file() and arquivo.suffix.lower() in [".jpeg", ".jpg"]:
            # Verifica se o termo buscado está no nome do arquivo
            if termo_busca in arquivo.name:
                # Copia o arquivo para a nova pasta
                shutil.copy2(arquivo, destino / arquivo.name)
                print(f"Copiado: {arquivo.name}")
                contador += 1

    print(f"\nConcluído! Total de {contador} foto(s) copiada(s) para '{destino}'.")


# ==========================================
# CONFIGURAÇÕES
# ==========================================
busca = str(input("Insira o termo que filtra as fotos na pasta train_rain: ")).strip()

PASTA_ORIGEM = "./train_rain"
PASTA_DESTINO = f"./test_rain_{busca}"
TERMO_BUSCA = busca  # Altere para a palavra/termo que você procura

# Executa a função
filtrar_e_copiar_fotos(PASTA_ORIGEM, PASTA_DESTINO, TERMO_BUSCA)
