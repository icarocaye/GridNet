import glob         # Procurar por arquivos com chaves coringas (*.jpg, *.png ...)
import random
import os           # os.path.join usado para construir caminhos de diretório

from torch.utils.data import Dataset            # Classe para criar um dataset customizado
from PIL import Image                           # Permite abrir imagens
import torchvision.transforms as transforms     # Transformações de imagens (resize, crop ...)


# Converter imagens em outros esquemas de cor para RGB
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


# Classe do dataset de imagens (herda Dataset)
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)    # Junta todas as transformações numa sequência
        self.unaligned = unaligned                          # Se for unaligned, não há correspondência exata do local entre o par imagem aérea [n] de satélite [n] (lembrando que a ideia do CycleGAN é não decorar a correspondência)

        # Pega todos os arquivos em root/mode/256a/ (Domínio A)
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/256a" % mode) + "/*.*")) #need change (comentário original)
        # Pega todos os arquivos em root/mode/256a/ (Domínio B)
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/256b" % mode) + "/*.*"))

    # Retorna uma amostra do dataset, com base no index
    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])   # Carrega a imagem A no index ajustado para caso len(A) != len(B)

        # Aqui entra a questão do unaligned. Se for unaligned, pega uma imagem B aleatória, senão pega a imagem B correspondente
        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb (comentário original)
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        # Transforma as imagens em tensores
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
