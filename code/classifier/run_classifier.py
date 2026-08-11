import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import timm
from tqdm import tqdm

# ===========================
# Configurações
# ===========================

numero = "202112" #para testar os datasets de rain

MODEL_PATH = "../../models/mobilenetv2_100_fold_0_374"

IMAGE_FOLDER = f"../../data/test_rain_{numero}"

CSV_PATH = f"../../data/csv/test_rain_{numero}.csv"

OUTPUT_CSV = f"./predictions/predictions_rain_{numero}.csv"

IMG_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================
# Transformações
# ===========================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===========================
# Descobrir número de classes
# ===========================

print("Carregando pesos...")

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

""" DEBUG
for k, v in state_dict.items():
    print(k, v.shape)
"""

# procura automaticamente a última camada
last_key = None

for key in state_dict.keys():
    if key.endswith("weight"):
        last_key = key

print("Última camada encontrada:", last_key)

num_classes = state_dict[last_key].shape[0]

print("Número de classes:", num_classes)

# ===========================
# Modelo
# ===========================

class ImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()

        self.model = timm.create_model(
            model_arch,
            pretrained=pretrained,
            num_classes=n_class
        )

    def forward(self, x):
        return self.model(x)


model = ImgClassifier(
    "mobilenetv2_100",
    num_classes,
    pretrained=False
)

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("Modelo carregado com sucesso!")

# ===========================
# Lista de imagens
# ===========================

df = pd.read_csv(CSV_PATH)

results = []

# ===========================
# Inferência
# ===========================

sucessos = 0

with torch.no_grad():

    for image_number, row in tqdm(df.iterrows(), total=len(df)):

        filename = row["image_id"]

        path = os.path.join(IMAGE_FOLDER, filename)

        image = Image.open(path).convert("RGB")

        x = transform(image)

        x = x.unsqueeze(0).to(DEVICE)

        logits = model(x)

        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()

        confidence = probs[0, pred].item()

        result = {
            "image_id": filename,
            "image_number": image_number,
            "prediction": pred,
            "confidence": confidence,
            "correct_if_number_is_label": pred == image_number
        }

        if pred == image_number:
            sucessos += 1

        # salva todas as probabilidades
        for i, p in enumerate(probs.cpu().numpy()[0]):
            result[f"prob_{i}"] = float(p)

        results.append(result)

print(f"Acertos: {sucessos}\nErros: {num_classes - sucessos}\nTaxa de sucesso: {sucessos / num_classes}")

# ===========================
# Salvar CSV
# ===========================

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

print()
print("Inferência concluída.")
print("Resultado salvo em:", OUTPUT_CSV)
