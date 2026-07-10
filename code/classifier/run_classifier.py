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

MODEL_PATH = "./models/mobilenetv2_100_fold_0_374"

IMAGE_FOLDER = "./data/train_rain"

CSV_PATH = "./data/test_rain.csv"

OUTPUT_CSV = "predictions.csv"

IMG_SIZE = 128

DEVICE = "cpu" # mudar para cuda quando for fazer testes maiores

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

for k, v in state_dict.items():
    print(k, v.shape)

# procura automaticamente a última camada
last_key = None

for key in state_dict.keys():
    if key.endswith("weight"):
        last_key = key

print("Última camada encontrada:", last_key)

num_classes = state_dict[last_key].shape[0]

print("Número de classes:", num_classes)
"""
# ===========================
# Modelo
# ===========================

model = timm.create_model(
    "mobilenetv2_100",
    pretrained=False,
    num_classes=num_classes
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

with torch.no_grad():

    for _, row in tqdm(df.iterrows(), total=len(df)):

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
            "prediction": pred,
            "confidence": confidence
        }

        # salva todas as probabilidades
        for i, p in enumerate(probs.cpu().numpy()[0]):
            result[f"prob_{i}"] = float(p)

        results.append(result)

# ===========================
# Salvar CSV
# ===========================

pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

print()
print("Inferência concluída.")
print("Resultado salvo em:", OUTPUT_CSV)
"""