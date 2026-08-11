import csv

numero = "202112"

with open(f"test_rain_{numero}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "label"])

    for i in range(520):
        writer.writerow([f"{numero}_1_{i}.jpeg", 0])

print(f"Arquivo test_rain_{numero}.csv criado com sucesso!")

