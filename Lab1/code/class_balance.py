import pandas as pd


data = pd.read_csv("A29.csv", encoding="utf-8-sig")
counts = data["class"].value_counts().sort_index()

for class_name, count in counts.items():
    print(f"{class_name}: {count} ({100 * count / len(data):.2f}%)")
