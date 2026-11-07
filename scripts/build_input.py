import os
import pandas as pd
a = pd.read_csv("data/raw/vn_news_label_0_real.csv")
b = pd.read_csv("data/raw/vn_news_label_1_fake.csv")
df = pd.concat([a, b], ignore_index=True)[["text", "label"]]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
os.makedirs("data/interim", exist_ok=True)
df.to_csv("data/interim/input_all.csv", index=False, encoding="utf-8")
print("input_all.csv:", len(df), df["label"].value_counts().to_dict())
