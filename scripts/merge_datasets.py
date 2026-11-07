import json, os, glob, sys
import pandas as pd

ROOT = "Du_lieu_Da_Tai"

rp = os.path.join(ROOT, "ReINTEL", "train-00000-of-00001.parquet")
if not os.path.exists(rp):
    sys.exit(f"Không thấy {rp}")
r = pd.read_parquet(rp, columns=["post_message", "label"]).rename(columns={"post_message": "text"})
r["source"] = "ReINTEL"

rows = []
for name, label in (("Fake", 1), ("Real", 0)):
    for p in glob.glob(os.path.join(ROOT, "VFND", "Fake_Real_Dataset", name, "*", "*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print("Bỏ qua file lỗi:", p, e); continue
        text = (d.get("text") or "").strip()
        title = (d.get("title") or "").strip()
        if title and text:
            text = title + ". " + text
        rows.append({"text": text, "label": label, "source": "VFND"})
v = pd.DataFrame(rows)

df = pd.concat([r, v], ignore_index=True)
df["text"] = df["text"].astype(str).str.strip()
df = df[(df["text"].str.len() > 0) & (df["text"] != "None")].dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df = df[df["label"].isin([0, 1])].drop_duplicates(subset="text")

print("Theo nguồn:"); print(df.groupby(["source", "label"]).size())
print("Tổng:", len(df), df["label"].value_counts().to_dict())
print("Độ dài ký tự:"); print(df["text"].str.len().describe().round(0))

os.makedirs("data/raw", exist_ok=True)
df[df.label == 0][["text", "label", "source"]].to_csv("data/raw/vn_news_label_0_real.csv", index=False, encoding="utf-8")
df[df.label == 1][["text", "label", "source"]].to_csv("data/raw/vn_news_label_1_fake.csv", index=False, encoding="utf-8")
print("Đã lưu 2 file trong data/raw/")
