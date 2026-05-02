import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Veriyi yükle
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

aims_and_scope = data["aims_and_scope"]
papers = data["papers"]
print(f"Loaded {len(papers)} papers")

# Modeli yükle
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Aims & Scope'u vektöre dönüştür
print("Embedding Aims & Scope...")
aims_embedding = model.encode(aims_and_scope)

# Tüm abstract'ları vektöre dönüştür
print("Embedding abstracts...")
abstracts = [p["abstract"] for p in papers]
paper_embeddings = model.encode(abstracts, show_progress_bar=True)

# Kaydet
np.save("aims_embedding.npy", aims_embedding)
np.save("paper_embeddings.npy", paper_embeddings)

print(f"\n✅ Done!")
print(f"   Aims vektör boyutu: {aims_embedding.shape}")
print(f"   Papers matris boyutu: {paper_embeddings.shape}")