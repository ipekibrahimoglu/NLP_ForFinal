import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# LOAD
# ============================================================
# fetch_data.py'dan gelen metin verisi
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
papers = data["papers"]

# embed.py'dan gelen vektörler
aims_embedding = np.load("aims_embedding.npy")
paper_embeddings = np.load("paper_embeddings.npy")

print(f"Loaded {len(papers)} papers and their embeddings")

# ============================================================
# COSINE SIMILARITY
# ============================================================
# aims_embedding: (384,) → (1, 384) yapıyoruz, cosine_similarity matris ister
aims_vec = aims_embedding.reshape(1, -1)

# Her makalenin Aims & Scope ile benzerliğini hesapla
# Sonuç: (300, 1) matris → düzleştirip (300,) yapıyoruz
scores = cosine_similarity(paper_embeddings, aims_vec).flatten()

print(f"Scores calculated! Min: {scores.min():.3f}, Max: {scores.max():.3f}, Mean: {scores.mean():.3f}")

# ============================================================
# HER MAKALEYE SKORU EKLE
# ============================================================
for i, paper in enumerate(papers):
    paper["alignment_score"] = float(scores[i])

# ============================================================
# SIRALAMA
# ============================================================
# En yüksek skordan en düşüğe sırala
papers_sorted = sorted(papers, key=lambda x: x["alignment_score"], reverse=True)

# ============================================================
# SONUÇLARI GÖSTER
# ============================================================
print("\n🏆 En uyumlu 3 makale (Aims & Scope'a en yakın):")
for p in papers_sorted[:3]:
    print(f"  [{p['alignment_score']:.3f}] {p['title']}")

print("\n⚠️  En az uyumlu 3 makale (potansiyel outlier):")
for p in papers_sorted[-3:]:
    print(f"  [{p['alignment_score']:.3f}] {p['title']}")

# ============================================================
# KAYDET
# ============================================================
data["papers"] = papers_sorted
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved results to 'results.json'")