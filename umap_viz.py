import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

# ============================================================
# LOAD
# ============================================================
with open("results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

papers = data["papers"]
df = pd.DataFrame(papers)

# Embeddings
paper_embeddings = np.load("paper_embeddings.npy")
aims_embedding = np.load("aims_embedding.npy")

print(f"Loaded {len(df)} papers")
print(f"Paper embeddings shape: {paper_embeddings.shape}")

# Topic bilgisi varsa yükle
try:
    topics_df = pd.read_csv("results_with_topics.csv")
    df["topic_id"] = topics_df["topic_id"].values
    has_topics = True
    print("Topic info loaded!")
except:
    has_topics = False
    print("No topic info found, coloring by alignment score.")

# ============================================================
# UMAP — 384 boyuttan 2 boyuta indir
# ============================================================
print("\nRunning UMAP... (may take 1-2 minutes)")

# Aims & Scope vektörünü de ekle — grafikte göstereceğiz
all_embeddings = np.vstack([paper_embeddings, aims_embedding.reshape(1, -1)])

reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embedding_2d = reducer.fit_transform(all_embeddings)

# Son nokta = Aims & Scope
paper_2d = embedding_2d[:-1]
aims_2d = embedding_2d[-1]

df["umap_x"] = paper_2d[:, 0]
df["umap_y"] = paper_2d[:, 1]

# ============================================================
# VISUALIZATION 1 — Alignment Score ile renklendir
# ============================================================
plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    df["umap_x"], df["umap_y"],
    c=df["alignment_score"],
    cmap="RdYlGn",  # kırmızı=düşük, yeşil=yüksek
    alpha=0.6,
    s=10
)

# Aims & Scope noktası
plt.scatter(aims_2d[0], aims_2d[1],
            c="blue", s=200, marker="*",
            zorder=5, label="Aims & Scope")

plt.colorbar(scatter, label="Alignment Score")
plt.title("UMAP: Papers colored by Alignment Score\n(green=aligned, red=misaligned, ★=Aims & Scope)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend()
plt.tight_layout()
plt.savefig("umap_alignment.png", dpi=200)
plt.show()
print("✅ umap_alignment.png saved")

# ============================================================
# VISUALIZATION 2 — Topic ile renklendir
# ============================================================
if has_topics:
    plt.figure(figsize=(12, 8))

    # Topic -1 (noise) gri, diğerleri renkli
    colors = df["topic_id"].apply(lambda x: "grey" if x == -1 else x)
    
    unique_topics = sorted(df[df["topic_id"] != -1]["topic_id"].unique())
    cmap = plt.cm.get_cmap("tab20", len(unique_topics))

    for i, topic in enumerate(unique_topics[:20]):  # max 20 topic göster
        mask = df["topic_id"] == topic
        plt.scatter(
            df[mask]["umap_x"], df[mask]["umap_y"],
            c=[cmap(i)], alpha=0.6, s=10, label=f"Topic {topic}"
        )

    # Noise noktaları
    mask_noise = df["topic_id"] == -1
    plt.scatter(df[mask_noise]["umap_x"], df[mask_noise]["umap_y"],
                c="lightgrey", alpha=0.3, s=5, label="No topic (-1)")

    # Aims & Scope
    plt.scatter(aims_2d[0], aims_2d[1],
                c="black", s=200, marker="*",
                zorder=5, label="Aims & Scope")

    plt.title("UMAP: Papers colored by Topic (BERTopic)\n(★=Aims & Scope)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, markerscale=3)
    plt.tight_layout()
    plt.savefig("umap_topics.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("✅ umap_topics.png saved")

# ============================================================
# SAVE
# ============================================================
df.to_csv("results_with_umap.csv", index=False)
print("✅ results_with_umap.csv saved")