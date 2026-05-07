# Analyzing Thematic Alignment in Scientific Journals

A computational pipeline to quantitatively assess whether articles published in a scientific journal align with its stated "Aims & Scope", detecting thematic drift over time and identifying outlier papers.

---

## Project Overview

This project analyzes the thematic alignment between papers published in the **Computational Linguistics** domain (arXiv cs.CL category) and the journal's stated Aims & Scope. Using sentence embeddings and topic modeling, we compute alignment scores for each paper and analyze how thematic coherence evolves from 2015 to 2026.

---

## Pipeline

---

## Methodology

### 1. Data Collection
- Source: arXiv API (`cat:cs.CL`)
- Period: 2015–2026
- Total papers: ~1918
- Each year queried separately using `submittedDate` filter to ensure balanced distribution

### 2. Content Modelling
- Model: `all-MiniLM-L6-v2` (Sentence-BERT)
- Each abstract → 384-dimensional vector
- Aims & Scope text → single reference vector

### 3. Alignment Scoring
- Method: Cosine Similarity between each paper embedding and Aims & Scope embedding
- Score range: -1 to 1 (higher = more aligned)
- Outlier threshold: bottom 5% percentile

### 4. Topic Modeling
- Model: BERTopic
- Automatically discovers latent topics in the corpus
- Each topic assigned a mean alignment score

### 5. Visualization
- Alignment score distribution (histogram)
- Thematic drift over time (line chart)
- Score distribution by year (boxplot)
- Percentile trends p10/median/p90
- Outlier rate by year
- UMAP 2D projection colored by alignment score and topic

---

## Key Findings

- **Mean alignment score: ~0.28** — moderate alignment between cs.CL papers and Computational Linguistics scope
- **Thematic drift detected**: alignment peaked in 2016 (0.305) and has been declining, with a sharp drop in 2025-2026
- **Most misaligned topics**: image/video processing, RNN/LSTM architectures, adversarial attacks
- **Most aligned topics**: dependency parsing, natural language generation, morphological analysis
- **38% of papers** (Topic -1) could not be assigned to any coherent topic — reflecting the breadth of cs.CL
- **Outlier rate increasing** in recent years, especially 2021 and 2025

---

## Results

| Metric | Value |
|--------|-------|
| Total papers | 1918 |
| Year range | 2015–2026 |
| Mean alignment score | 0.28 |
| Max alignment score | 0.566 |
| Min alignment score | -0.004 |
| Outliers (bottom 5%) | 96 / 1918 |

---

## Installation

```bash
# Create virtual environment
python -m venv nlp-env
nlp-env\Scripts\activate  # Windows

# Install dependencies
pip install requests sentence-transformers bertopic umap-learn hdbscan matplotlib pandas numpy scikit-learn
```

---

## Usage

Run the pipeline in order:

```bash
python fetch_data.py      # Collect data
python embed.py           # Generate embeddings
python align.py           # Compute alignment scores
python topic.py           # Topic modeling
python visualize.py       # Generate visualizations
python umap_viz.py        # UMAP visualization
python report.py          # Generate report
```

---

## Output Files

| File | Description |
|------|-------------|
| `data.json` | Raw collected papers |
| `aims_embedding.npy` | Aims & Scope vector |
| `paper_embeddings.npy` | All paper vectors |
| `results.json` | Papers with alignment scores |
| `results_with_topics.csv` | Papers with topic assignments |
| `report.json` | Summary statistics |
| `report_summary.csv` | Year-by-year summary |
| `histogram.png` | Score distribution |
| `drift.png` | Thematic drift over time |
| `boxplot.png` | Score distribution by year |
| `trend_p10_median_p90.png` | Percentile trends |
| `outlier_rate.png` | Outlier rate by year |
| `topic_alignment.png` | Topic alignment scores |
| `umap_alignment.png` | UMAP colored by alignment |
| `umap_topics.png` | UMAP colored by topic |

---

## References

- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- McInnes, L., et al. (2018). UMAP: Uniform Manifold Approximation and Projection.