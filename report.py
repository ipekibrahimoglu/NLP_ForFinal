import json
import pandas as pd
import numpy as np

# ============================================================
# LOAD
# ============================================================
with open("results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

papers = data["papers"]
df = pd.DataFrame(papers)

print("=" * 60)
print("JOURNAL THEMATIC ALIGNMENT — DATA CURATION REPORT")
print("=" * 60)

# ============================================================
# GENEL İSTATİSTİKLER
# ============================================================
print(f"\n📊 GENERAL STATISTICS")
print(f"  Journal:        {data['journal']}")
print(f"  Year range:     {data['year_range']}")
print(f"  Total papers:   {len(df)}")
print(f"\n  Alignment Score:")
print(f"    Mean:         {df['alignment_score'].mean():.4f}")
print(f"    Std:          {df['alignment_score'].std():.4f}")
print(f"    Min:          {df['alignment_score'].min():.4f}")
print(f"    Max:          {df['alignment_score'].max():.4f}")
print(f"    Median:       {df['alignment_score'].median():.4f}")

# ============================================================
# YIL BAZLI DAĞILIM
# ============================================================
print(f"\n📅 PAPERS PER YEAR")
year_counts = df.groupby("year").size().reset_index(name="count")
for _, row in year_counts.iterrows():
    bar = "█" * int(row["count"] / 10)
    print(f"  {int(row['year'])}: {row['count']:4d} {bar}")

# ============================================================
# YIL BAZLI ORTALAMA SKOR
# ============================================================
print(f"\n📈 MEAN ALIGNMENT SCORE PER YEAR")
yearly_scores = df.groupby("year")["alignment_score"].mean().reset_index()
for _, row in yearly_scores.iterrows():
    bar = "█" * int(row["alignment_score"] * 100)
    print(f"  {int(row['year'])}: {row['alignment_score']:.4f} {bar}")

# ============================================================
# OUTLIER ANALİZİ
# ============================================================
threshold = df["alignment_score"].quantile(0.05)
df["is_outlier"] = df["alignment_score"] <= threshold
outliers = df[df["is_outlier"]].sort_values("alignment_score")

print(f"\n⚠️  OUTLIER ANALYSIS (bottom 5%)")
print(f"  Threshold:      {threshold:.4f}")
print(f"  Total outliers: {len(outliers)} / {len(df)} ({len(outliers)/len(df)*100:.1f}%)")

print(f"\n🔴 10 MOST MISALIGNED PAPERS:")
for _, row in outliers.head(10).iterrows():
    print(f"  [{row['alignment_score']:.3f}] {row['title'][:70]}")

print(f"\n🟢 10 MOST ALIGNED PAPERS:")
for _, row in df.nlargest(10, "alignment_score").iterrows():
    print(f"  [{row['alignment_score']:.3f}] {row['title'][:70]}")

# ============================================================
# KAYDET — JSON
# ============================================================
report = {
    "journal": data["journal"],
    "year_range": data["year_range"],
    "total_papers": len(df),
    "alignment_score": {
        "mean": round(float(df["alignment_score"].mean()), 4),
        "std": round(float(df["alignment_score"].std()), 4),
        "min": round(float(df["alignment_score"].min()), 4),
        "max": round(float(df["alignment_score"].max()), 4),
        "median": round(float(df["alignment_score"].median()), 4),
    },
    "papers_per_year": {
        str(int(row["year"])): int(row["count"])
        for _, row in year_counts.iterrows()
    },
    "mean_score_per_year": {
        str(int(row["year"])): round(float(row["alignment_score"]), 4)
        for _, row in yearly_scores.iterrows()
    },
    "outlier_analysis": {
        "threshold": round(float(threshold), 4),
        "total_outliers": len(outliers),
        "outlier_rate": round(len(outliers) / len(df), 4),
    },
    "top10_aligned": [
        {"title": row["title"], "score": round(float(row["alignment_score"]), 4), "year": int(row["year"])}
        for _, row in df.nlargest(10, "alignment_score").iterrows()
    ],
    "top10_misaligned": [
        {"title": row["title"], "score": round(float(row["alignment_score"]), 4), "year": int(row["year"])}
        for _, row in outliers.head(10).iterrows()
    ],
}

with open("report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n✅ report.json saved")

# ============================================================
# KAYDET — CSV
# ============================================================
summary_rows = []
for year in sorted(df["year"].unique()):
    year_df = df[df["year"] == year]
    summary_rows.append({
        "year": int(year),
        "n_papers": len(year_df),
        "mean_score": round(float(year_df["alignment_score"].mean()), 4),
        "median_score": round(float(year_df["alignment_score"].median()), 4),
        "std_score": round(float(year_df["alignment_score"].std()), 4),
        "n_outliers": int((year_df["alignment_score"] <= threshold).sum()),
        "outlier_rate": round(float((year_df["alignment_score"] <= threshold).mean()), 4),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("report_summary.csv", index=False)
print("✅ report_summary.csv saved")

print("\n" + "=" * 60)
print("REPORT COMPLETE")
print("=" * 60)