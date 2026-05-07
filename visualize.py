import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# LOAD
# ============================================================
with open("results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

papers = data["papers"]
df = pd.DataFrame(papers)

print(f"Loaded {len(df)} papers")
print(f"Score range: {df['alignment_score'].min():.3f} - {df['alignment_score'].max():.3f}")

# ============================================================
# 1) HISTOGRAM — score distribution
# ============================================================
plt.figure(figsize=(8, 4))
plt.hist(df["alignment_score"], bins=30, color="skyblue", edgecolor="black")
plt.xlabel("Alignment Score")
plt.ylabel("Number of Papers")
plt.title("Alignment Score Distribution")
plt.tight_layout()
plt.savefig("histogram.png", dpi=200)
plt.show()
print("✅ histogram.png saved")

# ============================================================
# 2) YEARLY DRIFT — mean score per year
# ============================================================
yearly = df.groupby("year")["alignment_score"].mean().reset_index()
yearly = yearly.sort_values("year")

plt.figure(figsize=(9, 4))
plt.plot(yearly["year"], yearly["alignment_score"], marker="o", color="steelblue")
plt.xlabel("Year")
plt.ylabel("Mean Alignment Score")
plt.title("Mean Alignment Score by Year (Thematic Drift)")
plt.xticks(yearly["year"], rotation=35, ha="right")
plt.tight_layout()
plt.savefig("drift.png", dpi=200)
plt.show()
print("✅ drift.png saved")

# ============================================================
# 3) OUTLIER DETECTION — bottom 5% percentile
# ============================================================
threshold = df["alignment_score"].quantile(0.05)
df["is_outlier"] = df["alignment_score"] <= threshold

print(f"\n⚠️  Outlier threshold (bottom 5%): {threshold:.3f}")
print(f"   Number of outliers: {int(df['is_outlier'].sum())} / {len(df)}")

outliers = df[df["is_outlier"]].sort_values("alignment_score")
print("\n🔴 5 lowest scoring papers (outliers):")
for _, row in outliers.head(5).iterrows():
    print(f"  [{row['alignment_score']:.3f}] {row['title']}")

print("\n🟢 5 highest scoring papers:")
for _, row in df.nlargest(5, "alignment_score").iterrows():
    print(f"  [{row['alignment_score']:.3f}] {row['title']}")

# ============================================================
# 4) OUTLIER RATE BY YEAR
# ============================================================
yearly_outlier = df.groupby("year").agg(
    n=("alignment_score", "size"),
    outliers=("is_outlier", "sum")
).reset_index()
yearly_outlier["outlier_rate"] = yearly_outlier["outliers"] / yearly_outlier["n"]

plt.figure(figsize=(9, 4))
plt.bar(yearly_outlier["year"].astype(str), yearly_outlier["outlier_rate"],
        color="salmon", edgecolor="black")
plt.xlabel("Year")
plt.ylabel("Outlier Rate")
plt.title("Outlier Rate by Year")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("outlier_rate.png", dpi=200)
plt.show()
print("✅ outlier_rate.png saved")

# ============================================================
# 5) SAVE
# ============================================================
df.to_csv("results_with_outliers.csv", index=False)
print("\n✅ results_with_outliers.csv saved")

# ============================================================
# 5) BOXPLOT BY YEAR
# ============================================================
years = sorted(df["year"].unique())
data_by_year = [df[df["year"] == y]["alignment_score"].values for y in years]

plt.figure(figsize=(12, 5))
plt.boxplot(data_by_year, tick_labels=[str(y) for y in years], showfliers=True)
plt.xlabel("Year")
plt.ylabel("Alignment Score")
plt.title("Alignment Score Distribution by Year (Boxplot)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("boxplot.png", dpi=200)
plt.show()
print("✅ boxplot.png saved")

# ============================================================
# 6) p10 / MEDIAN / p90 TREND
# ============================================================
yearly_stats = []
for y in years:
    scores = df[df["year"] == y]["alignment_score"].values
    yearly_stats.append({
        "year": y,
        "p10": np.percentile(scores, 10),
        "median": np.percentile(scores, 50),
        "p90": np.percentile(scores, 90),
    })

stats_df = pd.DataFrame(yearly_stats)

plt.figure(figsize=(12, 5))
plt.plot(stats_df["year"], stats_df["p10"], marker="o", label="p10 (bottom 10%)", color="red")
plt.plot(stats_df["year"], stats_df["median"], marker="o", label="Median", color="steelblue")
plt.plot(stats_df["year"], stats_df["p90"], marker="o", label="p90 (top 10%)", color="green")
plt.fill_between(stats_df["year"], stats_df["p10"], stats_df["p90"], alpha=0.1, color="steelblue")
plt.xlabel("Year")
plt.ylabel("Alignment Score")
plt.title("Alignment Score Trends: p10 / Median / p90 by Year")
plt.xticks(stats_df["year"], rotation=35, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("trend_p10_median_p90.png", dpi=200)
plt.show()
print("✅ trend_p10_median_p90.png saved")