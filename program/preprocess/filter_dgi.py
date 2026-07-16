import pickle
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

raw_path = PROJECT_ROOT / "data" / "raw" / "drug_gene_interaction" / "interactions.tsv"
output_dir = PROJECT_ROOT / "data" / "processed" / "drug_gene_interaction"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "interactions_filtered.csv"

df = pd.read_csv(raw_path, sep="\t")
print(f"Total interactions: {len(df)}")

df = df.dropna(subset=["interaction_score"]).copy()
print(f"After dropping NaN scores: {len(df)}")

scores = df["interaction_score"].values.astype(np.float64)
log_scores = np.log1p(scores)

log_min = log_scores.min()
log_max = log_scores.max()
print(f"Log-score range: [{log_min:.4f}, {log_max:.4f}]")

norm_scores = (log_scores - log_min) / (log_max - log_min)
print(f"Normalized score range: [{norm_scores.min():.4f}, {norm_scores.max():.4f}]")


score_threshold = 0.175
print(f"raw_score > {score_threshold:.4f}")

mask = scores > score_threshold
df_filtered = df[mask].copy()
df_filtered["score_norm"] = norm_scores[mask]

print(f"After score > {score_threshold} filter: {len(df_filtered)}")

df_filtered.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

interaction_list = df_filtered[["gene_name", "drug_name", "score_norm"]].values.tolist()
print(f"Interaction list length: {len(interaction_list)}")

list_path = output_dir / "interaction_list.pkl"
with open(list_path, "wb") as f:
    pickle.dump(interaction_list, f)
print(f"List saved to {list_path}")
