import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

common_cl_path = PROJECT_ROOT / "data" / "common_cell_lines.csv"
common_drugs_path = PROJECT_ROOT / "data" / "common_drugs.csv"
gdsc_raw_path = PROJECT_ROOT / "data" / "raw" / "drug_sensitivity" / "GDSC2_fitted_dose_response_27Oct23.csv"
output_dir = PROJECT_ROOT / "data" / "processed" / "drug_sensitivity"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "ic50_matrix.csv"

common_cl = pd.read_csv(common_cl_path)
gdsc_to_name = dict(zip(common_cl["GDSC_CELL_LINE_NAME"], common_cl["Name"]))
valid_names = set(common_cl["Name"])

common_drugs = pd.read_csv(common_drugs_path)
common_drug_set = set(common_drugs["GDSC_DRUG_NAME"])

gdsc = pd.read_csv(gdsc_raw_path, usecols=["CELL_LINE_NAME", "DRUG_NAME", "LN_IC50"])

gdsc = gdsc[gdsc["CELL_LINE_NAME"].isin(gdsc_to_name)]
gdsc = gdsc[gdsc["DRUG_NAME"].isin(common_drug_set)]

gdsc["cell_line_name"] = gdsc["CELL_LINE_NAME"].map(gdsc_to_name)

matrix = gdsc.pivot_table(
    index="cell_line_name",
    columns="DRUG_NAME",
    values="LN_IC50",
    aggfunc="first",
)

matrix.index.name = "cell_line_name"

existing_names = [n for n in valid_names if n in matrix.index]
print(f"Cell lines in common list but missing IC50 data: {len(valid_names - set(matrix.index))}")

matrix.to_csv(output_path)

print(f"Output shape: {matrix.shape[0]} cell lines x {matrix.shape[1]} drugs")
print(f"Non-NaN entries: {matrix.notna().sum().sum()} / {matrix.shape[0] * matrix.shape[1]}")
print(f"Saved to: {output_path}")
