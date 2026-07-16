import pubchempy as pcp
from tqdm import tqdm
import pandas as pd

with open("data/common_drugs_list.txt") as handle:
    file = handle.read()

compounds = file.split('\n')
compounds.remove('')
results = []
for compound in tqdm(compounds):
    try:
        search_results = pcp.get_compounds(compound, 'name')
        cid = search_results [0].cid
        smiles = search_results [0].smiles
        results.append([compound, cid, smiles])
    except:
        results.append([compound, None, None])
print(results)
results_dataframe = pd.DataFrame(results, columns=['compound', 'cid', 'smiles'])
results_dataframe.to_csv("data/raw/drug_structures/compound_cid_smiles.csv")