from Bio import SwissProt
import torch
from transformers import EsmModel, EsmTokenizer
import numpy as np
from tqdm import tqdm

def parse_uniprot_dat(dat_path):
    """解析 UniProt DAT 文件，建立基因名 → 序列的映射"""
    gene_to_seq = {}

    with open(dat_path) as f:
        for record in tqdm(SwissProt.parse(f), desc = "parsing sequences"):
            gene_names = record.gene_name  # 可能是列表
            if gene_names:
                # 取第一个基因名
                primary_gene = gene_names[0].get('Name', '')
                if primary_gene:
                    gene_to_seq[primary_gene] = record.sequence

    return gene_to_seq

# ===== 1. 加载 ESM-2 模型（仅需一次，约3-5 分钟） =====
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # 33层，6.5亿参数
# 如果 GPU 显存不足，可以用轻量版：
# MODEL_NAME = "facebook/esm2_t6_8M_UR50D"    # 6层，800万参数，推荐初学使用

print("loading model")
model = EsmModel.from_pretrained(MODEL_NAME)
tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
print("loaded model")

# 移到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# ===== 2. 对单个蛋白质序列编码 =====
def get_protein_embedding(sequence):
    """
    输入: 氨基酸序列字符串（如 'MRPSGTAGAA...'）
    输出: (1280,) 的 ESM-2 嵌入向量
    """
    # Tokenize：将氨基酸字母转为数字ID
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # 取所有位置的平均（也可以用 [CLS] token）
    embedding = outputs.last_hidden_state.mean(dim=1)  # (1, 1280)
    return embedding.squeeze(0).cpu().numpy()

# ===== 3. 批量编码所有基因 =====
protein_sequences = parse_uniprot_dat("data/uniprot_sprot.dat")
embeddings = {}
for gene, seq in tqdm(protein_sequences.items(), desc = "processing proteins"):
    if seq:
        emb = get_protein_embedding(seq)
        embeddings[gene] = emb

# ===== 4. 保存嵌入矩阵（避免重复计算！） =====
# 构建 (num_genes, 1280) 的矩阵
gene_order = sorted(embeddings.keys())
embedding_matrix = np.stack([embeddings[g] for g in gene_order])

np.save("esm2_gene_embeddings.npy", embedding_matrix)

# 保存基因顺序
with open("gene_order.txt", 'w') as f:
    for gene in gene_order:
        f.write(f"{gene}\n")