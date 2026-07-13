import numpy as np
embedding_matrix = np.load("data/processed/gene_embeddings/esm2_gene_embeddings.npy")
print(embedding_matrix.shape)