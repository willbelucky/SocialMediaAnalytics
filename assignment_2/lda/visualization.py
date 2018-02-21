# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 20.
"""
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd

clusters = [24, 16, 1, 19, 20, 19, 21, 20, 11, 8, 25, 10, 19, 19, 14, 22, 20, 12, 6, 2, 5, 3, 17, 16, 22, 14, 22, 15, 7,
            9, 16, 4, 9, 22, 13, 21, 26, 7, 18, 5, 20, 23, 25, 0]

# The distance between each speaker = 1/(cosine similarity+0.0001)
similarity_matrix = pd.read_csv('assignment_2/lda/president_similarity.csv')
speakers = similarity_matrix.iloc[:, 0]
similarity_matrix = similarity_matrix.iloc[:, 1:]
similarity_matrix = similarity_matrix.as_matrix()
cosine_matrix = cosine_similarity(similarity_matrix)
for i in range(len(speakers)):
    for j in range(len(speakers)):
        cosine_matrix[i][j] = 1 / (cosine_matrix[i][j] + 0.001)

# Generate a 2-dimensional plane
mds = manifold.MDS(n_components=2, metric=True, n_init=4, max_iter=1000, eps=1e-3, n_jobs=1, random_state=1,
                   dissimilarity='precomputed')
pos = mds.fit(cosine_matrix).embedding_
pos_x = [x[0] for x in pos]
pos_y = [x[1] for x in pos]
plt.figure(figsize=(15, 15))
plt.scatter(pos_x, pos_y, c=clusters, cmap='tab10')
for i in range(len(speakers)):
    plt.text(pos_x[i], pos_y[i], speakers[i])
    plt.colormaps()
plt.show()
