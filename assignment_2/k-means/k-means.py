import pandas as pd
from matplotlib import pyplot as plt
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance
from sklearn import manifold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from assignment_2.data.data_reader import get_speeches
from assignment_2.data.president import Trump

speeches = get_speeches()

speeches_sum = speeches.groupby(['president'])[['script']].sum()
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(speeches_sum.script).toarray()
tfidf_matrix = tfidf.fit_transform(speeches_sum.script).toarray()
kmeans = KMeansClusterer(27, distance=cosine_distance, repeats=25)
clusters = kmeans.cluster(X, assign_clusters=True)

speakers = speeches_sum.index.tolist()

cluster_df = pd.DataFrame(speakers, clusters)
cluster_df = cluster_df.reset_index()
cluster_df.columns = ['president_index', 'president']
print(clusters)

trump_index = cluster_df[cluster_df['president'] == Trump]['president_index'].iloc[0]

similar_speakers = [speakers[i] for i, c in enumerate(clusters) if (c == trump_index) and (i != 8)]
print(similar_speakers)

# The distance between each speaker = 1/(cosine similarity+0.0001)
cosine_matrix = cosine_similarity(tfidf_matrix)
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
