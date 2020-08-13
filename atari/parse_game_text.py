from build_graph import *
import sys
import json
import spacy
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity



nlp = spacy.load("en_core_web_lg")

def run_kmeans(embeddings, n_clusters: int):
    """
    :param embeddings: matrix of textual embeddings
    :param n_clusters: number of clusters
    :return: kmeans labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.labels_

def reduce_dims(embeddings, reduction):
    """
    :param embeddings: matrix of textual embeddings
    :param reduction: function to reduce dimension of embeddings
    :return: first two dimensions of reduction
    """
    if reduction == "TSNE":
        X_embedded = TSNE(n_components=2).fit_transform(embeddings)
    elif reduction == "PCA":
        X_embedded = PCA(n_components=2).fit_transform(embeddings)

    x = X_embedded[:,0]
    y = X_embedded[:,1]

    return x,y


def visualize_embeddings(text_to_embedding, reduce_fn, num_clusters,show=True):
    """
    :param text_to_embedding: dictionary of text to embeddings
    :param reduce_fn: function to reduce dimension of embeddings
    :param num_clusters: number of clusters to find
    :param write_to_file: boolean to write to file
    :param file_name: name of file to write to
    :return: text and labels for the text
    """
    items = list(sorted(text_to_embedding.items()))
    text =  [k for k,_ in items]
    normalize = lambda v: v/np.linalg.norm(v) if np.sum(v) != 0 else v
    vector_representation =  [normalize(v) for _,v in items]
    x,y = reduce_dims(vector_representation,reduction=reduce_fn)

    # labels = run_kmeans(vector_representation,num_clusters)
    # print(num)

    if show:
  
        plt.clf()
        plt.scatter(x,y)
        for t,xi,yi in zip(text,x,y):
           plt.annotate(t,(xi,yi))
        plt.show()

    return text #,labels

with open("game_to_gameplay_text.json","r",encoding="latin-1") as game_file:
    	text_dict = json.load(game_file)

print(text_dict.keys())

all_docs = [(k,nlp(v)) for k,v in text_dict.items()]

name_to_vector = {}

for k,d in all_docs:
	name_to_vector[k] = d.vector

visualize_embeddings(name_to_vector, "PCA", 5,show=True)



for k,d in all_docs:
	print(k,": \n",sorted([(d.similarity(x[1]),x[0]) for x in all_docs if x[0] != k],key=lambda x: x[0],reverse=True)[:3])



exit()

if sys.argv[1] == "all":
	print(process_text(" ".join([v.encode("ascii", "ignore").decode() for _,v in text_dict.items()]),sys.argv[1]))


else:
   print(process_text(text_dict[sys.argv[1]+'\n'],sys.argv[1]))
