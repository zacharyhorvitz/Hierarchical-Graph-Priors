from build_graph import *
import sys
import json
import spacy
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from nltk.corpus import stopwords

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

STOP_WORDS = stopwords.words('english')

nlp = spacy.load("en_core_web_lg")


def extract_cluster_names(text, labels):
    label_to_all_text = {l: "" for l in labels}
    for t, l in zip(text, labels):
        label_to_all_text[l] += " " + t

    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)
    corpus = [t for _, t in sorted(label_to_all_text.items(), key=lambda x: x[0])]
    labels = [label for label, _ in sorted(label_to_all_text.items(), key=lambda x: x[0])]
    X = vectorizer.fit_transform(corpus)

    words_to_index = {w: i for i, w in enumerate(vectorizer.get_feature_names())}
    all_words = vectorizer.get_feature_names()

    for tfidf_score, l in zip(X, labels):
        tfidf_score = np.reshape(tfidf_score.toarray(), (-1))
        most_characteristic_words = sorted(all_words,
                                           key=lambda x: tfidf_score[words_to_index[x]],
                                           reverse=True)[:20]
        print(most_characteristic_words)


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

    x = X_embedded[:, 0]
    y = X_embedded[:, 1]

    return x, y


def calculate_num_clusters(embeddings, kmax: int = 30):
    """
    :param embeddings: matrix of textual embeddings
    :param kmax: maximum number of clusters
    :return: optimal number of clusters
    """
    sil = []
    embs = embeddings
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(embs)
        labels = kmeans.labels_
        sil.append((k, silhouette_score(embs, labels, metric='euclidean')))

    print(sil)
    return max(sil, key=lambda x: x[1])


def visualize_embeddings(text_to_embedding, reduce_fn, num_clusters, show=True):
    """
    :param text_to_embedding: dictionary of text to embeddings
    :param reduce_fn: function to reduce dimension of embeddings
    :param num_clusters: number of clusters to find
    :param write_to_file: boolean to write to file
    :param file_name: name of file to write to
    :return: text and labels for the text
    """
    items = list(sorted(text_to_embedding.items()))
    text = [k for k, _ in items]
    # normalize = lambda v: v/np.linalg.norm(v) if np.sum(v) != 0 else v
    vector_representation = [v for _, v in items]
    x, y = reduce_dims(vector_representation, reduction=reduce_fn)

    labels = run_kmeans(vector_representation, num_clusters)
    # print(num)

    if show:

        plt.clf()
        colors = ['r', 'g', 'b', 'y', 'm']
        for c, label in enumerate(set(labels)):
            plt.scatter([i for i, l in zip(x, labels) if l == label],
                        [i for i, l in zip(y, labels) if l == label],
                        c=colors[c])
        for t, l, xi, yi in zip(text, labels, x, y):
            plt.annotate(t + "-{}".format(l), (xi, yi))
        plt.show()

    return text, labels


def load_data(exclude={"Tennis\n"}, print_most_similar=False):

    with open("game_to_gameplay_text.json", "r", encoding="latin-1") as game_file:
        text_dict = json.load(game_file)

    print(text_dict.keys())

    all_docs = [(k, nlp(v)) for k, v in text_dict.items() if not k in exclude]

    if print_most_similar:
        for k, d in all_docs:
            print(
                k,
                ": \n",
                sorted([(d.similarity(x[1]), x[0]) for x in all_docs if x[0] != k],
                       key=lambda x: x[0],
                       reverse=True)[:3])

    name_to_vector = {}

    for k, d in all_docs:
        name_to_vector[k] = d.vector

    return text_dict, name_to_vector


text_dict, name_to_vector = load_data()

if sys.argv[1] == "VIZ":
    text, labels = visualize_embeddings(name_to_vector, "PCA", 4, show=True)
    print("\n\n")
    extract_cluster_names([text_dict[t] for t in text], labels)

    for label in set(labels):
        print(
            process_text(
                " ".join([
                    text_dict[t].encode("ascii", "ignore").decode() for l,
                    t in zip(labels, text) if l == label
                ]),
                str(label)))
        print("\n")

elif sys.argv[1] == "all":
    print(
        process_text(" ".join([v.encode("ascii", "ignore").decode() for _, v in text_dict.items()]),
                     sys.argv[1]))
else:
    print(process_text(text_dict[sys.argv[1] + '\n'], sys.argv[1]))
