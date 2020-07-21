from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances


import json
from tqdm import tqdm
import json
import sys

default_names =  {
                0:"air",
                1:"player",
                2:"stone",
                3:"pickaxe_item",
                4:"cobblestone_item",
                5:"log",
                6:"axe_item",
                7:"dirt",
                8:"farmland",
                9:"hoe_item",
                10:"water",
                11:"bucket_item",
                12:"water_bucket_item",
                13:"log_item"
            }


class Reduction(Enum):
    TSNE = 'TSNE'
    PCA = 'PCA'


def reduce_dims(embeddings, reduction: Reduction):
    """
    :param embeddings: matrix of textual embeddings
    :param reduction: function to reduce dimension of embeddings
    :return: first two dimensions of reduction
    """
    if reduction == Reduction.TSNE:
        X_embedded = TSNE(n_components=2).fit_transform(embeddings)
    elif reduction == Reduction.PCA:
        X_embedded = PCA(n_components=2).fit_transform(embeddings)

    x = X_embedded[:,0]
    y = X_embedded[:,1]

    return x,y

def load_embeddings(embedding_file,name_dict):
    embeddings = {}
    save_numpy = []
    with open(embedding_file,"r") as embed_file:
        lines = sorted(list(embed_file.readlines())[1:],key=lambda x: int(x.split()[0]))
        for l in lines:
            print(l)
            embeddings[float(l.split()[0])] = np.array([float(x) for x in l.split()[1:]])
            save_numpy.append(np.array([float(x) for x in l.split()[1:]]))


    # with open("embeddings_written_8.npy","w+") as embed_file_out:
    # print(np.stack(embeddings).shape)
    # exit()
    # np.save("sky_hier_embeddings_written_8.npy",np.stack(save_numpy))

    name_to_embedding = {}

    with open(name_dict,"r") as name_file:
        name_map = json.load(name_file)
        for name,value in name_map.items():
            name_to_embedding[name] = embeddings[value]

    return name_to_embedding

def load_embed_from_model(file_name,key="embed",names={1:"player",0:"air",2:"stone",3:"pickaxe_item",4:"cobblestone_item",5:"log",6:"axe_item",13:"log_item"}):
    import torch

    checkpoint = torch.load(file_name,map_location=torch.device('cpu'))
    params = checkpoint["model_state_dict"]

    name_to_embedding = {}

    for name in params.keys():
        print(name)
        if key in name:
            for i,p in enumerate(params[name]):
                if i in names:
                    name_to_embedding[names[i]] = p.tolist()
                    print(p)

    return name_to_embedding

def load_embed_from_npy(file_name,names={1:"player",0:"air",2:"stone",3:"pickaxe_item",4:"cobblestone_item",5:"log",6:"axe_item",13:"log_item"}):
    name_to_embedding = {}
    for i,p in enumerate(np.load(file_name)):
        if i in names:
            name_to_embedding[names[i]] = p.tolist()
            print(p)

    return name_to_embedding


def visualize_similarity(name_to_embedding,file_name=None):

    keys = sorted(name_to_embedding.keys())
    embeddings = [name_to_embedding[k] for k in keys ]
    matrix = euclidean_distances(embeddings)

    # np.save("converged_4task_distance.npy",matrix)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(keys)))
    ax.set_yticks(np.arange(len(keys)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(keys)
    ax.set_yticklabels(keys)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    for i in range(len(keys)):
        for j in range(len(keys)):
            text = ax.text(j, i, round(matrix[i, j],2),
                           ha="center", va="center", color="w")

    # ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name+"embed.png")
        with open(file_name+"embed.json","w+") as json_file:
            json.dump(name_to_embedding,json_file)

# def visualize_embeddings(text_to_embedding, reduce_fn: Reduction, num_clusters, write_to_file=False,
#                          file_name="embeddings_out",show=True):
#     """
#     :param text_to_embedding: dictionary of text to embeddings
#     :param reduce_fn: function to reduce dimension of embeddings
#     :param num_clusters: number of clusters to find
#     :param write_to_file: boolean to write to file
#     :param file_name: name of file to write to
#     :return: text and labels for the text
#     """
#     items = list(sorted(text_to_embedding.items()))
#     text =  [k for k,_ in items]
#     normalize = lambda v: v/np.linalg.norm(v) if np.sum(v) != 0 else v
#     vector_representation =  [normalize(v) for _,v in items]
#     x,y = reduce_dims(vector_representation,reduction=reduce_fn)

#     # labels = run_kmeans(vector_representation,num_clusters)
#     # print(num)

#     # if write_to_file: 
#     #     with open(file_name,"w",encoding="utf-8") as embedding_file:
#     #         embedding_file.write("text\tx\ty\tcluster\n")
#     #         for a,x1,y1,cluster in zip(text,x,y,labels):
#     #             embedding_file.write(a+"\t{}\t{}\t{}\n".format(x1,y1,cluster))

#     if show:

#         plt.clf()
#         plt.scatter(x,y)
#         for t,x0,y0 in zip(text,x,y):
#             plt.annotate(t,(x0,y0))
                
#         plt.show()

#     return text,labels

# name_to_embeddings = load_embeddings(sys.argv[1],"sky_hier_graph.json")
if __name__ == "__main__":

    if ".npy" in sys.argv[1]:
        name_to_embeddings = load_embed_from_npy(sys.argv[1],default_names)
    else:
        name_to_embeddings = load_embed_from_model(sys.argv[1],names=default_names)

    visualize_similarity(name_to_embeddings,"test")

# visualize_embeddings(name_to_embeddings,Reduction.PCA,-1)
