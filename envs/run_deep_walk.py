#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
# from io import open
# from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from deepwalk import graph
from deepwalk import walks as serialized_walks
from gensim.models import Word2Vec
from deepwalk.skipgram import Skipgram
import numpy as np

def run_dw(matrix,num_walks=100,walk_length=5,representation_size=32,window_size=2,undirected=True,seed=0,workers=1):
  random.seed(seed)
  np.random.seed(seed)
  adj_list = []
  for n,edges in enumerate(matrix):
    adj_list.append([n]+edges.nonzero()[0].tolist())

  print(adj_list)

  G = graph.from_adjlist(adj_list)
  if undirected:
    G.make_undirected()

  print("Number of nodes: {}".format(len(G.nodes())))
  num_walks = len(G.nodes()) * num_walks

  print("Number of walks: {}".format(num_walks))

  data_size = num_walks * walk_length

  print("Data size (walks*length): {}".format(data_size))

  if data_size < 1000000000:
    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=num_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
    print("Training...")
    model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1, workers=workers)
  else:
    print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, 1000000000))
    print("Walking...")

    walks_filebase = str(adj_list) + ".walks"
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=num_walks,
                                         path_length=walk_length, alpha=0, rand=random.Random(seed),
                                         num_workers=workers)

    print("Counting vertex frequency...")
    #if not args.vertex_freq_degree:
    vertex_counts = serialized_walks.count_textfiles(walk_files, workers)
    #else:
    #  # use degree distribution for frequency in tree
    #  vertex_counts = G.degree(nodes=G.iterkeys())

    print("Training...")
    walks_corpus = serialized_walks.WalksCorpus(walk_files)
    model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                     size=representation_size,
                     window=window_size, min_count=0, trim_rule=None, workers=workers,seed=seed)

  embeddings = np.zeros((len(G.nodes()),representation_size))

  for i in range(len(G.nodes())):
      embeddings[i] = model.wv.get_vector(str(i))

  return embeddings


def main():
  input_test = np.ones((2,2))
  # input_test[0][1] = 1
  # input_test[1][0] = 1

  print(run_dw(input_test))

if __name__ == "__main__":
  sys.exit(main())
