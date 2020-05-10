""" Some code borrowed from https://github.com/tkipf/pygcn."""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self,adj_mat,num_nodes,num_types,idx_2_game_char,use_cuda=True,use_graph=True):
        super(GCN, self).__init__()

        print("starting init")
        # n = 5
        self.n = num_nodes #n
        self.num_types = num_types
        self.emb_sz = 4

        self.nodes = torch.arange(0,self.n)
        if use_cuda: 
            self.nodes = self.nodes.cuda()
        self.node_to_game_char = idx_2_game_char #{i:i+1 for i in self.objects.tolist()}
        self.game_char_to_node = {v:k for k,v in self.node_to_game_char.items()}
        # get and normalize adjacency matrix.
        A_raw = adj_mat #torch.eye(self.n) #torch.load("") #./data/gcn/adjmat.dat")
        A = A_raw #normalize_adj(A_raw).tocsr().toarray()
        self.A = A
        if use_cuda: 
            self.A = self.A.cuda() 

        self.use_graph = use_graph

        if self.use_graph:
            print("Using graph network")
            self.W0 = nn.Linear(self.emb_sz, 16, bias=False)
            self.W1 = nn.Linear(16, 16, bias=False)
            self.W2 = nn.Linear(16, 16, bias=False)
            self.get_node_emb = nn.Embedding(self.n, self.emb_sz)
            self.final_mapping = nn.Linear(16, self.emb_sz)

        self.get_obj_emb = nn.Embedding(self.num_types, self.emb_sz)
      

        print("finished initializing")

    def gcn_embed(self):        
        node_embeddings = self.get_node_emb(self.nodes)

        x = torch.mm(self.A, node_embeddings)
        x = F.relu(self.W0(x))
        x = torch.mm(self.A, x)
        x = F.relu(self.W1(x))
        x = torch.mm(self.A, x)
        x = F.relu(self.W2(x))
        x = self.final_mapping(x)
        return x

    def embed_state(self,game_state):
        game_state_embed = self.get_obj_emb(game_state.view(-1,game_state.shape[-2]*game_state.shape[-1]))
        game_state_embed = game_state_embed.view(-1,game_state.shape[-2],game_state.shape[-1],self.emb_sz)
       
        node_embeddings = None
        if self.use_graph:
            print("USING GRAPHS!!!!!")
            node_embeddings = self.gcn_embed()
            for n,embedding in zip(self.nodes.tolist(),node_embeddings):
                if n in self.node_to_game_char:
                    indx = (game_state == self.node_to_game_char[n]).nonzero()
                    game_state_embed[indx[:, 0], indx[:, 1], indx[:, 2]] = embedding

        return game_state_embed.permute((0,3,1,2)),node_embeddings

#Build GCN on torch with identity matrix adjacency, with 5 nodes, 6 types and a mapping each node to its state character

# num_nodes = 3
# total_objects = 5
# dict_2_game = {0:2,1:3,2:4} #0->stone, 1->pickaxe, 2->cobblestone
# adjacency = torch.tensor([[1,0,1],[1,1,0],[0,0,1]]).float()

# #stone's adjacencies [1,0,1]
# #pickaxe's adjacencies [1,1,0]
# #cobblestones's adjacencies [0,0,1]

# test = GCN(adjacency,num_nodes,total_objects,dict_2_game)
# new_state = test.embed_state(torch.ones((1,10,10)).long())
# print(new_state.shape)