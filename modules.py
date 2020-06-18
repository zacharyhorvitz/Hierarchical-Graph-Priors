import torch
import sys
import numpy as np
import random
from collections import deque, namedtuple
import torch.nn.functional as F
from torch.nn.functional import relu

from utils import sync_networks, conv2d_size_out

# Adapted from https://github.com/tkipf/pygcn.
class GCN(torch.nn.Module):

    def __init__(self,
                 adj_matrices,
                 device,
                 num_nodes,
                 num_types,
                 idx_2_game_char,
                 embed_wall=False,
                 use_graph=True,
                 atten=False, 
                 one_layer=False,
                 node_glove_embed=None,
                 emb_size=16):
        super(GCN, self).__init__()

        print("starting init")
        # n = 5
        self.n = num_nodes  #n
        self.num_types = num_types
        self.emb_sz = emb_size
        if embed_wall:
            self.wall_embed = torch.FloatTensor(torch.ones(
                self.emb_sz)).to(device)
        else:
            self.wall_embed = None
        self.atten = atten
        self.nodes = torch.arange(0, self.n)
        self.nodes = self.nodes.to(device)
        self.node_to_game_char = idx_2_game_char  #{i:i+1 for i in self.objects.tolist()}
        self.game_char_to_node = {
            v: k for k, v in self.node_to_game_char.items()
        }
        self.num_types=num_types
        A_raw = adj_matrices  
        self.A = [x.to(device) for x in A_raw] 
        self.use_graph = use_graph
        self.num_edges = len(A_raw)
        if self.use_graph:
            if self.atten:
                print('Using attention')
                self.attention = torch.nn.ParameterList([torch.nn.Parameter(A.detach(), requires_grad=True) for A in A_raw])

            self.layer_sizes = [(self.emb_sz,self.emb_sz//self.num_edges),(self.emb_sz,self.emb_sz//self.num_edges),(self.emb_sz,self.emb_sz//self.num_edges)]
            self.num_layers = len(self.layer_sizes)
            self.weights = [[torch.nn.Linear(in_dim,out_dim,bias=False).to(device) for (in_dim,out_dim) in self.layer_sizes] for e in range(self.num_edges)]
            for i in range(self.num_edges):   
                for j in range(self.num_layers):  
                     self.add_module(str((i,j)),self.weights[i][j])           
            self.get_node_emb = torch.nn.Embedding(self.n, self.emb_sz)
            if node_glove_embed is not None:
                 print("using glove!")
                 self.get_node_emb.weight.data.copy_(node_glove_embed)
                 self.get_node_emb.requires_grad = True #False
            self.final_mapping = torch.nn.Linear(self.emb_sz,self.emb_sz)
        self.obj_emb = torch.nn.Embedding(self.num_types, self.emb_sz)
        print("finished initializing")



    def gcn_embed(self):
        node_embeddings = self.get_node_emb(self.nodes)
    #make size 16, first layer
        x = node_embeddings
        for l in range(self.num_layers):
            layer_out = []
            for e in range(self.num_edges):
                 if self.atten:
                     weighting = F.normalize(self.attention[e] * self.A[e]) 
                 else:
                     weighting = F.normalize(self.A[e])
                 layer_out.append(torch.mm(weighting, x))#)
            x = torch.cat([relu(self.weights[e][l](type_features))  for e,type_features in enumerate(layer_out)],axis=1)
        x = self.final_mapping(x)
        return x

#     def embed_state(self, game_state):


         
#          # game_state = bz,9,9
#          # game_state_embed = bz,9,9,4



#         game_state_embed = self.obj_emb(
#             game_state.view(-1, game_state.shape[-2] * game_state.shape[-1]))
#         game_state_embed = game_state_embed.view(-1, game_state.shape[-2],
#                                                  game_state.shape[-1],
#                                                  self.emb_sz)

#         if self.wall_embed:
#             indx = (game_state == 1).nonzero()
#             game_state_embed[indx[:, 0], indx[:, 1], indx[:,
#                                                           2]] = self.wall_embed

#         node_embeddings = None
#         if self.use_graph:
#             # print("USING GRAPHS!!!!!")
#             node_embeddings = self.gcn_embed()
#             #print('node embeddings size:', node_embeddings.shape)
#         #    node_embeddings_shrunk = self.state_mapping(relu(node_embeddings))
#             #print('node_emb_shrunk_sz:', node_embeddings_shrunk.shape)
#             for n, embedding in zip(self.nodes.tolist(), node_embeddings):
#                 if n in self.node_to_game_char:
#                     indx = (game_state == self.node_to_game_char[n]).nonzero()
#                     game_state_embed[indx[:, 0], indx[:, 1],
#                                      indx[:, 2]] = embedding

#         return game_state_embed.permute((0, 3, 1, 2)), node_embeddings

    def embed_state(self, game_board,state,node_embeddings):
        node_embeddings = None
        if self.use_graph:
            # node_embeddings = self.gcn_embed()
            for n, embedding in zip(self.nodes.tolist(), node_embeddings):
                if n in self.node_to_game_char:
                    indx = (game_board == self.node_to_game_char[n]).nonzero()
                    state[indx[:, 0], indx[:, 1],
                                     indx[:, 2]] = embedding

        return state.permute((0, 3, 1, 2)), node_embeddings

def self_attention(K, V , Q):
	"""
	STUDENT MUST WRITE:
	This functions runs a single attention head.
	  
	:param K: is [batch_size x window_size_keys x embedding_size]
	:param V: is [batch_size x window_size_values x embedding_size]
	:param Q: is [batch_size x window_size_queries x embedding_size]
	:return: attention
	"""	
	window_size_queries = Q.shape[1] # window size of queries
	window_size_keys = K.shape[1] # window size of keys
	key_size = K.shape[-1] # window size of keys
	#TODO:
	#-1) compute attention weights using queries and key matrices (if use_mask==True, then make sure to add the attention mask before softmax)
	#-2) build new embeddings by applying the attention weights to the values matrices


	# Check lecture slides for how to compute self-attention
	# Remember: 
	# - Q is [batch_size x window_size_queries x embedding_size]
	# - V is [batch_size x window_size_values x embedding_size]
	# - K is [batch_size x window_size_keys x embedding_size]
	# - Mask is [batch_size x window_size_queries x window_size_keys]


	# Here, queries are matmuled with the transpose of keys to produce for every query vector, weights per key vector. 
	# This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
	# Those weights are then used to create linear combinations of the corresponding values for each query.
	# Those queries will become the new embeddings.

	# kdu3 note: 
	# - Q shape is something like (100, 14, 64)
	# - K shape is something like (100, 14, 64)
	# - transpose_b of K is the same as tf.transpose(K, [0, 2, 1]), which gives you a shape of something like (100, 64, 14)
	# - your result is seomthing like (100, 14, 14) - you leave 100 by itself and do a matrix multiplication across (14, 64) x (64, 14)
	
	query_key_scores_raw = torch.matmul(Q, torch.transpose(K,2,1))
	query_key_scores_raw = query_key_scores_raw/torch.sqrt(key_size) # THIS LINE IS NOT NEEDED, just running it to test perplexity/see if it does better
	weights = torch.functional.softmax(query_key_scores_aw)
	build_new_embeds = torch.matmul(weights, V)

	return build_new_embeds

class NodeAtten(torch.nn.Module):
    def __init__(self,node_emb_size,key_size,final_size):
        super(Model, self).__init__()
        self.wQ = torch.nn.Parameter(torch.zeros(2*node_emb_size,key_size))
        self.wK = torch.nn.Parameter(torch.zeros(2*node_emb_size,key_size))
        self.wV = torch.nn.Parameter(torch.zeros(2*node_emb_size,key_size))

        torch.nn.init.xavier_uniform(self.wQ)
        torch.nn.init.xavier_uniform(self.wK)
        torch.nn.init.xavier_uniform(self.wV)

        self.concat_goal = concat_goal

        self.final_layer = torch.nn.Linear(key_size,final_size)

    def forward(self, node_embs, goal_embs):
    	#node embs = b sz x num_nodes x emb sz
    	# goal embs = b sz x emb sz

		goal_embeddings = goal_embs.view(node_embs.goal_embs[-1],1,-1).repeat(1,node_embs.shape[1],1)
		node_embs = torch.cat((node_embs,goal_embs))

		Q = torch.mm(node_embs,wQ)
		K= torch.mm(node_embs,wK)
		V = torch.mm(node_embs,wV)

		return self.final_layer(self_attention(K,V,Q))



