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
                 emb_size=16,
                 use_layers=3):
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
        self.num_types = num_types
        A_raw = adj_matrices
        self.A = [x.to(device) for x in A_raw]
        self.use_graph = use_graph
        self.num_edges = len(A_raw)
        self.use_layers = use_layers
        if self.use_graph:
            if self.atten:
                print('Using attention')
                self.attention = torch.nn.ParameterList([
                    torch.nn.Parameter(A.detach(), requires_grad=True)
                    for A in A_raw
                ])
            self.layer_sizes = [(self.emb_sz, self.emb_sz // self.num_edges)] * self.use_layers
            #self.layer_sizes = [(self.emb_sz, self.emb_sz // self.num_edges),
            #                    (self.emb_sz, self.emb_sz // self.num_edges),
            #                    (self.emb_sz, self.emb_sz // self.num_edges)]
            self.num_layers = len(self.layer_sizes)
            self.weights = [[
                torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
                for (in_dim, out_dim) in self.layer_sizes
            ]
                            for e in range(self.num_edges)]
            for i in range(self.num_edges):
                for j in range(self.num_layers):
                    self.add_module(str((i, j)), self.weights[i][j])
            self.get_node_emb = torch.nn.Embedding(self.n, self.emb_sz)
            if node_glove_embed is not None:
                print("using glove!")
                self.get_node_emb.weight.data.copy_(node_glove_embed)
                self.get_node_emb.requires_grad = True  #False
            self.final_mapping = torch.nn.Linear(self.emb_sz, self.emb_sz)
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
                layer_out.append(torch.mm(weighting, x))  #)
            x = torch.cat([
                relu(self.weights[e][l](type_features))
                for e, type_features in enumerate(layer_out)
            ],
                          axis=1)
        x = self.final_mapping(x)
        return x


def embed_state(game_board, state, node_embeddings, node_2_game_char):
    #node_embeddings = None
    # node_embeddings = self.gcn_embed()
    state = state.permute((0, 2, 3, 1))
    game_board = torch.unsqueeze(game_board, 1)
    emb_sz = node_embeddings.shape[-1]
    num_nodes = node_embeddings.shape[1]
    b_sz = node_embeddings.shape[0]
    grid_rows = game_board.shape[1]
    grid_columns = game_board.shape[2]

    # print("BOARD",game_board.shape)
    #for b in range(len(node_embeddings)):
    node_mask = torch.zeros((num_nodes, b_sz, grid_rows, grid_columns)).cuda()
    for n in range(num_nodes):
        if n in node_2_game_char:
            indx = (game_board == node_2_game_char[n]).nonzero()
            node_mask[n][
                indx[:, 0], indx[:, 1],
                indx[:, 2]] += 1  #node_embeddings[indx[:,0]][n]  #embedding

    node_mask = torch.unsqueeze(torch.transpose(node_mask, 1, 0),
                                -1).repeat(1, 1, 1, 1, emb_sz)
    #print("MASK",node_mask.shape)
    #print(node_embeddings.shape)
    node_embeddings = node_embeddings.view(b_sz, num_nodes, 1, 1, -1).repeat(
        1, 1, grid_rows, grid_columns, 1) * node_mask
    #print(node_embeddings.shape)
    node_embeddings = torch.sum(node_embeddings, 1)

    #print(node_embeddings.shape)
    #print(state.shape)

    state += node_embeddings
    return state.permute((0, 3, 1, 2))  #, node_embeddings

def embed_state2D(game_board, state, node_embeddings, node_2_game_char):
    #node_embeddings = None
    # node_embeddings = self.gcn_embed()
    state = state.permute((0, 2, 3, 1))
    game_board = torch.squeeze(game_board, 1)
    #emb_sz = node_embeddings.shape[-1]
    #num_nodes = node_embeddings.shape[0]
    #b_sz = state.shape[0]
    #grid_rows = game_board.shape[1]
    #grid_columns = game_board.shape[2]
    #print("node_emb", node_embeddings.shape)
    #print("state", state.shape)
    #print("game_board", game_board.shape) 
    for n, embedding in enumerate(node_embeddings):
           if n in node_2_game_char:
                indx = (game_board == node_2_game_char[n]).nonzero()
                state[indx[:, 0], indx[:, 1],indx[:, 2]] += embedding
    return state.permute((0, 3, 1, 2))

def self_attention(K, V, Q):
    """
	STUDENT MUST WRITE:
	This functions runs a single attention head.
	  
	:param K: is [batch_size x window_size_keys x embedding_size]
	:param V: is [batch_size x window_size_values x embedding_size]
	:param Q: is [batch_size x window_size_queries x embedding_size]
	:return: attention
	"""
    window_size_queries = Q.shape[1]  # window size of queries
    window_size_keys = K.shape[1]  # window size of keys
    key_size = K.shape[-1]  # window size of keys

    query_key_scores_raw = torch.matmul(Q, torch.transpose(K, 2, 1))
    query_key_scores_raw = query_key_scores_raw / np.sqrt(
        key_size
    )  # THIS LINE IS NOT NEEDED, just running it to test perplexity/see if it does better
    weights = F.softmax(query_key_scores_raw,-1)
    build_new_embeds = torch.matmul(weights, V)

    return build_new_embeds


class NodeAtten(torch.nn.Module):

    def __init__(self, node_emb_size, key_size):
        super(NodeAtten, self).__init__()
        self.wQ = torch.nn.Parameter(torch.zeros(2 * node_emb_size, key_size))
        self.wK = torch.nn.Parameter(torch.zeros(2 * node_emb_size, key_size))
        self.wV = torch.nn.Parameter(torch.zeros(2 * node_emb_size, key_size))

        torch.nn.init.xavier_uniform(self.wQ)
        torch.nn.init.xavier_uniform(self.wK)
        torch.nn.init.xavier_uniform(self.wV)

    def forward(self, node_embs, goal_embs):
        #node embs = b sz x num_nodes x emb sz
        # goal embs = b sz x emb sz

        node_embs = torch.cat((node_embs, goal_embeddings), axis=-1)
        Q = torch.tensordot(node_embs, self.wQ, dims=([2], [0]))
        K = torch.tensordot(node_embs, self.wK, dims=([2], [0]))
        V = torch.tensordot(node_embs, self.wV, dims=([2], [0]))
        return self_attention(K, V, Q)


class CNN_NODE_ATTEN_BLOCK(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel, node_embed_size,
                 node_2_game_char, use_self_atten, proj_embs):
        super(CNN_NODE_ATTEN_BLOCK, self).__init__()
        self.node_2_game_char = node_2_game_char
        self.conv_layer = torch.nn.Conv2d(in_channels,
                                          out_channels,
                                          kernel_size=(kernel, kernel),
                                          stride=1,
                                          padding=(kernel // 2, kernel // 2))
        self.use_self_atten = use_self_atten
        self.proj_embs = proj_embs

        if self.use_self_atten and self.proj_embs:
            self.node_atten = NodeAtten(node_embed_size, node_embed_size)
            self.final_layer = torch.nn.Linear(node_embed_size, out_channels)
        elif self.proj_embs:
            self.final_layer = torch.nn.Linear(2 * node_embed_size,
                                               out_channels)

    def forward(self, game_board, state, node_embeds, goal_embed):
        conv_out = self.conv_layer(state)  #do a convolution of state
        if self.proj_embs:
            node_embeds = node_embeds.view(1, node_embeds.shape[0],
                                           node_embeds.shape[1]).repeat(
                                               goal_embed.shape[0], 1, 1)
            goal_embeddings = goal_embed.view(goal_embed.shape[0], 1,
                                              -1).repeat(
                                                  1, node_embeds.shape[1], 1)
            if self.use_self_atten:
                out_node_embeds = self.node_atten(node_embeds, goal_embeddings)
            else:
                out_node_embeds = torch.cat((node_embeds, goal_embeddings),
                                            axis=-1)
            out_node_embeds = self.final_layer(out_node_embeds)
            conv_out = embed_state(game_board, conv_out, out_node_embeds,
                                   self.node_2_game_char)
        return conv_out


class CNN_2D_NODE_BLOCK(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel, node_embed_size,
                 node_2_game_char, proj_embs):
        #super(CNN_NODE_ATTEN_BLOCK, self).__init__()
        super(CNN_2D_NODE_BLOCK, self).__init__()
        self.node_2_game_char = node_2_game_char
        self.conv_layer = torch.nn.Conv2d(in_channels,
                                          out_channels,
                                          kernel_size=(kernel, kernel),
                                          stride=1,
                                          padding=(kernel // 2, kernel // 2))
        self.proj_embs = proj_embs

        if self.proj_embs:
            self.final_layer = torch.nn.Linear(node_embed_size,
                                               out_channels)

    def forward(self, game_board, state, node_embeds, goal_embed):
        conv_out = self.conv_layer(state)  #do a convolution of state
        if self.proj_embs:
            #node_embeds = node_embeds.view(1, node_embeds.shape[0],
                                          # node_embeds.shape[1]).repeat(
                                           #    goal_embed.shape[0], 1, 1)
            #goal_embeddings = goal_embed.view(goal_embed.shape[0], 1,
                                            #  -1).repeat(
                                             #     1, node_embeds.shape[1], 1)
           # if self.use_self_atten:
           #     out_node_embeds = self.node_atten(node_embeds, goal_embeddings)
           # else:
            #out_node_embeds = torch.cat((node_embeds, goal_embeddings),
                                     #       axis=-1)
            out_node_embeds = self.final_layer(node_embeds)
            conv_out = embed_state2D(game_board, conv_out, out_node_embeds,
                                   self.node_2_game_char)
        return conv_out

class LINEAR_INV_BLOCK(torch.nn.Module):

    def __init__(self,input_embed_sz, output_size,node_2_game_char,n=9):
        #super(CNN_NODE_ATTEN_BLOCK, self).__init__()
        super(LINEAR_INV_BLOCK, self).__init__()

        self.final_layer = torch.nn.Linear(n*input_embed_sz,
                                               output_size)
        self.n = n
        self.node_2_game_char = node_2_game_char
        self.input_embed_sz = input_embed_sz

    def forward(self, inventory, node_embeds,goal_embed=None):
         inventory = inventory[:,:self.n]
         inv_state = torch.zeros_like(inventory).float()
         inv_state = inv_state.unsqueeze(-1).repeat(1,1,self.input_embed_sz)

         for n, embedding in enumerate(node_embeds):
               if n in self.node_2_game_char:
                   indx = (inventory == self.node_2_game_char[n]).nonzero()
                   inv_state[indx[:, 0], indx[:, 1]] = embedding
         out = self.final_layer(inv_state.view(inv_state.shape[0],-1))
        
         return out

class GOAL_ATTEN_INV_BLOCK(torch.nn.Module):

    def __init__(self,input_embed_sz, output_size,node_2_game_char,n=9):
        #super(CNN_NODE_ATTEN_BLOCK, self).__init__()
        super(GOAL_ATTEN_INV_BLOCK, self).__init__()

        self.input_embed_sz = input_embed_sz
        self.keys = torch.nn.Linear(input_embed_sz,input_embed_sz,bias=False)
        self.final_layer = torch.nn.Linear(input_embed_sz,
                                               output_size)
        self.n = n
        self.node_2_game_char = node_2_game_char

    def forward(self, inventory, node_embeds,goal_embed):
         inventory = inventory[:,:self.n]
         inv_state = torch.zeros_like(inventory).float()
         inv_state = inv_state.unsqueeze(-1).repeat(1,1,self.input_embed_sz)

         for n, embedding in enumerate(node_embeds):
               if n in self.node_2_game_char:
                   indx = (inventory == self.node_2_game_char[n]).nonzero()
                   inv_state[indx[:, 0], indx[:, 1]] = embedding

         K = self.keys(inv_state)
         inv_state = self_attention(K,inv_state,goal_embed.unsqueeze(1))
         out = self.final_layer(inv_state.view(inv_state.shape[0],-1))

         return out
