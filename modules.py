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
                 idx_2_game_char,
                 atten=False,
                 emb_size=16,
                 use_layers=3):
        super(GCN, self).__init__()

        print("starting init")

        self.n = num_nodes
        self.emb_sz = emb_size

        self.atten = atten
        # self.nodes = torch.arange(0, self.n)
        # self.nodes = self.nodes.to(device)
        # self.node_to_game_char = idx_2_game_char
        # self.game_char_to_node = {v:k for k,v in self.node_to_game_char.items()}
        # self.node_to_game_char = idx_2_game_char

        A_raw = adj_matrices

        self.A = [x.to(device) for x in A_raw]
        self.num_edges = len(A_raw)
        self.use_layers = use_layers

        if self.atten:
            print('Using attention')
            self.attention = torch.nn.ParameterList(
                [torch.nn.Parameter(A.detach(), requires_grad=True) for A in A_raw])
        self.layer_sizes = [(self.emb_sz, self.emb_sz // self.num_edges)] * self.use_layers

        self.num_layers = len(self.layer_sizes)
        self.weights = [[
            torch.nn.Linear(in_dim, out_dim, bias=False).to(device)
            for (in_dim, out_dim) in self.layer_sizes
        ]
                        for e in range(self.num_edges)]
        for i in range(self.num_edges):
            for j in range(self.num_layers):
                self.add_module(str((i, j)), self.weights[i][j])

        # self.get_node_emb = torch.nn.Embedding(self.n, self.emb_sz)
        # if pre_init_embeds is not None:
        #     print("using preinit embeds!")
        #     self.get_node_emb.weight.data.copy_(pre_init_embeds)
        #     self.get_node_emb.requires_grad = True
        self.final_mapping = torch.nn.Linear(self.emb_sz, self.emb_sz)

        print("finished initializing")

    def gcn_embed(self, embedded_obj):
        node_embeddings = embedded_obj  #self.get_node_emb(self.nodes)
        #make size 16, first layer
        x = node_embeddings
        for l in range(self.num_layers):
            layer_out = []
            for e in range(self.num_edges):
                if self.atten:
                    weighting = F.normalize(self.attention[e] * self.A[e].float())
                else:
                    weighting = F.normalize(self.A[e].float())
                layer_out.append(torch.mm(weighting, x))  #)
            x = torch.cat([
                relu(self.weights[e][l](type_features)) for e,
                type_features in enumerate(layer_out)
            ],
                          axis=1)
        x = self.final_mapping(x)
        return x


class LINEAR_INV_BLOCK(torch.nn.Module):
    def __init__(self, input_embed_sz, output_size, n=9):
        #super(CNN_NODE_ATTEN_BLOCK, self).__init__()
        super(LINEAR_INV_BLOCK, self).__init__()

        self.final_layer = torch.nn.Linear(n * input_embed_sz, output_size)
        self.n = n
        # self.node_2_game_char = node_2_game_char
        self.input_embed_sz = input_embed_sz

    def forward(self, inventory, node_embeds, goal_embed=None):
        inventory = inventory[:, :self.n]
        # inv_state = torch.zeros_like(inventory).float()
        # inv_state = inv_state.unsqueeze(-1).repeat(1,1,self.input_embed_sz)

        # for n, embedding in enumerate(node_embeds):
        #       if n in self.node_2_game_char:
        #           indx = (inventory == self.node_2_game_char[n]).nonzero()
        #           inv_state[indx[:, 0], indx[:, 1]] = embedding

        inv_flat = inventory.reshape(-1)
        embedded_inv = torch.index_select(node_embeds, 0, inv_flat).reshape(inventory.shape[0], -1)
        out = self.final_layer(embedded_inv)

        return out


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
            node_mask[n][indx[:, 0], indx[:, 1],
                         indx[:, 2]] += 1  #node_embeddings[indx[:,0]][n]  #embedding

    node_mask = torch.unsqueeze(torch.transpose(node_mask, 1, 0), -1).repeat(1, 1, 1, 1, emb_sz)
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


def self_attention(K, V, Q):
    """
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
    weights = F.softmax(query_key_scores_raw, -1)
    build_new_embeds = torch.matmul(weights, V)

    return build_new_embeds


def malmo_build_gcn_param(object_to_char, mode, hier, use_layers, reverse_direction, multi_edge):

    # self.object_to_char = object_to_char

    # if not hier and mode == "ling_prior":
    #     print("{} requires hier".format(mode))
    #     exit()

    #non_node_objects = ["air", "wall"]
    game_nodes = sorted([k for k in object_to_char.keys()],
                        key=lambda x: object_to_char[x])  # if k not in non_node_objects])

    if mode in ["skyline", "skyline_atten"]:  # and not hier:
        edges = [[("pickaxe_item", "stone"), ("axe_item", "log"), ("log", "log_item"),
                  ("hoe_item", "dirt"), ("bucket_item", "water"), ("stone", "cobblestone_item"),
                  ("dirt", "farmland"), ("water", "water_bucket_item")]]
        latent_nodes = []
        use_graph = True
    elif mode in [
            "skyline_hier",
            "skyline_hier_atten",
            "fully_connected",
            "skyline_hier_multi",
            "skyline_hier_multi_atten"
    ] and multi_edge:

        latent_nodes = ["edge_tool", "tool", "material", "product"]
        skyline_edges = [("pickaxe_item", "stone"), ("axe_item", "log"), ("log", "log_item"),
                         ("hoe_item", "dirt"), ("bucket_item", "water"),
                         ("stone", "cobblestone_item"), ("dirt", "farmland"), ("water",
                                                                               "water_bucket_item")]

        hier_edges = [("edge_tool", "pickaxe_item"), ("edge_tool", "axe_item"),
                      ("tool", "hoe_item"), ("tool", "bucket_item"), ("material", "stone"),
                      ("material", "log"), ("material", "dirt"), ("material", "water"),
                      ("product", "log_item"), ("product", "cobblestone_item"),
                      ("product", "farmland"), ("product", "water_bucket_item")]

        edges = [skyline_edges, hier_edges]
        use_graph = True

    elif mode in [
            "skyline_hier",
            "skyline_hier_dw_noGCN",
            "skyline_hier_dw_noGCN_dynamic",
            "skyline_hier_atten",
            "fully_connected"
    ]:
        latent_nodes = ["edge_tool", "tool", "material", "product"]
        edges = [[("pickaxe_item", "stone"), ("axe_item", "log"), ("hoe_item", "dirt"),
                  ("bucket_item", "water"), ("stone", "cobblestone_item"), ("log", "log_item"),
                  ("dirt", "farmland"), ("water", "water_bucket_item"),
                  ("edge_tool", "pickaxe_item"), ("edge_tool", "axe_item"), ("tool", "hoe_item"),
                  ("tool", "bucket_item"), ("material", "stone"), ("material", "log"),
                  ("material", "dirt"), ("material", "water"), ("product", "log_item"),
                  ("product", "cobblestone_item"), ("product", "farmland"), ("product",
                                                                             "water_bucket_item")]]
        use_graph = True

    elif mode in ["skyline_simple", "skyline_simple_atten"]:
        latent_nodes = ["edge_tool", "tool", "material", "product"]
        edges = [[("edge_tool", "pickaxe_item"), ("edge_tool", "axe_item"), ("tool", "hoe_item"),
                  ("tool", "bucket_item"), ("material", "stone"), ("material", "log"),
                  ("material", "dirt"), ("material", "water"), ("product", "log_item"),
                  ("product", "cobblestone_item"), ("product", "farmland"), ("product",
                                                                             "water_bucket_item")]]
        use_graph = True
    # elif mode == "skyline_simple_trash" and hier:
    #     latent_nodes = ["edge_tool", "tool", "material", "product"]
    #     edges = [[("edge_tool", "pickaxe_item"), ("edge_tool", "axe_item"),
    #               ("tool", "hoe_item"), ("tool", "bucket_item"),
    #               ("material", "stone"), ("material", "log"),
    #               ("material", "dirt"), ("material", "water"),
    #               ("product", "log_item"), ("product", "cobblestone_item"),
    #               ("product", "farmland"), ("product", "water_bucket_item"),
    #               ("product", "hoe_item")]]
    #     use_graph = True

    # elif mode == "ling_prior" and hier:
    #   print("Using incorrect mode")
    #   exit()
    # use_graph = True
    # latent_nodes = [
    #     "physical_entity", "abstraction", "substance", "artifact",
    #     "object", "edge_tool", "tool", "instrumentality", "material",
    #     "body_waste"
    # ]

    # edges = [[('substance', 'stone'), ('object', 'stone'),
    #           ('stone', 'stone'), ('material', 'stone'),
    #           ('artifact', 'stone'), ('bucket_item', 'bucket_item'),
    #           ('instrumentality', 'bucket_item'),
    #           ('abstraction', 'bucket_item'), ('object', 'bucket_item'),
    #           ('artifact', 'bucket_item'), ('hoe_item', 'hoe_item'),
    #           ('physical_entity', 'hoe_item'), ('tool', 'hoe_item'),
    #           ('instrumentality', 'hoe_item'), ('object', 'hoe_item'),
    #           ('artifact', 'hoe_item'),
    #           ('physical_entity', 'pickaxe_item'),
    #           ('tool', 'pickaxe_item'),
    #           ('pickaxe_item', 'pickaxe_item'),
    #           ('instrumentality', 'pickaxe_item'),
    #           ('edge_tool', 'pickaxe_item'), ('object', 'pickaxe_item'),
    #           ('artifact', 'pickaxe_item'),
    #           ('physical_entity', 'axe_item'), ('axe_item', 'axe_item'),
    #           ('tool', 'axe_item'), ('instrumentality', 'axe_item'),
    #           ('edge_tool', 'axe_item'), ('object', 'axe_item'),
    #           ('artifact', 'axe_item'),
    #           ('water_bucket_item', 'water_bucket_item'),
    #           ('physical_entity', 'log'), ('log', 'log'),
    #           ('substance', 'log'), ('instrumentality', 'log'),
    #           ('material', 'log'), ('artifact', 'log'),
    #           ('farmland', 'farmland'), ('physical_entity', 'farmland'),
    #           ('object', 'farmland'), ('physical_entity', 'water'),
    #           ('substance', 'water'), ('water', 'water'),
    #           ('body_waste', 'water'), ('artifact', 'water'),
    #           ('physical_entity', 'dirt'), ('dirt', 'dirt'),
    #           ('abstraction', 'dirt'), ('body_waste', 'dirt'),
    #           ('material', 'dirt'),
    #           ('cobblestone_item', 'cobblestone_item'),
    #           ('dirt_item', 'dirt_item'), ('log_item', 'log_item'),
    #           ('farmland_item', 'farmland_item')]]

    elif mode == "cnn" or mode == "embed_bl":
        use_graph = False
        latent_nodes = []
        edges = []

    # total_objects = len(game_nodes + latent_nodes + non_node_objects)
    name_to_node = {e: i for i, e in enumerate(game_nodes + latent_nodes)}
    node_to_game = {i: object_to_char[name] for i, name in enumerate(game_nodes)}
    num_nodes = len(game_nodes + latent_nodes)

    for node in name_to_node.keys():
        if node in object_to_char:
            assert object_to_char[node] == name_to_node[node]
            assert object_to_char[node] == node_to_game[name_to_node[node]]

    assert len(object_to_char.keys()) == len(set([v for k, v in object_to_char.items()]))
    assert len(node_to_game.keys()) == len(set([v for k, v in node_to_game.items()]))

    print("==== GRAPH NETWORK =====")
    print("Game Nodes:", game_nodes)
    print("Latent Nodes:", latent_nodes)
    print("Edges:", edges)

    adjacency = torch.zeros(len(edges), num_nodes, num_nodes, dtype=torch.uint8)
    for edge_type in range(len(edges)):
        for i in range(num_nodes):
            adjacency[edge_type][i][i] = 1
        for s, d in edges[edge_type]:
            adjacency[edge_type][name_to_node[d]][name_to_node[s]] = 1  #corrected transpose!!!!
    if reverse_direction:
        adjacency = torch.transpose(adjacency, 1, 2)
    if mode == "fully_connected":
        adjacency = torch.ones(1, num_nodes, num_nodes, dtype=torch.uint8)

    #if self.use_glove:
    # import json
    # with open("mine_embs_deepwalk.json", "r", encoding="latin-1") as glove_file:
    #     glove_dict = json.load(glove_file)

    #     if mode in ["skyline", "skyline_atten"]:
    #         glove_dict = glove_dict["skyline"]

    #     elif mode in [
    #             "skyline_hier", "skyline_hier_atten", "fully_connected"
    #     ]:
    #         glove_dict = glove_dict["skyline_hier"]

    #     elif mode in ["skyline_simple", "skyline_simple_atten"]:
    #         glove_dict = glove_dict["skyline_simple"]
    #     else:
    #   print("Invalid config use of use_glove")
    #  exit()

    # node_glove_embed = []
    # for node in sorted(game_nodes + latent_nodes,
    #                    key=lambda x: name_2_node[x]):
    #     embed = np.array([
    #         glove_dict[x]
    #         for x in node.replace(" ", "_").replace("-", "_").split("_")
    #     ])
    #     embed = torch.FloatTensor(np.mean(embed, axis=0))
    #     node_glove_embed.append(embed)

    # node_glove_embed = torch.stack(node_glove_embed)
    # else:
    #     node_glove_embed = None

    node_2_name = {v: k for k, v in name_to_node.items()}

    return num_nodes, node_2_name, node_to_game, adjacency, edges, latent_nodes
