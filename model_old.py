import torch
import sys
import numpy as np
import random
from collections import deque, namedtuple
import torch.nn.functional as F
from torch.nn.functional import relu

from utils import sync_networks, conv2d_size_out

Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])


class DQN_Base_model(torch.nn.Module):
    """Docstring for DQN MLP model """

    def __init__(self, device, state_space, action_space, num_actions):
        """Defining DQN MLP model
        """
        # initialize all parameters
        super(DQN_Base_model, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.device = device
        self.num_actions = num_actions

    def build_model(self):
        # output should be in batchsize x num_actions
        raise NotImplementedError

    def forward(self, state):
        raise NotImplementedError

    def max_over_actions(self, state):
        state = state.to(self.device)
        return torch.max(self(state), dim=1)

    def argmax_over_actions(self, state):
        state = state.to(self.device)
        return torch.argmax(self(state), dim=1)

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.Tensor(state).unsqueeze(0)
                action_tensor = self.argmax_over_actions(state_tensor)
                action = action_tensor.cpu().detach().numpy().flatten()[0]
                assert self.action_space.contains(action)
            return action


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
    def embed_state(self, game_state):
        game_state_embed = self.obj_emb(
            game_state.view(-1, game_state.shape[-2] * game_state.shape[-1]))
        game_state_embed = game_state_embed.view(-1, game_state.shape[-2],
                                                 game_state.shape[-1],
                                                 self.emb_sz)

        if self.wall_embed:
            indx = (game_state == 1).nonzero()
            game_state_embed[indx[:, 0], indx[:, 1], indx[:,
                                                          2]] = self.wall_embed

        node_embeddings = None
        if self.use_graph:
            # print("USING GRAPHS!!!!!")
            node_embeddings = self.gcn_embed()
            #print('node embeddings size:', node_embeddings.shape)
        #    node_embeddings_shrunk = self.state_mapping(relu(node_embeddings))
            #print('node_emb_shrunk_sz:', node_embeddings_shrunk.shape)
            for n, embedding in zip(self.nodes.tolist(), node_embeddings):
                if n in self.node_to_game_char:
                    indx = (game_state == self.node_to_game_char[n]).nonzero()
                    game_state_embed[indx[:, 0], indx[:, 1],
                                     indx[:, 2]] = embedding

        return game_state_embed.permute((0, 3, 1, 2)), node_embeddings


class DQN_MALMO_CNN_model(DQN_Base_model):
    """Docstring for DQN CNN model """

    def __init__(
        self,
        device,
        state_space,
        action_space,
        num_actions,
        num_frames=1,
        final_dense_layer=50,
        #input_shape=(9, 9),
        mode="skyline",  #skyline,ling_prior,embed_bl,cnn
        hier=False,
        atten=False, 
        one_layer=False, 
        emb_size=16,
        multi_edge=False,
        use_glove=False):
        """Defining DQN CNN model
        """
        # initialize all parameters
        print("using MALMO CNN {} {} {}".format(num_frames, final_dense_layer,
                                                state_space))
        super(DQN_MALMO_CNN_model, self).__init__(device, state_space,
                                                  action_space, num_actions)
        self.num_frames = num_frames
        self.final_dense_layer = final_dense_layer
        self.input_shape = state_space
        self.mode = mode
        self.atten = atten
        self.hier = hier
        self.emb_size = emb_size
        self.multi_edge = multi_edge
        self.use_glove = use_glove
        print("building model")
        self.build_model()

    def build_model(self):
        # output should be in batchsize x num_actions
        # First layer takes in states
        self.body = torch.nn.Sequential(*[
            torch.nn.Conv2d(self.num_frames, 32, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU()  #,
            # torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            # torch.nn.ReLU()
        ])
        #print(self.input_shape) 
        #exit()
        final_size = conv2d_size_out(self.input_shape, (3, 3), 1)
        #print(final_size)
        final_size = conv2d_size_out(final_size, (3, 3), 1)

        #print(final_size)
        # final_size = conv2d_size_out(final_size, (3, 3), 1)
        # final_size = conv2d_size_out(final_size, (3, 3), 1)
        #print(self.emb_size)
        self.head = torch.nn.Sequential(*[
            torch.nn.Linear(final_size[0] * final_size[1] * 32 +
                            self.emb_size, self.final_dense_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(self.final_dense_layer, self.num_actions)
        ])
 

        self.build_gcn(self.mode, self.hier)

        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_parameters}")

    def build_gcn(self, mode, hier):

        #if hier and mode != "skyline" and mode != "ling_prior" and mode != "skyline_simple":
        #    print("{} incompatible mode with hier".format(mode))
        #    exit()
        if not hier and mode == "ling_prior":
            print("{} requires hier".format(mode))
            exit()

        #name_2_node = {e:i for i,e in enumerate(["stone","pickaxe","cobblestone","log","axe","dirt","farmland","hoe","water","bucket","water_bucket"])}
        #game_nodes = ["stone","pickaxe","cobblestone","log","axe","dirt","farmland","hoe","water","bucket","water_bucket"]
        #game_nodes = ["stone","pickaxe_item","cobblestone","log","axe","dirt","farmland","hoe","water","bucket","water_bucket"]
        object_to_char = {
            "air": 0,
            "wall": 1,
            "stone": 2,
            "pickaxe_item": 3,
            "cobblestone_item": 4,
            "log": 5,
            "axe_item": 6,
            "dirt": 7,
            "farmland": 8,
            "hoe_item": 9,
            "water": 10,
            "bucket_item": 11,
            "water_bucket_item": 12,
            "log_item": 13,
            "dirt_item": 14,
            "farmland_item": 15
        }
        #object_to_char = {"air":0,"bedrock":1,"stone":2,"pickaxe_item":3,"cobblestone_item":4,"log":5,"axe_item":6,"dirt":7,"farmland":8,"hoe_item":9,"water":10,"bucket_item":11,"water_bucket_item":12,"log_item":13,"dirt_item":14,"farmland_item":15}
        non_node_objects = ["air", "wall"]
        game_nodes = sorted(
            [k for k in object_to_char.keys() if k not in non_node_objects])

        if mode in ["skyline", "skyline_atten"] and not hier:
            # edges = [ ("pickaxe","stone"),("axe","log"),("hoe","dirt"),("bucket","water"),("stone","cobblestone"),("dirt","farmland"),("water","water_bucket")]
            edges = [[("pickaxe_item","stone"),("axe_item","log"),("log","log_item"),("hoe_item","dirt"),("bucket_item","water"),("stone","cobblestone_item"),("dirt","farmland"),("water","water_bucket_item")]]
            # edges = [("pickaxe_item", "stone"), ("axe_item", "log"),
            # ("hoe_item", "dirt"), ("bucket_item", "water"),
            # ("stone", "cobblestone_item"), ("dirt", "farmland"),
            # ("water", "water_bucket_item")]
            latent_nodes = []
            use_graph = True
        #else:
        #    exit()
        elif mode in ["skyline_hier", "skyline_hier_atten", "fully_connected", "skyline_hier_multi", "skyline_hier_multi_atten"] and self.multi_edge:

            latent_nodes = ["edge_tool","tool", "material","product"]
            skyline_edges = [("pickaxe_item","stone"),("axe_item","log"),("log","log_item"),("hoe_item","dirt"),("bucket_item","water"),("stone","cobblestone_item"),("dirt","farmland"),("water","water_bucket_item")]

            hier_edges = [("edge_tool", "pickaxe_item"), ("edge_tool", "axe_item"), ("tool", "hoe_item"), ("tool", "bucket_item"), ("material", "stone"), ("material", "log"), ("material", "dirt"), ("material", "water"),("product","log_item"),("product","cobblestone_item"),("product","farmland"),("product","water_bucket_item")]
 
            edges = [skyline_edges,hier_edges]
            use_graph = True

        elif mode in ["skyline_hier", "skyline_hier_atten", "fully_connected"]:
            latent_nodes = ["edge_tool","tool", "material","product"]
            edges =[[ ("pickaxe_item","stone"),("axe_item","log"),("hoe_item","dirt"),("bucket_item","water"),("stone","cobblestone_item"),("log","log_item"),("dirt","farmland"),("water","water_bucket_item"), ("edge_tool", "pickaxe_item"), ("edge_tool", "axe_item"), ("tool", "hoe_item"), ("tool", "bucket_item"), ("material", "stone"), ("material", "log"), ("material", "dirt"), ("material", "water"),("product","log_item"),("product","cobblestone_item"),("product","farmland"),("product","water_bucket_item")]]
            use_graph = True

        elif mode in ["skyline_simple", "skyline_simple_atten"]:
            latent_nodes = ["edge_tool","tool", "material","product"]
            edges = [[("edge_tool", "pickaxe_item"), ("edge_tool", "axe_item"), ("tool", "hoe_item"), ("tool", "bucket_item"), ("material", "stone"), ("material", "log"), ("material", "dirt"), ("material", "water"),("product","log_item"),("product","cobblestone_item"),("product","farmland"),("product","water_bucket_item")]]
            use_graph = True
        elif mode == "skyline_simple_trash" and hier:
            latent_nodes = ["edge_tool","tool", "material","product"]
            edges = [[("edge_tool", "pickaxe_item"), ("edge_tool", "axe_item"), ("tool", "hoe_item"), ("tool", "bucket_item"), ("material", "stone"), ("material", "log"), ("material", "dirt"), ("material", "water"),("product","log_item"),("product","cobblestone_item"),("product","farmland"),("product","water_bucket_item"),("product","hoe_item")]]
            use_graph = True
   
        elif mode == "ling_prior" and hier:
            use_graph = True
            latent_nodes = ["physical_entity","abstraction","substance","artifact","object","edge_tool","tool","instrumentality","material","body_waste"]

            edges = [[('substance', 'stone'), ('object', 'stone'), ('stone', 'stone'), ('material', 'stone'), ('artifact', 'stone'), ('bucket_item', 'bucket_item'), ('instrumentality', 'bucket_item'), ('abstraction', 'bucket_item'), ('object', 'bucket_item'), ('artifact', 'bucket_item'), ('hoe_item', 'hoe_item'), ('physical_entity', 'hoe_item'), ('tool', 'hoe_item'), ('instrumentality', 'hoe_item'), ('object', 'hoe_item'), ('artifact', 'hoe_item'), ('physical_entity', 'pickaxe_item'), ('tool', 'pickaxe_item'), ('pickaxe_item', 'pickaxe_item'), ('instrumentality', 'pickaxe_item'), ('edge_tool', 'pickaxe_item'), ('object', 'pickaxe_item'), ('artifact', 'pickaxe_item'), ('physical_entity', 'axe_item'), ('axe_item', 'axe_item'), ('tool', 'axe_item'), ('instrumentality', 'axe_item'), ('edge_tool', 'axe_item'), ('object', 'axe_item'), ('artifact', 'axe_item'), ('water_bucket_item', 'water_bucket_item'), ('physical_entity', 'log'), ('log', 'log'), ('substance', 'log'), ('instrumentality', 'log'), ('material', 'log'), ('artifact', 'log'), ('farmland', 'farmland'), ('physical_entity', 'farmland'), ('object', 'farmland'), ('physical_entity', 'water'), ('substance', 'water'), ('water', 'water'), ('body_waste', 'water'), ('artifact', 'water'), ('physical_entity', 'dirt'), ('dirt', 'dirt'), ('abstraction', 'dirt'), ('body_waste', 'dirt'), ('material', 'dirt'), ('cobblestone_item', 'cobblestone_item'), ('dirt_item', 'dirt_item'), ('log_item', 'log_item'), ('farmland_item', 'farmland_item')]]

# edges = [('object', 'stone'), ('material', 'stone'), ('substance', 'stone'), ('artifact', 'stone'), ('stone', 'stone'), ('object', 'bucket_item'), ('abstraction', 'bucket_item'), ('artifact', 'bucket_item'), ('instrumentality', 'bucket_item'), ('bucket_item', 'bucket_item'), ('physical_entity', 'hoe_item'), ('object', 'hoe_item'), ('hoe_item', 'hoe_item'), ('artifact', 'hoe_item'), ('tool', 'hoe_item'), ('instrumentality', 'hoe_item'), ('physical_entity', 'pickaxe_item'), ('object', 'pickaxe_item'), ('artifact', 'pickaxe_item'), ('tool', 'pickaxe_item'), ('edge_tool', 'pickaxe_item'), ('pickaxe_item', 'pickaxe_item'), ('instrumentality', 'pickaxe_item'), ('physical_entity', 'axe_item'), ('object', 'axe_item'), ('axe_item', 'axe_item'), ('artifact', 'axe_item'), ('tool', 'axe_item'), ('edge_tool', 'axe_item'), ('instrumentality', 'axe_item'), ('water_bucket_item', 'water_bucket_item'), ('physical_entity', 'log'), ('material', 'log'), ('substance', 'log'), ('artifact', 'log'), ('log', 'log'), ('instrumentality', 'log'), ('farmland', 'farmland'), ('physical_entity', 'farmland'), ('object', 'farmland'), ('physical_entity', 'cobblestone'), ('object', 'cobblestone'), ('artifact', 'cobblestone'), ('cobblestone', 'cobblestone'), ('stone', 'cobblestone'), ('physical_entity', 'water'), ('substance', 'water'), ('artifact', 'water'), ('body_waste', 'water'), ('water', 'water'), ('physical_entity', 'dirt'), ('abstraction', 'dirt'), ('material', 'dirt'), ('dirt', 'dirt'), ('body_waste', 'dirt'), ('cobblestone_item', 'cobblestone_item'), ('dirt_item', 'dirt_item'), ('log_item', 'log_item'), ('farmland_item', 'farmland_item')]




#      latent_nodes = ["physical_entity","abstraction","substance","artifact","object","edge_tool","tool","instrumentality","material","body_waste"]
#     edges = [('farmland', 'farmland'), ('farmland', 'physical_entity'), ('farmland', 'object'), ('abstraction', 'abstraction'), ('abstraction', 'bucket'), ('abstraction', 'dirt'), ('substance', 'substance'), ('substance', 'stone'), ('substance', 'log'), ('substance', 'water'), ('cobblestone', 'cobblestone'), ('cobblestone', 'stone'), ('cobblestone', 'artifact'), ('cobblestone', 'physical_entity'), ('cobblestone', 'object'), ('axe', 'axe'), ('axe', 'artifact'), ('axe', 'physical_entity'), ('axe', 'object'), ('axe', 'edge_tool'), ('axe', 'tool'), ('axe', 'instrumentality'), ('stone', 'substance'), ('stone', 'cobblestone'), ('stone', 'stone'), ('stone', 'artifact'), ('stone', 'dirt'), ('stone', 'water'), ('stone', 'material'), ('stone', 'object'), ('artifact', 'substance'), ('artifact', 'cobblestone'), ('artifact', 'axe'), ('artifact', 'stone'), ('artifact', 'artifact'), ('artifact', 'bucket'), ('artifact', 'log'), ('artifact', 'hoe'), ('artifact', 'water'), ('artifact', 'pickaxe'), ('bucket', 'abstraction'), ('bucket', 'artifact'), ('bucket', 'bucket'), ('bucket', 'object'), ('bucket', 'instrumentality'), ('dirt', 'abstraction'), ('dirt', 'dirt'), ('dirt', 'physical_entity'), ('dirt', 'body_waste'), ('dirt', 'material'), ('physical_entity', 'farmland'), ('physical_entity', 'cobblestone'), ('physical_entity', 'axe'), ('physical_entity', 'dirt'), ('physical_entity', 'physical_entity'), ('physical_entity', 'log'), ('physical_entity', 'hoe'), ('physical_entity', 'water'), ('physical_entity', 'pickaxe'), ('log', 'substance'), ('log', 'artifact'), ('log', 'physical_entity'), ('log', 'log'), ('log', 'material'), ('log', 'instrumentality'), ('hoe', 'artifact'), ('hoe', 'physical_entity'), ('hoe', 'hoe'), ('hoe', 'object'), ('hoe', 'tool'), ('hoe', 'instrumentality'), ('body_waste', 'dirt'), ('body_waste', 'body_waste'), ('body_waste', 'water'), ('water', 'substance'), ('water', 'artifact'), ('water', 'physical_entity'), ('water', 'body_waste'), ('water', 'water'), ('material', 'substance'), ('material', 'stone'), ('material', 'dirt'), ('material', 'log'), ('material', 'material'), ('object', 'farmland'), ('object', 'cobblestone'), ('object', 'axe'), ('object', 'stone'), ('object', 'bucket'), ('object', 'hoe'), ('object', 'object'), ('object', 'pickaxe'), ('edge_tool', 'axe'), ('edge_tool', 'edge_tool'), ('edge_tool', 'pickaxe'), ('tool', 'axe'), ('tool', 'hoe'), ('tool', 'object'), ('tool', 'tool'), ('tool', 'pickaxe'), ('pickaxe', 'artifact'), ('pickaxe', 'physical_entity'), ('pickaxe', 'object'), ('pickaxe', 'edge_tool'), ('pickaxe', 'tool'), ('pickaxe', 'pickaxe'), ('pickaxe', 'instrumentality'), ('instrumentality', 'axe'), ('instrumentality', 'bucket'), ('instrumentality', 'log'), ('instrumentality', 'hoe'), ('instrumentality', 'pickaxe'), ('instrumentality', 'instrumentality'), ('water_bucket', 'water_bucket')]
#     use_graph = True
        elif mode == "cnn" or mode == "embed_bl":
            use_graph = False
            latent_nodes = []
            edges = []

        total_objects = len(game_nodes + latent_nodes + non_node_objects)
        name_2_node = {e: i for i, e in enumerate(game_nodes + latent_nodes)}
        node_2_game = {
            i: object_to_char[name] for i, name in enumerate(game_nodes)
        }  #{0:2,1:3,2:4,3:5,4:6,5:7,6:8,7:9,8:10,9:11,10:12}
        num_nodes = len(game_nodes + latent_nodes)

        print("==== GRAPH NETWORK =====")
        print("Game Nodes:", game_nodes)
        print("Latent Nodes:", latent_nodes)
        print("Edges:", edges)

        adjacency = torch.FloatTensor(torch.zeros(len(edges),num_nodes, num_nodes))
        #if "simple" in mode: adjacency = torch.FloatTensor(torch.ones(num_nodes, num_nodes))
        for edge_type in range(len(edges)):
            for i in range(num_nodes):
                adjacency[edge_type][i][i] = 1.0
            for s, d in edges[edge_type]:
                adjacency[edge_type][name_2_node[d]][
                name_2_node[s]] = 1.0  #corrected transpose!!!!
        if mode == "fully_connected":
            adjacency = torch.FloatTensor(torch.ones(1,num_nodes, num_nodes))

        if self.use_glove:
             import json
             with open("mine_embs.json","r",encoding="latin-1") as glove_file:
                glove_dict = json.load(glove_file)

                if mode in ["skyline","skyline_atten"]:
                    glove_dict = glove_dict["skyline"]

                elif mode in ["skyline_hier", "skyline_hier_atten", "fully_connected"]:
                    glove_dict = glove_dict["skyline_hier"]
                
                elif mode in ["skyline_simple", "skyline_simple_atten"]:
                    glove_dict = glove_dict["skyline_simple"]
                else:
                    print("Invalid config use of use_glove")
                    exit()

             node_glove_embed = []
             for node in sorted(game_nodes+latent_nodes,key=lambda x: name_2_node[x]):
                  embed = np.array([glove_dict[x] for x in node.replace(" ","_").replace("-","_").split("_")])
                  embed = torch.FloatTensor(np.mean(embed,axis=0))
                  node_glove_embed.append(embed)
            
             node_glove_embed = torch.stack(node_glove_embed)
        else:
             node_glove_embed = None

        self.gcn = GCN(adjacency,
                       self.device,
                       num_nodes,
                       total_objects,
                       node_2_game,
                       use_graph=use_graph,
                       atten=self.atten,
                       emb_size=self.emb_size,
                       node_glove_embed=node_glove_embed)

        print("...finished initializing gcn")

    def forward(self, state, extract_goal=True):
        #print(state.shape)
        if extract_goal:
            goals = state[:, :, :, 0][:, 0, 0].clone().detach().long()
            state = state[:, :, :, 1:]
        #   print(goals)

        #print(self.mode,self.hier)

        graph_modes = [
                "skyline", "skyline_hier", "skyline_atten",
                "skyline_hier_atten", "skyline_simple", "skyline_simple_atten",
                "skyline_simple_trash", "ling_prior", "fully_connected",
                "skyline_hier_multi", "skyline_hier_multi_atten"
        ]

        if self.mode in graph_modes:
            state, node_embeds = self.gcn.embed_state(state.long())
            #   print(state.shape)
            cnn_output = self.body(state)
            cnn_output = cnn_output.reshape(cnn_output.size(0), -1)
            goal_embeddings = node_embeds[[
                self.gcn.game_char_to_node[g.item()] for g in goals
            ]]
            cnn_output = torch.cat((cnn_output, goal_embeddings), -1)
            q_value = self.head(cnn_output)

        elif self.mode == "embed_bl":
            state, _ = self.gcn.embed_state(state.long())
            cnn_output = self.body(state)
            cnn_output = cnn_output.reshape(cnn_output.size(0), -1)
            goal_embeddings = self.gcn.obj_emb(goals)
            cnn_output = torch.cat((cnn_output, goal_embeddings), -1)
            q_value = self.head(cnn_output)

        elif self.mode == "cnn":
            #print(self.num_frames)
            cnn_output = self.body(state)
            cnn_output = cnn_output.reshape(cnn_output.size(0), -1)
            goal_embeddings = self.gcn.obj_emb(goals)
            cnn_output = torch.cat((cnn_output, goal_embeddings), -1)
            q_value = self.head(cnn_output)
        else:
            print("Invalid mode")
            sys.exit()
        return q_value

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.Tensor(state).unsqueeze(0)
                action_tensor = self.argmax_over_actions(state_tensor)
                action = action_tensor.cpu().detach().numpy().flatten()[0]
                assert self.action_space.contains(action)
            return action


class DQN_agent:
    """Docstring for DQN agent """

    def __init__(self,
                 device,
                 state_space,
                 action_space,
                 num_actions,
                 target_moving_average,
                 gamma,
                 replay_buffer_size,
                 epsilon_decay,
                 epsilon_decay_end,
                 warmup_period,
                 double_DQN,
                 model_type="mlp",
                 num_frames=None,
                 mode="skyline",
                 hier=False,
                 atten=False,
                 one_layer=False, 
                 emb_size=16,
                 multi_edge=False,
                 use_glove=False):
        """Defining DQN agent
        """
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        if model_type == "cnn":
            assert num_frames
            self.num_frames = num_frames
            self.online = DQN_MALMO_CNN_model(device,
                                              state_space,
                                              action_space,
                                              num_actions,
                                              num_frames=num_frames,
                                              mode=mode,
                                              hier=hier,
                                              atten=atten,
                                              multi_edge=multi_edge,
                                              use_glove=use_glove,
                                              emb_size=emb_size)
            self.target = DQN_MALMO_CNN_model(device,
                                              state_space,
                                              action_space,
                                              num_actions,
                                              num_frames=num_frames,
                                              mode=mode,
                                              hier=hier,
                                              atten=atten,
                                              multi_edge=multi_edge,
                                              use_glove=use_glove,
                                              emb_size=emb_size)


            #stone's adjacencies [1,0,1]
            #pickaxe's adjacencies [1,1,0]
            #cobblestones's adjacencies [0,0,1]
            # new_state = test.embed_state(torch.ones((1,10,10)).long())
            # print(new_state.shape)

        else:
            raise NotImplementedError(model_type)

        self.online = self.online.to(device)
        self.target = self.target.to(device)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.gamma = gamma
        self.target_moving_average = target_moving_average
        self.epsilon_decay = epsilon_decay
        self.epsilon_decay_end = epsilon_decay_end
        self.warmup_period = warmup_period
        self.device = device

        self.model_type = model_type
        self.double_DQN = double_DQN

    def loss_func(self, minibatch, writer=None, writer_step=None):
        # Make tensors
        state_tensor = torch.Tensor(np.array(minibatch.state)).to(self.device)
        next_state_tensor = torch.Tensor(np.array(minibatch.next_state)).to(
            self.device)

        action_tensor = torch.Tensor(minibatch.action).to(self.device)
        reward_tensor = torch.Tensor(minibatch.reward).to(self.device)
        done_tensor = torch.Tensor(minibatch.done).to(self.device)

        # Get q value predictions
        q_pred_batch = self.online(state_tensor).gather(
            dim=1, index=action_tensor.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_DQN:
                selected_actions = self.online.argmax_over_actions(
                    next_state_tensor)
                q_target = self.target(next_state_tensor).gather(
                    dim=1,
                    index=selected_actions.long().unsqueeze(1)).squeeze(1)
            else:
                q_target = self.target.max_over_actions(
                    next_state_tensor.detach()).values

        q_label_batch = reward_tensor + (self.gamma) * (1 -
                                                        done_tensor) * q_target
        q_label_batch = q_label_batch.detach()

        # Logging
        if writer:
            writer.add_scalar('training/batch_q_label', q_label_batch.mean(),
                              writer_step)
            writer.add_scalar('training/batch_q_pred', q_pred_batch.mean(),
                              writer_step)
            writer.add_scalar('training/batch_reward', reward_tensor.mean(),
                              writer_step)
        return torch.nn.functional.mse_loss(q_label_batch, q_pred_batch)

    def sync_networks(self):
        sync_networks(self.target, self.online, self.target_moving_average)

    def set_epsilon(self, global_steps, writer=None):
        if global_steps < self.warmup_period:
            self.online.epsilon = 1
            self.target.epsilon = 1
        else:
            self.online.epsilon = max(
                self.epsilon_decay_end,
                1 - (global_steps - self.warmup_period) / self.epsilon_decay)
            self.target.epsilon = max(
                self.epsilon_decay_end,
                1 - (global_steps - self.warmup_period) / self.epsilon_decay)
        if writer:
            writer.add_scalar('training/epsilon', self.online.epsilon,
                              global_steps)
