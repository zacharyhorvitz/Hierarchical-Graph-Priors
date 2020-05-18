import torch
import numpy as np
import random
from collections import deque, namedtuple
from gcn import GCN

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

class DQN_MALMO_CNN_model(DQN_Base_model):
    """Docstring for DQN CNN model """

    def __init__(self,
                 device,
                 state_space,
                 action_space,
                 num_actions,
                 num_frames=4,
                 final_dense_layer=50,
                 input_shape=(9, 9),
                 mode="skyline", #skyline,ling_prior,embed_bl,cnn
                 hier=False):
        """Defining DQN CNN model
        """
        # initialize all parameters
        print("using MALMO CNN {} {} {}".format(num_frames,final_dense_layer,input_shape))
        super(DQN_MALMO_CNN_model, self).__init__(device, state_space, action_space,
                                            num_actions)
        self.num_frames = num_frames
        self.final_dense_layer = final_dense_layer
        self.input_shape = input_shape
        self.mode = mode
        self.hier = hier

        print("building model")
        self.build_model()

    def build_model(self):
        # output should be in batchsize x num_actions
        # First layer takes in states
        self.body = torch.nn.Sequential(*[
            torch.nn.Conv2d(self.num_frames, 32, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU()#,
            # torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            # torch.nn.ReLU(),
            # torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            # torch.nn.ReLU()
        ])

        final_size = conv2d_size_out(self.input_shape, (3, 3), 1)
        final_size = conv2d_size_out(final_size, (3, 3), 1)
        # final_size = conv2d_size_out(final_size, (3, 3), 1)
        # final_size = conv2d_size_out(final_size, (3, 3), 1)

        self.head = torch.nn.Sequential(*[
            torch.nn.Linear(final_size[0] * final_size[1] *
                            32+4, self.final_dense_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(self.final_dense_layer, self.num_actions)
        ])
        
        self.build_gcn(self.mode,self.hier)

        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_parameters}")

    def build_gcn(self,mode,hier):

        if hier and mode != "skyline" and mode != "ling_prior":
                print("{} incompatible mode with hier".format(mode))
                exit() 
        if not hier and mode == "ling_prior":
                print("{} requires hier".format(mode))
                exit() 

        #name_2_node = {e:i for i,e in enumerate(["stone","pickaxe","cobblestone","log","axe","dirt","farmland","hoe","water","bucket","water_bucket"])}    
        game_nodes = ["stone","pickaxe","cobblestone","log","axe","dirt","farmland","hoe","water","bucket","water_bucket"]
        non_node_objects = ["air","wall"]

        if mode == "skyline" and not hier:
            edges = [ ("pickaxe","stone"),("axe","log"),("hoe","dirt"),("bucket","water"),("stone","cobblestone"),("dirt","farmland"),("water","water_bucket")]
            latent_nodes = []
            use_graph = True
        elif mode == "skyline" and hier:
            latent_nodes = ["edge_tool","non_edge_tool", "material"] 
            edges = [ ("pickaxe","stone"),("axe","log"),("hoe","dirt"),("bucket","water"),("stone","cobblestone"),("dirt","farmland"),("water","water_bucket"), ("edge_tool", "pickaxe"), ("edge_tool", "axe"), ("non_edge_tool", "hoe"), ("non_edge_tool", "bucket"), ("material", "stone"), ("material", "log"), ("material", "dirt"), ("material", "water")]
            use_graph = True

        elif mode == "ling_prior" and hier:
            latent_nodes = ["physical_entity","abstraction","substance","artifact","object","edge_tool","tool","instrumentality","material","body_waste"]
            edges = [('farmland', 'farmland'), ('farmland', 'physical_entity'), ('farmland', 'object'), ('abstraction', 'abstraction'), ('abstraction', 'bucket'), ('abstraction', 'dirt'), ('substance', 'substance'), ('substance', 'stone'), ('substance', 'log'), ('substance', 'water'), ('cobblestone', 'cobblestone'), ('cobblestone', 'stone'), ('cobblestone', 'artifact'), ('cobblestone', 'physical_entity'), ('cobblestone', 'object'), ('axe', 'axe'), ('axe', 'artifact'), ('axe', 'physical_entity'), ('axe', 'object'), ('axe', 'edge_tool'), ('axe', 'tool'), ('axe', 'instrumentality'), ('stone', 'substance'), ('stone', 'cobblestone'), ('stone', 'stone'), ('stone', 'artifact'), ('stone', 'dirt'), ('stone', 'water'), ('stone', 'material'), ('stone', 'object'), ('artifact', 'substance'), ('artifact', 'cobblestone'), ('artifact', 'axe'), ('artifact', 'stone'), ('artifact', 'artifact'), ('artifact', 'bucket'), ('artifact', 'log'), ('artifact', 'hoe'), ('artifact', 'water'), ('artifact', 'pickaxe'), ('bucket', 'abstraction'), ('bucket', 'artifact'), ('bucket', 'bucket'), ('bucket', 'object'), ('bucket', 'instrumentality'), ('dirt', 'abstraction'), ('dirt', 'dirt'), ('dirt', 'physical_entity'), ('dirt', 'body_waste'), ('dirt', 'material'), ('physical_entity', 'farmland'), ('physical_entity', 'cobblestone'), ('physical_entity', 'axe'), ('physical_entity', 'dirt'), ('physical_entity', 'physical_entity'), ('physical_entity', 'log'), ('physical_entity', 'hoe'), ('physical_entity', 'water'), ('physical_entity', 'pickaxe'), ('log', 'substance'), ('log', 'artifact'), ('log', 'physical_entity'), ('log', 'log'), ('log', 'material'), ('log', 'instrumentality'), ('hoe', 'artifact'), ('hoe', 'physical_entity'), ('hoe', 'hoe'), ('hoe', 'object'), ('hoe', 'tool'), ('hoe', 'instrumentality'), ('body_waste', 'dirt'), ('body_waste', 'body_waste'), ('body_waste', 'water'), ('water', 'substance'), ('water', 'artifact'), ('water', 'physical_entity'), ('water', 'body_waste'), ('water', 'water'), ('material', 'substance'), ('material', 'stone'), ('material', 'dirt'), ('material', 'log'), ('material', 'material'), ('object', 'farmland'), ('object', 'cobblestone'), ('object', 'axe'), ('object', 'stone'), ('object', 'bucket'), ('object', 'hoe'), ('object', 'object'), ('object', 'pickaxe'), ('edge_tool', 'axe'), ('edge_tool', 'edge_tool'), ('edge_tool', 'pickaxe'), ('tool', 'axe'), ('tool', 'hoe'), ('tool', 'object'), ('tool', 'tool'), ('tool', 'pickaxe'), ('pickaxe', 'artifact'), ('pickaxe', 'physical_entity'), ('pickaxe', 'object'), ('pickaxe', 'edge_tool'), ('pickaxe', 'tool'), ('pickaxe', 'pickaxe'), ('pickaxe', 'instrumentality'), ('instrumentality', 'axe'), ('instrumentality', 'bucket'), ('instrumentality', 'log'), ('instrumentality', 'hoe'), ('instrumentality', 'pickaxe'), ('instrumentality', 'instrumentality'), ('water_bucket', 'water_bucket')]
            use_graph = True
        elif mode == "cnn" or mode == "embed_bl":
            use_graph = False
            latent_nodes = []
            edges = []
        else:
            print("Invalid configuration")

        total_objects = len(game_nodes + latent_nodes + non_node_objects)
        name_2_node = {e:i for i,e in enumerate(game_nodes+latent_nodes)}
        dict_2_game = {0:2,1:3,2:4,3:5,4:6,5:7,6:8,7:9,8:10,9:11,10:12} 
        num_nodes = len(game_nodes + latent_nodes)

        print("==== GRAPH NETWORK =====")
        print("Game Nodes:",game_nodes)
        print("Latent Nodes:",latent_nodes)
        print("Edges:",edges)

        adjacency = torch.FloatTensor(torch.zeros(num_nodes,num_nodes))
        for i in range(num_nodes):
            adjacency[i][i] = 1.0
        for s,d in edges:
            adjacency[name_2_node[d]][name_2_node[s]] = 1.0 #corrected transpose!!!!
     
        self.gcn = GCN(adjacency,num_nodes,total_objects,dict_2_game,use_graph=use_graph)

        print("...finished initializing gcn")


    def forward(self, state,extract_goal=True):
        if extract_goal:
            goals = state[:,:,:,0][:,0,0].clone().detach().long()
            state = state[:,:,:,1:]

        #print(self.mode,self.hier)

        if self.mode == "skyline" or self.mode == "ling_prior":
            state,node_embeds = self.gcn.embed_state(state.long())
         #   print(state.shape)
            cnn_output = self.body(state)
            cnn_output = cnn_output.reshape(cnn_output.size(0), -1)
            goal_embeddings = node_embeds[[self.gcn.game_char_to_node[g.item()] for g in goals]]
            cnn_output = torch.cat((cnn_output,goal_embeddings),-1)
            q_value = self.head(cnn_output)

        elif self.mode == "embed_bl":
            state,_ = self.gcn.embed_state(state.long())
            cnn_output = self.body(state)
            cnn_output = cnn_output.reshape(cnn_output.size(0), -1)
            goal_embeddings = self.gcn.get_obj_emb(goals)
            cnn_output = torch.cat((cnn_output,goal_embeddings),-1)
            q_value = self.head(cnn_output)

        elif self.mode == "cnn":
            cnn_output = self.body(state)
            cnn_output = cnn_output.reshape(cnn_output.size(0), -1)
            goal_embeddings = self.gcn.get_obj_emb(goals)
            cnn_output = torch.cat((cnn_output,goal_embeddings),-1)
            q_value = self.head(cnn_output)

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
                 hier=False):
        """Defining DQN agent
        """
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        if model_type == "mlp":
            self.online = DQN_MLP_model(device, state_space, action_space,
                                        num_actions)
            self.target = DQN_MLP_model(device, state_space, action_space,
                                        num_actions)
        elif model_type == "cnn":
            assert num_frames
            self.num_frames = num_frames
            self.online = DQN_MALMO_CNN_model(device,
                                        state_space,
                                        action_space,
                                        num_actions,
                                        num_frames=num_frames, mode=mode,hier=hier)
            self.target = DQN_MALMO_CNN_model(device,
                                        state_space,
                                        action_space,
                                        num_actions,
                                        num_frames=num_frames, mode=mode,hier=hier)
           
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
        # state_tensor = torch.Tensor(minibatch.state).to(self.device)
        # next_state_tensor = torch.Tensor(minibatch.next_state).to(self.device) 
        next_state_tensor = torch.Tensor(np.array(minibatch.next_state)).to(self.device) 

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
