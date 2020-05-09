import torch
import numpy as np
import random
from collections import deque, namedtuple
from gcn_hard_code import GCN

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


class DQN_MLP_model(DQN_Base_model):
    """Docstring for DQN MLP model """

    def __init__(self, device, state_space, action_space, num_actions):
        """Defining DQN MLP model
        """
        # initialize all parameters
        super(DQN_MLP_model, self).__init__(device, state_space, action_space,
                                            num_actions)
        # architecture
        self.layer_sizes = [(768, 768), (768, 768), (768, 512)]

        self.build_model()

    def build_model(self):
        # output should be in batchsize x num_actions
        # First layer takes in states
        layers = [
            torch.nn.Linear(self.state_space.shape[0], self.layer_sizes[0][0]),
            torch.nn.ReLU()
        ]
        for size in self.layer_sizes:
            layer = [torch.nn.Linear(size[0], size[1]), torch.nn.ReLU()]
            layers.extend(layer)

        layers.append(torch.nn.Linear(self.layer_sizes[-1][1],
                                      self.num_actions))

        self.body = torch.nn.Sequential(*layers)

        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_parameters}")

    def forward(self, state):
        q_value = self.body(state)
        return q_value


class DQN_CNN_model(DQN_Base_model):
    """Docstring for DQN CNN model """

    def __init__(self,
                 device,
                 state_space,
                 action_space,
                 num_actions,
                 num_frames=4,
                 final_dense_layer=512,
                 input_shape=(84, 84)):
        """Defining DQN CNN model
        """
        # initialize all parameters
        super(DQN_CNN_model, self).__init__(device, state_space, action_space,
                                            num_actions)
        self.num_frames = num_frames
        self.final_dense_layer = final_dense_layer
        self.input_shape = input_shape

        self.build_model()

    def build_model(self):
        # output should be in batchsize x num_actions
        # First layer takes in states
        self.body = torch.nn.Sequential(*[
            torch.nn.Conv2d(self.num_frames, 32, kernel_size=(8, 8), stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU()
        ])

        final_size = conv2d_size_out(self.input_shape, (8, 8), 4)
        final_size = conv2d_size_out(final_size, (4, 4), 2)
        final_size = conv2d_size_out(final_size, (3, 3), 1)

        self.head = torch.nn.Sequential(*[
            torch.nn.Linear(final_size[0] * final_size[1] *
                            64, self.final_dense_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(self.final_dense_layer, self.num_actions)
        ])

        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_parameters}")

    def forward(self, state):
        cnn_output = self.body(state)
        q_value = self.head(cnn_output.reshape(cnn_output.size(0), -1))
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

class DQN_MALMO_CNN_model(DQN_Base_model):
    """Docstring for DQN CNN model """

    def __init__(self,
                 device,
                 state_space,
                 action_space,
                 num_actions,
                 num_frames=4,
                 final_dense_layer=50,
                 input_shape=(9, 9)):
        """Defining DQN CNN model
        """
        # initialize all parameters
        print("using MALMO CNN {} {} {}".format(num_frames,final_dense_layer,input_shape))
        super(DQN_MALMO_CNN_model, self).__init__(device, state_space, action_space,
                                            num_actions)
        self.num_frames = num_frames
        self.final_dense_layer = final_dense_layer
        self.input_shape = input_shape

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
        print("gcn model")

        # num_nodes = 3
        # total_objects = 5
        # dict_2_game = {0:2,1:3,2:4} #0->stone, 1->pickaxe, 2->cobblestone
        # adjacency = torch.FloatTensor([[1.0,0.0,1.0],[1.0,1.0,.00],[0.0,0.0,1.0]]) #.cuda()
        # self.gcn = GCN(adjacency,num_nodes,total_objects,dict_2_game)

        total_objects = 13 #nodes, plus air and wall
        num_nodes = total_objects-2
        dict_2_game = {0:2,1:3,2:4,3:5,4:6,5:7,6:8,7:9,8:10,9:11,10:12} 
        name_2_node = {e:i for i,e in enumerate(["stone","pickaxe","cobblestone","log","axe","dirt","farmland","hoe","water","bucket","water_bucket"])}

        edges = [ ("pickaxe","stone"),("axe","log"),("hoe","dirt"),("bucket","water"),("stone","cobblestone"),("dirt","farmland"),("water","water_bucket")]

        adjacency = torch.FloatTensor(torch.zeros(num_nodes,num_nodes)) #
        for i in range(num_nodes):
            adjacency[i][i] = 1.0
        for s,d in edges:
            adjacency[name_2_node[d]][name_2_node[s]] = 1.0 #corrected transpose!!!!

        # print(adjacency)
        # exit()
        #0->stone
        #1->pickaxe
        #2->cobblestone
        #3->log
        #4->axe
        #5->dirt
        #6->farmland
        #7->hoe
        #8->water
        #9->bucket
        #10->water_bucket
     

        self.gcn = GCN(adjacency,num_nodes,total_objects,dict_2_game)
        print("finished initializing gcn")


        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_parameters}")

    def forward(self, state,extract_goal=True,use_graph=False,embedding_baseline=False):
        if extract_goal:
            goals = state[:,:,:,0][:,0,0].clone().detach().long()
            state = state[:,:,:,1:]

        # print(state[0][0])
        # print(goals[0])
        if use_graph:
            state,node_embeds = self.gcn.embed_state(state.long(),add_graph_embs=use_graph)
            cnn_output = self.body(state)
            cnn_output = cnn_output.reshape(cnn_output.size(0), -1)
            goal_embeddings = node_embeds[[self.gcn.game_char_to_node[g.item()] for g in goals]]
            cnn_output = torch.cat((cnn_output,goal_embeddings),-1)
            q_value = self.head(cnn_output)
        elif embedding_baseline:
            state,node_embeds = self.gcn.embed_state(state.long(),add_graph_embs=use_graph)
            cnn_output = self.body(state)
            cnn_output = cnn_output.reshape(cnn_output.size(0), -1)
            goal_embeddings = self.gcn.get_obj_emb(goals)
            cnn_output = torch.cat((cnn_output,goal_embeddings),-1)
            q_value = self.head(cnn_output)
        else:
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
                 num_frames=None):
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
                                        num_frames=num_frames)
            self.target = DQN_MALMO_CNN_model(device,
                                        state_space,
                                        action_space,
                                        num_actions,
                                        num_frames=num_frames)
           
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
