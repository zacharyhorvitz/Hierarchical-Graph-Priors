class DUELING_DQN_MALMO_CNN_model(torch.nn.Module):
    """Docstring for DDQN CNN model """

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
        use_glove=False,
        self_attention=False):
        # initialize all parameters
        super(DUELING_DQN_MALMO_CNN_model, self).__init__()
        print("using DDQN MALMO CNN {} {} {}".format(num_frames, final_dense_layer, state_space))
        self.state_space = state_space
        self.action_space = action_space
        self.device = device
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.final_dense_layer = final_dense_layer
        if isinstance(state_space, Space):
            self.input_shape = state_space.shape
        else:
            self.input_shape = state_space
        self.emb_size = emb_size
        self.embedded_state_size = self.emb_size * self.input_shape[0] * self.input_shape[1]
        self.mode = mode
        self.hier = hier

        self.mode = mode
        self.atten = atten
        self.hier = hier
        self.multi_edge = multi_edge
        self.use_glove = use_glove
        self.self_attention = self_attention
        print("building model")
        self.build_model()

    def build_model(self):
        # output should be in batchsize x num_actions
        # First layer takes in states
        node_2_game_char = self.build_gcn(self.mode, self.hier)

        self.block_1 = CNN_NODE_ATTEN_BLOCK(1,
                                            32,
                                            3,
                                            self.emb_size,
                                            node_2_game_char,
                                            self.self_attention,
                                            self.mode not in ['cnn', 'dueling'])
        self.block_2 = CNN_NODE_ATTEN_BLOCK(32,
                                            32,
                                            3,
                                            self.emb_size,
                                            node_2_game_char,
                                            self.self_attention,
                                            self.mode not in ['cnn', 'dueling'])

        final_size = conv2d_size_out(self.input_shape, (3, 3), 1)
        final_size = conv2d_size_out(final_size, (3, 3), 1)

        val_additional_embedding_size = 0
        adv_additional_embedding_size = 0

        if self.mode in ['embed_bl', 'both', 'val_graph']:
            val_additional_embedding_size = self.embedded_state_size
        if self.mode in ['embed_bl', 'both', 'adv_graph']:
            adv_additional_embedding_size = self.embedded_state_size

        self.value_stream = torch.nn.Sequential(*[
            torch.nn.Linear(
                final_size[0] * final_size[1] * 32 + self.embedding_size +
                val_additional_embedding_size,
                self.final_dense_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(self.final_dense_layer, 1)
        ])

        self.advantage_stream = torch.nn.Sequential(*[
            torch.nn.Linear(
                final_size[0] * final_size[1] * 32 + self.embedding_size +
                adv_additional_embedding_size,
                self.final_dense_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(self.final_dense_layer, self.num_actions)
        ])

        self.build_gcn(self.mode, self.hier)

        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_parameters}")

    def build_gcn(self, mode, hier):

        if not hier and mode == "ling_prior":
            raise ValueError("{} requires hier".format(mode))

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
            "farmland_item": 15,
        }
        non_node_objects = ["air", "wall"]
        game_nodes = sorted([k for k in object_to_char.keys() if k not in non_node_objects])

        if mode in ["both", "adv_graph", "val_graph"] and not hier:
            # yapf: disable
            noun_edges = [("pickaxe_item", "stone"), ("stone", "cobblestone_item"),
                          ("axe_item", "log"), ("log", "log_item"),
                          ("hoe_item", "dirt"), ("dirt", "farmland"),
                          ("bucket_item", "water"), ("water", "water_bucket_item")]
            verb_edges = [("mine_1", "pickaxe_item"), ("mine_1", "stone"),
                          ("chop_1", "axe_item"), ("chop_1", "log"),
                          ("farm_1", "hoe_item"), ("farm_1", "dirt"),
                          ("fill_1", "bucket_item"), ("fill_1", "water")]
            # yapf: enable
            edges = [noun_edges + verb_edges]
            latent_nodes = ['mine_1', 'chop_1', 'farm_1', 'fill_1']
            use_graph = True
        elif mode in ["cnn", "embed_bl"]:
            use_graph = False
            latent_nodes = []
            edges = []
        else:
            print("Invalid configuration")
            sys.exit()

        total_objects = len(game_nodes + latent_nodes + non_node_objects)
        name_2_node = {e: i for i, e in enumerate(game_nodes + latent_nodes)}
        node_2_game = {i: object_to_char[name] for i, name in enumerate(game_nodes)}

        num_nodes = len(game_nodes + latent_nodes)

        print("==== GRAPH NETWORK =====")
        print("Game Nodes:", game_nodes)
        print("Latent Nodes:", latent_nodes)
        print("Edges:", edges)

        if mode in ["both", "adv_graph", "val_graph"] and not hier:
            v_adjacency = torch.FloatTensor(torch.zeros(len(edges), num_nodes, num_nodes))
            a_adjacency = torch.FloatTensor(torch.zeros(len(edges), num_nodes, num_nodes))

            for edge_type in range(len(edges)):
                for i in range(num_nodes):
                    v_adjacency[edge_type][i][i] = 1.0
                    a_adjacency[edge_type][i][i] = 1.0

            for edge_type in range(len(edges)):
                for s, d in noun_edges:
                    v_adjacency[name_2_node[d]][name_2_node[s]] = 1.0  # corrected transpose!!!!

            for edge_type in range(len(edges)):
                for s, d in verb_edges:
                    a_adjacency[name_2_node[d]][name_2_node[s]] = 1.0  # corrected transpose!!!!

            if self.use_glove:
                import json
                with open("mine_embs.json", "r", encoding="latin-1") as glove_file:
                    glove_dict = json.load(glove_file)

                    if mode in ["skyline", "skyline_atten"]:
                        glove_dict = glove_dict["skyline"]

                    elif mode in ["skyline_hier", "skyline_hier_atten", "fully_connected"]:
                        glove_dict = glove_dict["skyline_hier"]

                    elif mode in ["skyline_simple", "skyline_simple_atten"]:
                        glove_dict = glove_dict["skyline_simple"]
                    else:
                        print("Invalid config use of use_glove")
                        exit()

                node_glove_embed = []
                for node in sorted(game_nodes + latent_nodes, key=lambda x: name_2_node[x]):
                    embed = np.array([
                        glove_dict[x] for x in node.replace(" ", "_").replace("-", "_").split("_")
                    ])
                    embed = torch.FloatTensor(np.mean(embed, axis=0))
                    node_glove_embed.append(embed)

                node_glove_embed = torch.stack(node_glove_embed)
            else:
                node_glove_embed = None

            self.val_gcn = GCN(v_adjacency,
                               self.device,
                               num_nodes,
                               total_objects,
                               node_2_game,
                               use_graph=use_graph,
                               atten=self.atten,
                               emb_size=self.emb_size,
                               node_glove_embed=node_glove_embed)

            self.adv_gcn = GCN(a_adjacency,
                               self.device,
                               num_nodes,
                               total_objects,
                               node_2_game,
                               use_graph=use_graph,
                               atten=self.atten,
                               emb_size=self.emb_size,
                               node_glove_embed=node_glove_embed)

        elif mode in ["cnn", "embed_bl"]:
            adjacency = torch.FloatTensor(torch.zeros(len(edges), num_nodes, num_nodes))
            for edge_type in range(len(edges)):
                for i in range(num_nodes):
                    adjacency[edge_type][i][i] = 1.0
            for edge_type in range(len(edges)):
                for s, d in edges:
                    adjacency[edge_type][name_2_node[d]][name_2_node[s]] = 1.0

            self.gcn = GCN(adjacency,
                           self.device,
                           num_nodes,
                           total_objects,
                           node_2_game,
                           use_graph=use_graph,
                           atten=self.atten,
                           emb_size=self.emb_size,
                           node_glove_embed=node_glove_embed)
        else:
            print("Invalid configuration")
            sys.exit()

        print("...finished initializing gcn")

    def forward(self, state, extract_goal=True):
        #print(state.shape)
        if extract_goal:
            goals = state[:, :, :, 0][:, 0, 0].clone().detach().long()
            state = state[:, :, :, 1:]

        #   print(goals)

        #print(self.mode,self.hier)
        graph_modes = {"both", "val_graph", "adv_graph"}

        if self.mode in graph_modes:
            node_embeds = self.gcn.gcn_embed()
            goal_embeddings = node_embeds[[self.gcn.game_char_to_node[g.item()] for g in goals]]
        elif self.mode == "cnn":
            node_embeds = None
            goal_embeddings = self.gcn.obj_emb(goals)

        else:
            print("Invalid mode")
            sys.exit()

        if self.mode == "both":
            out = self.block_1(state, state, node_embeds, goal_embeddings)
            out = F.relu(out)
            out = self.block_2(state, out, node_embeds, goal_embeddings)
            out = F.relu(out)
            cnn_output = out.reshape(out.size(0), -1)
            val_state_embeddings, val_node_embeds = self.val_gcn.embed_state(state.long())
            val_state_embeddings = val_state_embeddings.reshape(val_state_embeddings.shape[0], -1)
            val_goal_embeddings = val_node_embeds[[
                self.val_gcn.game_char_to_node[g.item()] for g in goals
            ]]
            val_input = torch.cat((cnn_output, val_goal_embeddings, val_state_embeddings), -1)
            values = self.value_stream(val_input)

            adv_state_embeddings, adv_node_embeds = self.adv_gcn.embed_state(state.long())
            adv_state_embeddings = adv_state_embeddings.reshape(adv_state_embeddings.shape[0], -1)
            adv_goal_embeddings = adv_node_embeds[[
                self.adv_gcn.game_char_to_node[g.item()] for g in goals
            ]]
            adv_input = torch.cat((cnn_output, adv_goal_embeddings, adv_state_embeddings), -1)
            advantage = self.advantage_stream(adv_input)
            q_value = values + (advantage - advantage.mean())

        elif self.mode == "embed_bl":
            out = self.block_1(state, state, node_embeds, goal_embeddings)
            out = F.relu(out)
            out = self.block_2(state, out, node_embeds, goal_embeddings)
            out = F.relu(out)
            cnn_output = out.reshape(out.size(0), -1)
            cnn_output = torch.cat((cnn_output, goal_embeddings), -1)
            state, _ = self.gcn.embed_state(state.long())
            state = state.reshape(state.shape[0], -1)
            values = self.value_stream(cnn_output)
            advantage = self.advantage_stream(cnn_output)
            q_value = values + (advantage - advantage.mean())

        elif self.mode == "cnn":
            out = self.block_1(state, state, node_embeds, goal_embeddings)
            out = F.relu(out)
            out = self.block_2(state, out, node_embeds, goal_embeddings)
            out = F.relu(out)
            cnn_output = out.reshape(out.size(0), -1)
            cnn_output = torch.cat((cnn_output, goal_embeddings), -1)
            values = self.value_stream(cnn_output)
            advantage = self.advantage_stream(cnn_output)
            q_value = values + (advantage - advantage.mean())

        return q_value

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
