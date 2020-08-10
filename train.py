import random
import time
import os
import json

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
#import minerl

from utils import parse_args, append_timestamp

#https://github.com/pytorch/pytorch/issues/31554

from envs.malmo_numpy_env import MalmoEnvSpecial as EnvNpy
from envs.advanced_malmo_numpy_env import MalmoEnvSpecial as AdvEnvNpy
from envs.malmo_env_skyline import MalmoEnvSpecial as EnvMalmo

from envs.advanced_malmo_numpy_env_all_tools import MalmoEnvSpecial as EnvNpyAllTools
from envs.advanced_malmo_numpy_env_correct_tool import MalmoEnvSpecial as EnvNpyCorrectTool
from envs.advanced_malmo_numpy_env_all_tools_equip import MalmoEnvSpecial as EnvNpyAllToolsEquip

from envs.numpy_easy import MalmoEnvSpecial as EnvEasy
from envs.numpy_easy_4task import MalmoEnvSpecial as EnvEasy4
from envs.numpy_easy_4task_mask_init import MalmoEnvSpecial as EnvEasy4_mask
from envs.proc_env_creator import MalmoEnvSpecial as EnvEasy_proc_gen

from embed_utils.vis_distance_methods import load_embed_from_torch, visualize_similarity

args = parse_args()
# Setting cuda seeds
if torch.cuda.is_available():
    torch.backends.cuda.deterministic = True
    torch.backends.cuda.benchmark = False

# if 'npy_easy' in args.env:
#     from model_easy import DQN_agent, Experience
# else:

from model import DQN_agent, Experience
# Setting random seed
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

run_tag = args.run_tag

env_graph_data = None

#env = MalmoEnvSpecial("pickaxe_stone",port=args.port, addr=args.address)
if args.env == 'npy':
    env = EnvNpy(random=True, mission=None)
    test_env = EnvNpy(random=True, mission=None)
elif args.env == 'npy_easy':
    env = EnvEasy(random=True, mission=None)
    test_env = EnvEasy(random=True, mission=None)
elif args.env == 'npy_easy_4task':
    env = EnvEasy4(random=True, mission=None)
    test_env = EnvEasy4(random=True, mission=None)
elif args.env == 'npy_easy_gen':
    env = EnvEasy_proc_gen(1,2)
    test_env = env #EnvEasy_proc_gen(random=True, mission=None)

elif args.env == 'npy_easy_4task_mask':
    env = EnvEasy4_mask(random=True, mission=None, init_window=(0, 76))
    test_env = EnvEasy4_mask(random=True, mission=None, init_window=(76, None))

elif args.env == 'npy_stone':
    env = EnvNpy(random=False, mission="pickaxe_stone")
    test_env = EnvNpy(random=False, mission="pickaxe_stone")
elif args.env == 'adv_npy_all_tools':
    env = EnvNpyAllTools(random=True, mission=None)
    test_env = EnvNpyAllTools(random=True, mission=None)
elif args.env == 'adv_npy_all_tools_equip':
    env = EnvNpyAllToolsEquip(random=True, mission=None)
    test_env = EnvNpyAllToolsEquip(random=True, mission=None)
elif args.env == 'adv_npy_correct_tool':
    env = EnvNpyCorrectTool(random=True, mission=None)
    test_env = EnvNpyCorrectTool(random=True, mission=None)
elif args.env == 'adv_npy':
    env = AdvEnvNpy(random=True, mission=None)
    test_env = AdvEnvNpy(random=True, mission=None)
elif args.env == 'malmo_server':
    assert args.address is not None
    assert args.port is not None
    env = EnvMalmo(random=True, mission=None)
    test_env = EnvMalmo(random=True, mission=None)
    #"hoe_farmland")#"pickaxe_stone",train_2=True,port=args.port, addr=args.address)
else:
    env = gym.make(args.env)
    test_env = gym.make(args.env)

env.seed(args.seed)
test_env.seed(args.seed)

# Check if GPU can be used and was asked for
if args.gpu and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

if args.mode in ['skyline_simple'] or '_hier' in args.mode:
    args.use_hier = True

if '_atten' in args.mode:
    args.atten = True

if '_multi' in args.mode:
    args.multi_edge = True

if args.mode == 'cnn':
    args.num_frames = 1

num_actions = 0
if isinstance(env.action_space, gym.spaces.Dict):
    for action_name, action in env.action_space.spaces.items():
        if isinstance(action, gym.spaces.Discrete):
            num_actions += action.n
        print("{} has {} actions".format(action_name, action.n))
else:
    num_actions = env.action_space.n

# Initialize model
agent_args = {
    "device": device,
    "state_space": (env.observation_space[0], env.observation_space[1]),
    "action_space": env.action_space,
    "num_actions": num_actions,
    "target_moving_average": args.target_moving_average,
    "gamma": args.gamma,
    "replay_buffer_size": args.replay_buffer_size,
    "epsilon_decay": args.epsilon_decay,
    "epsilon_decay_end": args.epsilon_decay_end,
    "warmup_period": args.warmup_period,
    "double_DQN": not (args.vanilla_DQN),
    "model_type": args.model_type,
    "num_frames": args.emb_size,
    "mode": args.mode,
    "hier": args.use_hier,
    "atten": args.atten,
    "emb_size": args.emb_size,
    "one_layer": args.one_layer,
    "multi_edge": args.multi_edge,
    "model_size": args.model_size,
    "final_dense_layer": args.final_dense_layer,
    "aux_dist_loss_coeff": args.aux_dist_loss_coeff,
    "contrastive_loss_coeff": args.contrastive_loss_coeff,
    "positive_margin": args.positive_margin,
    "negative_margin": args.negative_margin,
    "self_attention": args.self_attention,
    "use_layers": args.use_layers,
    "converged_init": args.converged_init,
    "dist_path": args.dist_path,
    "reverse_direction": args.reverse_direction,
    "env_graph_data":  env.generate_graph_info(args.mode in {"skyline_hier"},args.reverse_direction) if args.env == 'npy_easy_gen' else None
}
agent = DQN_agent(**agent_args)

# Initialize optimizer
optimizer = torch.optim.Adam(agent.online.parameters(), lr=args.lr)

# Load checkpoint
if args.load_checkpoint_path:
    checkpoint = torch.load(args.load_checkpoint_path)
    agent.online.load_state_dict(checkpoint['model_state_dict'])
    agent.target.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.online.train()

# Save path
if args.model_path:
    os.makedirs(args.model_path, exist_ok=True)

if args.output_path:
    os.makedirs(args.output_path, exist_ok=True)

# Logging via csv
if args.output_path:
    base_filename = os.path.join(args.output_path, run_tag)
    os.makedirs(base_filename, exist_ok=True)
    log_filename = os.path.join(base_filename, 'reward.csv')
    mission_filename = os.path.join(base_filename, 'mission_distribution.csv')
    with open(log_filename, "w") as f:
        f.write("episode,steps,reward\n")
    with open(mission_filename, "w") as f:
        f.write("episode,steps,type,reward\n")
    with open(os.path.join(base_filename, 'params.json'), 'w') as fp:
        param_dict = vars(args).copy()
        del param_dict['output_path']
        del param_dict['model_path']
        json.dump(param_dict, fp)
else:
    log_filename = None

# Logging for tensorboard
if not args.no_tensorboard:
    writer = SummaryWriter(comment=run_tag)
else:
    writer = None

# Episode loop
global_steps = 0
steps = 1
episode = 0
start = time.time()
end = time.time() + 1

if not os.path.exists(args.dist_save_folder):
    os.mkdir(args.dist_save_folder)

if args.save_dist_freq != -1:
    node_to_sim_index = {v: k for k, v in agent.online.object_to_char.items()}

if args.load_checkpoint_path and checkpoint is not None:
    global_steps = checkpoint['global_steps']
    episode = checkpoint['episode']

while global_steps < args.max_steps:
    print(f"Episode: {episode}, steps: {global_steps}, FPS: {steps/(end - start+0.001)}")
    start = time.time()
    state = env.reset()
    done = False
    agent.set_epsilon(global_steps, writer)

    cumulative_loss = 0
    steps = 1
    # Collect data from the environment
    while not done:
        global_steps += 1
        if args.save_dist_freq != -1 and int(global_steps - 1) % args.save_dist_freq == 0:
            print("SAVING")
            name_to_embeddings = load_embed_from_torch(agent.online.state_dict()["embeds.weight"],
                                                       node_to_sim_index)
            file_name = os.path.join(args.dist_save_folder,
                                     f"checkpoint_{run_tag}") + f"_{global_steps}"
            visualize_similarity(name_to_embeddings, file_name)
        action = agent.online.act(state, agent.online.epsilon)
        next_state, reward, done, info = env.step(action)
        steps += 1
        if args.reward_clip:
            clipped_reward = np.clip(reward, -args.reward_clip, args.reward_clip)
        else:
            clipped_reward = reward
        agent.replay_buffer.append(Experience(state, action, clipped_reward, next_state, int(done)))
        state = next_state

        # If not enough data, try again
        if len(agent.replay_buffer) < args.batchsize or global_steps < args.warmup_period:
            continue

        # This is list<experiences>
        minibatch = random.sample(agent.replay_buffer, args.batchsize)

        # This is experience<list<states>, list<actions>, ...>
        minibatch = Experience(*zip(*minibatch))
        optimizer.zero_grad()

        # Get loss
        if not args.no_tensorboard:
            loss = agent.loss_func(minibatch, writer, global_steps)
        else:
            loss = agent.loss_func(minibatch, None, global_steps)

        cumulative_loss += loss.item()
        loss.backward()
        if args.gradient_clip:
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(), args.gradient_clip)
        # Update parameters
        optimizer.step()
        agent.sync_networks()

        if args.model_path is not None:
            if global_steps % args.checkpoint_steps == 0:
                for filename in os.listdir(args.model_path):
                    if "checkpoint" in filename and run_tag in filename:
                        os.remove(os.path.join(args.model_path, filename))
                torch.save(
                    {
                        "global_steps": global_steps,
                        "model_state_dict": agent.online.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "episode": episode,
                    },
                    append_timestamp(os.path.join(args.model_path, f"checkpoint_{run_tag}")) +
                    f"_{global_steps}.tar")

        # Testing policy
        if global_steps % args.test_policy_steps == 0:
            cumulative_reward = 0
            mission_rewards = {k: 0 for k in env.mission_types}

            for _ in range(args.num_test_runs):
                test_reward = 0
                with torch.no_grad():
                    # Reset environment
                    test_state = test_env.reset()
                    test_action = agent.online.act(test_state, 0)
                    test_done = False
                    render = args.render and (episode % args.render_episodes == 0)

                    # Test episode loop
                    while not test_done:
                        # Take action in env
                        if render:
                            env.render()

                        test_state, t_reward, test_done, info = test_env.step(test_action)
                        test_action = agent.online.act(test_state, 0)  # passing in epsilon = 0
                        # Update reward
                        test_reward += t_reward

                    print(f"{info['mission']}: {test_reward}")
                    mission_rewards[info['mission']] += test_reward
                    cumulative_reward += test_reward
                    if render:
                        test_env.close()  # close viewer

            cumulative_reward /= args.num_test_runs
            for k in mission_rewards.keys():
                mission_rewards[k] /= args.num_test_runs
            print(f"Policy_reward for test: {cumulative_reward}")

            if not args.no_tensorboard:
                writer.add_scalar('validation/policy_reward',
                                  cumulative_reward / args.num_test_runs,
                                  global_steps)

            if log_filename:
                with open(log_filename, "a") as f:
                    f.write(f"{episode},{global_steps},{cumulative_reward}\n")

            if mission_filename:
                with open(mission_filename, "a") as f:
                    for k, v in mission_rewards.items():
                        f.write(f"{episode},{global_steps},{k},{v}\n")

    if not args.no_tensorboard:
        writer.add_scalar('training/avg_episode_loss', cumulative_loss / steps, episode)
    end = time.time()
    episode += 1
    if len(agent.replay_buffer) < args.batchsize or global_steps < args.warmup_period:
        continue

    #if episode % 500 == 0:
    #with open("embed_bl_{}".format(episode),'wb') as embed_file:
    #   np.save(embed_file,agent.online.state_dict()['gcn.obj_emb.weight'].cpu().data.numpy())

env.close()
if args.model_path:
    torch.save(agent.online, append_timestamp(os.path.join(args.model_path, run_tag)) + ".pth")
