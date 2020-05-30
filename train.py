import random
import time
import os
import json

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import parse_args, append_timestamp
from model import DQN_agent, Experience
#https://github.com/pytorch/pytorch/issues/31554

from envs.malmo_numpy_env import MalmoEnvSpecial as EnvNpy
from envs.malmo_env_skyline import MalmoEnvSpecial as EnvMalmo

args = parse_args()

# Setting cuda seeds
if torch.cuda.is_available():
    torch.backends.cuda.deterministic = True
    torch.backends.cuda.benchmark = False

# Setting random seed
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

run_tag = args.run_tag

#env = MalmoEnvSpecial("pickaxe_stone",port=args.port, addr=args.address)
if args.env == 'npy':
    env = EnvNpy(random=True, mission=None)
elif args.env == 'malmo_server':
    assert args.address is not None
    assert args.port is not None
    env = EnvMalmo(random=True, mission=None)
    #"hoe_farmland")#"pickaxe_stone",train_2=True,port=args.port, addr=args.address)

# Check if GPU can be used and was asked for
if args.gpu and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Initialize model
agent_args = {
    "device": device,
    "state_space": env.observation_space,
    "action_space": env.action_space,
    "num_actions": env.action_space.n,
    "target_moving_average": args.target_moving_average,
    "gamma": args.gamma,
    "replay_buffer_size": args.replay_buffer_size,
    "epsilon_decay": args.epsilon_decay,
    "epsilon_decay_end": args.epsilon_decay_end,
    "warmup_period": args.warmup_period,
    "double_DQN": not (args.vanilla_DQN),
    "model_type": args.model_type,
    "num_frames": args.num_frames,
    "mode": args.mode,  #skyline,ling_prior,embed_bl,cnn
    "hier": args.use_hier
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

if args.load_checkpoint_path and checkpoint is not None:
    global_steps = checkpoint['global_steps']
    episode = checkpoint['episode']

while global_steps < args.max_steps:
    print(f"Episode: {episode}, steps: {global_steps}, FPS: {steps/(end - start)}")
    start = time.time()
    state = env.reset()
    done = False
    agent.set_epsilon(global_steps, writer)

    cumulative_loss = 0
    steps = 1
    # Collect data from the environment
    while not done:
        global_steps += 1
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

    if not args.no_tensorboard:
        writer.add_scalar('training/avg_episode_loss', cumulative_loss / steps, episode)
    end = time.time()
    episode += 1
    if len(agent.replay_buffer) < args.batchsize or global_steps < args.warmup_period:
        continue

    # Testing policy
    num_test = args.num_test_runs
    if episode % args.test_policy_episodes == 0:
        cumulative_reward = 0
        mission_rewards = {k: 0 for k in env.mission_types}

        for _ in range(num_test):
            test_reward = 0
            with torch.no_grad():
                # Reset environment
                state = env.reset()
                action = agent.online.act(state, 0)
                done = False
                render = args.render and (episode % args.render_episodes == 0)

                # Test episode loop
                while not done:
                    # Take action in env
                    if render:
                        env.render()

                    state, reward, done, info = env.step(action)
                    action = agent.online.act(state, 0)  # passing in epsilon = 0
                    # Update reward
                    test_reward += reward

                print(f"{info['mission']}: {test_reward}")
                mission_rewards[info['mission']] += test_reward
                cumulative_reward += test_reward
                env.close()  # close viewer

        cumulative_reward /= num_test
        for k in mission_rewards.keys():
            mission_rewards[k] /= num_test
        print(f"Policy_reward for test: {cumulative_reward}")

        if not args.no_tensorboard:
            writer.add_scalar('validation/policy_reward', cumulative_reward / num_test, episode)

        if log_filename:
            with open(log_filename, "a") as f:
                f.write(f"{episode},{global_steps},{cumulative_reward}\n")

        if mission_filename:
            with open(mission_filename, "a") as f:
                for k,v in mission_rewards.items():
                    f.write(f"{episode},{global_steps},{k},{v}\n")

env.close()
if args.model_path:
    torch.save(agent.online,
               append_timestamp(os.path.join(args.model_path, run_tag)) + ".pth")
