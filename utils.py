from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import gym
from atariari.benchmark.wrapper import AtariARIWrapper

from gym_wrappers import SortedARIState

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



def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        torch.nn.init.uniform_(m.bias, -1, 1)


def sync_networks(target, online, alpha):
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.data.copy_(alpha * online_param.data + (1 - alpha) * target_param.data)


# Adapted from pytorch tutorials:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
def conv2d_size_out(size, kernel_size, stride):
    return ((size[0] - (kernel_size[0] - 1) - 1) // stride + 1,
            (size[1] - (kernel_size[1] - 1) - 1) // stride + 1)


def deque_to_tensor(last_num_frames):
    """ Convert deque of n frames to tensor """
    return torch.cat(list(last_num_frames), dim=0)


# Thanks to RoshanRane - Pytorch forums
# (https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10)
# Dec 2018
# Example: Gradient flow in network
# writer.add_figure('training/gradient_flow',
#                   plot_grad_flow(agent.online.named_parameters(),
#                                  episode),
#                   global_step=episode)
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([
        Line2D([0], [0], color="c", lw=4),
        Line2D([0], [0], color="b", lw=4),
        Line2D([0], [0], color="k", lw=4)
    ], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return plt.gcf()


def append_timestamp(string, fmt_string=None):
    now = datetime.now()
    if fmt_string:
        return string + "_" + now.strftime(fmt_string)
    else:
        return string + "_" + str(now).replace(" ", "_")

def initialize_env(args):
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
        env = EnvEasy_proc_gen(args.procgen_tools, args.procgen_blocks)
        test_env = env  #EnvEasy_proc_gen(random=True, mission=None)

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
    elif args.ari:
        env = SortedARIState(AtariARIWrapper(gym.make(args.env)))
        test_env = SortedARIState(AtariARIWrapper(gym.make(args.env)))
    else:
        env = gym.make(args.env)
        test_env = gym.make(args.env)

    env.seed(args.seed)
    test_env.seed(args.seed)
    __import__('pdb').set_trace()
    return env, test_env
