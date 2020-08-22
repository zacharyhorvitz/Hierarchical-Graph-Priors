import argparse

# Parse input arguments
# Use --help to see a pretty description of the arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env',
                    help='The malmo environment to train on',
                    type=str,
                    default='npy',
                    required=True)
parser.add_argument('--procgen-tools', help='Tools for procgen', type=int, default=1)
parser.add_argument('--procgen-blocks', help='Blocks for procgen', type=int, default=2)
parser.add_argument('--run-tag', help='Tag to identify experiment', type=str, required=True)
parser.add_argument('--address', help='ip address', type=str, required=False)
parser.add_argument('--port', help='Port', type=int, required=False)
parser.add_argument('--model-type',
                    help="Type of architecture",
                    type=str,
                    default='cnn',
                    choices=["cnn", "dueling"],
                    required=False)
parser.add_argument('--model-size',
                    help="Size of architecture",
                    type=str,
                    default='small',
                    choices=["small", "large", "extra_large"],
                    required=False)
parser.add_argument('--model-path',
                    help='The path to the save the pytorch model',
                    type=str,
                    required=False)
parser.add_argument('--gamma', help='Gamma parameter', type=float, default=0.99, required=False)
parser.add_argument('--output-path',
                    help='The output directory to store training stats',
                    type=str,
                    default="./logs",
                    required=False)
parser.add_argument('--load-checkpoint-path',
                    help='Path to checkpoint',
                    type=str,
                    required=False)
parser.add_argument('--no-tensorboard',
                    help='No tensorboard logging',
                    action='store_true',
                    required=False)
parser.add_argument('--ari',
                    help='Use ARI ram state features',
                    action='store_true',
                    required=False)
parser.add_argument('--no-atari',
                    help='Use atari preprocessing',
                    action='store_false',
                    required=False)
parser.add_argument('--gpu', help='Use the gpu or not', action='store_true', required=False)
parser.add_argument('--render',
                    help='Render visual or not',
                    action='store_true',
                    required=False)
parser.add_argument('--render-episodes',
                    help='Render every these many episodes',
                    type=int,
                    default=5,
                    required=False)
parser.add_argument('--num-frames',
                    help='Number of frames to stack (CNN only)',
                    type=int,
                    default=4,
                    required=False)
parser.add_argument('--max-steps',
                    help='Number of steps to run for',
                    type=int,
                    default=5000000,
                    required=False)
parser.add_argument('--checkpoint-steps',
                    help='Checkpoint every so often',
                    type=int,
                    default=50000,
                    required=False)
parser.add_argument('--test-policy-steps',
                    help='Policy is tested every these many steps',
                    type=int,
                    default=10000,
                    required=False)
parser.add_argument('--num-test-runs',
                    help='Number of times to test',
                    type=int,
                    default=50,
                    required=False)
parser.add_argument('--warmup-period',
                    help='Number of steps to act randomly and not train',
                    type=int,
                    default=250000,
                    required=False)
parser.add_argument('--batchsize',
                    help='Number of experiences sampled from replay buffer',
                    type=int,
                    default=32,
                    required=False)
parser.add_argument('--gradient-clip',
                    help='How much to clip the gradients by',
                    type=float,
                    default=2.5,
                    required=False)
parser.add_argument('--reward-clip',
                    help='How much to clip reward, clipped in [-rc, rc], 0 \
                    results in unclipped',
                    type=float,
                    default=0,
                    required=False)
parser.add_argument('--epsilon-decay',
                    help='Parameter for epsilon decay',
                    type=int,
                    default=1000000,
                    required=False)
parser.add_argument('--epsilon-decay-end',
                    help='Parameter for epsilon decay end',
                    type=int,
                    default=0.05,
                    required=False)
parser.add_argument('--replay-buffer-size',
                    help='Max size of replay buffer',
                    type=int,
                    default=1000000,
                    required=False)
parser.add_argument('--lr',
                    help='Learning rate for the optimizer',
                    type=float,
                    default=2.5e-4,
                    required=False)
parser.add_argument('--target-moving-average',
                    help='EMA parameter for target network',
                    type=float,
                    default=5e-3,
                    required=False)
parser.add_argument('--vanilla-DQN',
                    help='Use the vanilla dqn update instead of double DQN',
                    action='store_true',
                    required=False)
parser.add_argument('--seed',
                    help='The random seed for this run',
                    type=int,
                    default=10,
                    required=False)
parser.add_argument('--use-hier', help='Use latent nodes', action='store_true', required=False)
parser.add_argument('--mode', help='select mode', required=True)
parser.add_argument('--atten', help='Use block attention', action='store_true', required=False)
parser.add_argument('--one_layer',
                    help='just compute attention over neighbors',
                    action='store_true',
                    required=False)
parser.add_argument('--emb_size',
                    help='size of node embeddings',
                    type=int,
                    default=16,
                    required=False)
parser.add_argument('--final-dense-layer',
                    help='size of final dense layers',
                    type=int,
                    default=300,
                    required=False)
parser.add_argument('--multi_edge',
                    help='specify single edge or multi edge',
                    action='store_true',
                    required=False)
parser.add_argument('--aux_dist_loss_coeff',
                    help='specify whether to add distance loss to dqn loss',
                    type=float,
                    default=0,
                    required=False)
parser.add_argument('--contrastive-loss-coeff',
                    help='Contrastive loss coefficient',
                    type=float,
                    default=0,
                    required=False)
parser.add_argument('--negative-margin',
                    help='Negative margin for contrastive_loss',
                    type=float,
                    default=6.5,
                    required=False)
parser.add_argument('--positive-margin',
                    help='Positive margin for contrastive_loss',
                    type=float,
                    default=2.5,
                    required=False)
parser.add_argument('--self_attention',
                    help='specify whether self attention applied to node embeddings',
                    action='store_true',
                    required=False)
parser.add_argument('--use_layers',
                    help='number of GCN layers',
                    type=int,
                    default=3,
                    required=False)
parser.add_argument('--reverse_direction',
                    help='reverse directionality of edges in adj matrix',
                    action='store_true',
                    required=False)
parser.add_argument('--save_dist_freq',
                    help='save frequency of distances (must have json configured)',
                    type=int,
                    default=-1,
                    required=False)
parser.add_argument('--dist_save_folder',
                    help='path to save images of distances',
                    type=str,
                    default="dist_data",
                    required=False)
parser.add_argument('--converged_init',
                    help='path to saved embeddings',
                    type=str,
                    default=None,
                    required=False)

parser.add_argument('--dist_path',
                    help='path to folder with distance matrix and keys',
                    type=str,
                    default=None,
                    required=False)

parser.add_argument('--disconnect_graph',
                    help='replace adjacency matrix with disconnected graph',
                    action="store_true",
                    required=False)

parser.add_argument('--gcn_activation',
                    help='gcn activation relu/tanh',
                    type=str,
                    default="relu",
                    required=False)

parser.add_argument('--dw_init',
                    help='init using dw from env',
                    action="store_true",
                    required=False)
