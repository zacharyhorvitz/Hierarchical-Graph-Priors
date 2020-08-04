import json
import torch
from pprint import pprint
import numpy as np

from modules import contrastive_loss_func

filename = 'extra/1embed.json'

with open(filename, 'r') as fp:
    embeds = json.load(fp)

device = torch.device('cpu')
latent_nodes = ["edge_tool", "tool", "material", "product"]
edges = [[("pickaxe_item", "stone"), ("axe_item", "log"), ("hoe_item", "dirt"),
          ("bucket_item", "water"), ("stone", "cobblestone_item"), ("log", "log_item"),
          ("dirt", "farmland"), ("water", "water_bucket_item"), ("edge_tool", "pickaxe_item"),
          ("edge_tool", "axe_item"), ("tool", "hoe_item"), ("tool", "bucket_item"),
          ("material", "stone"), ("material", "log"), ("material", "dirt"), ("material", "water"),
          ("product", "log_item"), ("product", "cobblestone_item"), ("product", "farmland"),
          ("product", "water_bucket_item")]]

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

non_node_objects = ["air", "wall"]
game_nodes = sorted([k for k in object_to_char.keys() if k not in non_node_objects])

total_objects = len(game_nodes + latent_nodes + non_node_objects)
name_2_node = {e: i for i, e in enumerate(game_nodes + latent_nodes)}
node_2_name = {v: k for k, v in name_2_node.items()}
node_2_game = {i: object_to_char[name] for i, name in enumerate(game_nodes)}

num_nodes = len(game_nodes + latent_nodes)

adjacency = torch.FloatTensor(torch.zeros(len(edges), num_nodes, num_nodes))

for edge_type in range(len(edges)):
    for i in range(num_nodes):
        adjacency[edge_type][i][i] = 1.0
    for s, d in edges[edge_type]:
        adjacency[edge_type][name_2_node[d]][name_2_node[s]] = 1.0  #corrected transpose!!!!

edges = edges[0]
adjacency = adjacency[0]

print(adjacency.to(dtype=torch.int8))
print(latent_nodes)

for k in name_2_node.keys():
    if k not in embeds:
        embeds[k] = [0.0] * 8

sorted_embeds = sorted(embeds.items(),
                       key=lambda k: name_2_node[k[0]]
                       if k[0] in name_2_node.keys() else float('inf'))

sorted_embeds = torch.stack([torch.tensor(t[1]) for t in sorted_embeds])
# print(sorted_embeds.shape)
# pprint(torch.cdist(sorted_embeds, sorted_embeds, p=2), width=220)
prange = list(np.linspace(2, 9, num=int((9-2) * 1/0.5)))
nrange = list(np.linspace(0, 0.5, num=int((0.5-0) * 1/0.05)))
loss_tensor = torch.zeros((len(prange), len(nrange)))
for i, p in enumerate(prange):
    for j, n in enumerate(nrange):
        loss_tensor[i][j] = contrastive_loss_func(device,
                                  sorted_embeds,
                                  adjacency[0],
                                  latent_nodes,
                                  node_2_name,
                                  p,
                                  n)
print('\n')
print('      ' + ' '.join(['{:0.2f}'.format(x).zfill(6) for x in nrange]))
for i in range(len(prange)):
    print("{:0.2f}: {}".format(prange[i], ' '.join(['{:0.2f}'.format(x).zfill(6) for x in list(loss_tensor[i])])))
