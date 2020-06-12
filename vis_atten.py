import torch
import sys
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
            "farmland_item":15
        }
        #object_to_char = {"air":0,"bedrock":1,"stone":2,"pickaxe_item":3,"cobblestone_item":4,"log":5,"axe_item":6,"dirt":7,"farmland":8,"hoe_item":9,"water":10,"bucket_item":11,"water_bucket_item":12,"log_item":13,"dirt_item":14,"farmland_item":15}
non_node_objects = ["air", "wall"]
game_nodes = sorted([k for k in object_to_char.keys() if k not in non_node_objects])
latent_nodes = ["edge_tool","non_edge_tool", "material","product"]  

nodes = game_nodes + latent_nodes

checkpoint = torch.load(sys.argv[1],map_location=torch.device('cpu'))
params = checkpoint["model_state_dict"]
for name in params.keys():
    if "A" in name or "attention" in name:
        print(name)
        #print(params[name])
        for node,p in zip(nodes,params[name]):
             print("\n\n")
             print(node,sorted([(n,e) for n,e in zip(nodes,p)],key=lambda x: x[1],reverse=True))
        #print(game_nodes)
        #print(params[name][-1])
  #  if "A" in name or "atten" in name:
   #    print(name,param)
