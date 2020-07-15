import torch
import sys
#from sklearn.neighbors import DistanceMetric
#dist = DistanceMetric.get_metric('euclidean')
import numpy as np

def dist(x,y):
   x = x.data.numpy()
   y = y.data.numpy()
#   print(x-y)
#   exit()

   #return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
   return np.linalg.norm(x-y)
    



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

include = {"pickaxe_item","water_bucket_item","bucket_item","axe_item","farmland_item","hoe_item","log","cobblestone_item"}


char_to_obj = {v:k for k,v in object_to_char.items()}

        #object_to_char = {"air":0,"bedrock":1,"stone":2,"pickaxe_item":3,"cobblestone_item":4,"log":5,"axe_item":6,"dirt":7,"farmland":8,"hoe_item":9,"water":10,"bucket_item":11,"water_bucket_item":12,"log_item":13,"dirt_item":14,"farmland_item":15}
#non_node_objects = ["air", "wall"]
#game_nodes = sorted([k for k in object_to_char.keys() if k not in non_node_objects])
#latent_nodes = ["edge_tool","non_edge_tool", "material","product"]  

#nodes = game_nodes + latent_nodes

checkpoint = torch.load(sys.argv[1],map_location=torch.device('cpu'))
params = checkpoint["model_state_dict"]
for name in params.keys():
   # if "obj" in name:
    if "embed" in name:
        print(name)
       # continue
        print(params[name].shape)
        for i,p in enumerate(params[name]):
             if not i in char_to_obj: continue
             print("\n")
             print(char_to_obj[i])
             print(p)
             sim = sorted([(char_to_obj[j],dist(p,v)) for j,v in enumerate(params[name]) if i != j],key=lambda x: x[1])
             for n,v in sim:
                print(n,v)
        #print(game_nodes)
        #print(params[name][-1])
  #  if "A" in name or "atten" in name:
   #    print(name,param)
