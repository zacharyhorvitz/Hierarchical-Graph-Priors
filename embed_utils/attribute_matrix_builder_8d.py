import numpy as np
#names={1:"player",0:"air",2:"stone",3:"pickaxe_item",4:"cobblestone_item",5:"log",6:"axe_item",13:"log_item"}
object_to_char = {
"air": 0,
"player": 1,
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
attribute_vector = np.zeros((18,8))
index_to_attvec = {
"air": [1,-1,-1,-1,-1,-1,-1,-1],
"log": [-1,1,-1,-1,-1,1,-1,-1],
"stone": [-1,1,-1,-1,1,-1,-1,-1],
"log_item": [1,-1,1,-1,-1,1,-1,-1],
"cobblestone_item": [1,-1,1,-1,1,-1,-1,-1],
"dirt": [-1,1,-1,-1,-1,-1,1,-1],
"water": [1,1,-1,-1,-1,-1,-1,1],
"farmland": [-1,-1,1,-1,-1,-1,1,-1],
"water_bucket_item": [1,-1,1,-1,-1,-1,-1,1],
"hoe_item": [1,-1,-1,1,-1,-1,1,-1],
"bucket_item": [1,-1,-1,1,-1,-1,-1,1],
"axe_item": [1,-1,-1,1,-1,1,-1,-1],
"pickaxe_item": [1,-1,-1,1,1,-1,-1,-1],
"player": [-1,-1,-1,-1,-1,-1,-1,-1]
}
for index in index_to_attvec:
    attribute_vector[object_to_char[index], :] = index_to_attvec[index] 
np.save('attribute_vector_neg_8_fourtask.npy', np.array(attribute_vector))
