import numpy as np
names={1:"player",0:"air",2:"stone",3:"pickaxe_item",4:"cobblestone_item",5:"log",6:"axe_item",13:"log_item"}
attribute_vector = np.zeros((18,4))
index_to_attvec = {
0: [1,1,0,0],
1: [0,0,0,0],
2: [0,1,0,1],
3: [1,0,0,1],
4: [1,0,1,1],
5: [1,0,1,0],
6: [1,0,0,0],
13: [1,0,1,0]}
for index in names:
    attribute_vector[index, :] = index_to_attvec[index] 
np.save('attribute_vector.npy', np.array(attribute_vector))
