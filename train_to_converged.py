import torch
import numpy as np
from torch import optim
import random

# random.seed(2)

attributes = torch.FloatTensor(np.load("attribute_vector_neg_8_fourtask.npy"))
distances = torch.FloatTensor(np.load("converged_4task_distance.npy"))
weight = torch.nn.Parameter(torch.rand((8,4))) 

optimizer = optim.SGD([weight], lr=0.02)
lossFN = torch.nn.MSELoss(reduction="mean")
BatchSize = 1

#seed 0: [7 6 5 4 3 2 1 0]
#seed 1: [5 2 7 3 6 4 1 0]

possible_coords = []
for x in range(len(distances)):
    for y in range(x):
        possible_coords.append((x,y))

# print(possible_coords)
# exit()

random.shuffle(possible_coords)


train_coords = possible_coords #[:60]
test_coords = [] #possible_coords[60:]

# print(test_coords)
# exit()

for k in range(10):

	for e in range(1000):
	    random.shuffle(train_coords)

	    
	    # print(len(train_coords))

	  

	    for d,do_train in [(train_coords,True),(train_coords,False)]:

	        mse = []
	        loss = torch.tensor(0.0)
	        reg_loss = torch.tensor(0.0)
	        optimizer.zero_grad()
	        for i,(x,y) in enumerate(d):
	            # print(i)
	            if int(i) % BatchSize == 0 and int(i) != 0:
	                mse.append(loss.detach()/BatchSize)
	                if do_train:
	                    (loss+reg_loss).backward()
	                    optimizer.step()
	                optimizer.zero_grad()
	                loss = 0.0
	                reg_loss = 0.0
	                if not do_train:
	                	pass
	                    # print("Is training:",do_train,sum(mse[-100:])/len(mse[-100:]))
	                    # print(torch.sum(torch.abs(weight),dim=1).tolist())
	                    # print(np.argsort(torch.sum(torch.abs(weight),dim=1).tolist()))




	            e1 = torch.matmul(attributes[x],weight)
	            e2 = torch.matmul(attributes[y],weight)
	            reg_loss+= 0.5 * torch.norm(weight, 1)
	            loss+= lossFN(distances[x][y],torch.norm(e1-e2))



	print(torch.sum(torch.abs(weight),dim=1).tolist())
	print(np.argsort(torch.sum(torch.abs(weight),dim=1).tolist()))
