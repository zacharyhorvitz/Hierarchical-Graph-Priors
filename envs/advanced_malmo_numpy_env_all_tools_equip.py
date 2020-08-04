import numpy as np
import random
import sys
import gym
from gym.spaces import Discrete


class MalmoEnvSpecial(gym.Env):

    def init_map(self, mission):
        arena = np.ones((13, 13))
        arena[4:9, 4:9] = 0
        arena_shift = 4
        #item_x = 8
        #item_y = 8

 #       blocK_y = random.randint(1, 4) + arena_shift
  #      block_x = random.randint(0, 4) + arena_shift
        spawn_entities = [] #["stone","log","water","dirt"]+["pickaxe_item","axe_item","bucket_item","hoe_item"]
 
        all_missions = [mission] + [random.choice(self.mission_types)]
        for m in all_missions:
            if m == "pickaxe_stone":
                spawn_entities += ["stone"]
            elif m == "axe_log":
                spawn_entities += ["log"]
            elif m == "hoe_farmland":
                spawn_entities += ["dirt"]
            elif m == "bucket_water":
                spawn_entities += ["water"]
            else:
                print("Bad mission",m)
                exit()


        locations = np.random.choice(np.arange(len(self.poss_spawn_loc)), size=len(spawn_entities), replace=False)
        for ent,l in zip(spawn_entities,locations):
            ent_id = self.object_2_index[ent]
            coords = self.poss_spawn_loc[l]
            arena[coords[0]][coords[1]] = ent_id

        return arena

    def arena_obs(self, add_selected_item=True, add_goal=True,add_inv=True,render=False):
        cur_arena = self.arena.copy()
        if render:
           god_view = self.arena.copy()
           god_view[self.player_y][self.player_x] = -1
           print(god_view)
           print("EQUIPPED:",self.equipped_item) 
           print(self.inventory)
        if add_selected_item:
            cur_arena[self.player_y][self.player_x] = self.equipped_item #self.inventory[self.selected_inv_item]
        obs = cur_arena[self.player_y - 4:self.player_y + 5,
                        self.player_x - 4:self.player_x + 5]
        if add_inv:
#             if self.selected_inv_item == 0:
            inv_obs = self.inventory
#             else:
#                inv_obs = self.inventory # np.concatenate((self.inventory[self.selected_inv_item:],self.inventory[:self.selected_inv_item]))
            obs = np.concatenate((inv_obs.reshape(-1,1), obs),axis=1)
            
        if add_goal:
            obs = np.concatenate((np.ones(
                (9, 1)) * self.object_2_index[self.goal], obs),
                                 axis=1)
         

    #	print(obs)

       # for orig, new in (13, 5), (14, 7), (15, 8):
       #     obs = np.where(obs == orig, new, obs)
        return obs.reshape(1, 9, -1)

    def reset(self):
        if self.random:
            self.current_mission = random.choice(self.mission_types)
        if self.current_mission == "pickaxe_stone":
            self.goal = "cobblestone_item"
        elif self.current_mission == "axe_log":
            self.goal = "log_item"
        elif self.current_mission == "hoe_farmland":
            self.goal = "farmland"
        elif self.current_mission == "bucket_water":
            self.goal = "water_bucket_item"

        self.player_x = 6
        self.player_y = 4
        self.arena = self.init_map(self.current_mission)
        self.attacking = False
        self.using = False
        self.steps = 0
        self.inventory = np.array([3,6,9,11,0,0,0,0,0])
#         self.selected_inv_item = 0 
        self.equipped_item = 0
        return self.arena_obs()

    def check_reached_goal(self):
        if self.current_mission == "pickaxe_stone":
            if self.object_2_index["cobblestone_item"] in self.inventory:
                return True
        elif self.current_mission == "axe_log":
            if self.object_2_index["log_item"] in self.inventory:
                return True
        elif self.current_mission == "hoe_farmland":
            if self.arena[self.player_y +
                          1][self.player_x] == self.object_2_index["farmland"]:
                return True
        elif self.current_mission == "bucket_water":
            if self.object_2_index["water_bucket_item"] in self.inventory:
                return True

        return False

    def insert_inv(self, item):
        for i, value in enumerate(self.inventory):
            if value == 0:
                self.inventory[i] = item
                return True
        return False

#     def shift_inv(self):
#         if np.sum(self.inventory) == 0: 
#             pass
#         else:
#            shift = 0
#            while self.inventory[self.selected_inv_item] == 0 or shift == 0:
#                self.selected_inv_item+= 1
#                self.selected_inv_item = self.selected_inv_item % len(self.inventory)
#                shift+=1

    def equip(self,value):
        if self.object_2_index[value] in self.inventory:
            self.equipped_item = self.object_2_index[value]
        else:
            self.equipped_item = 0

    def step(self, action):
        if action >= len(self.actions) or action < 0:
            print("Invalid Action: {}".format(action))
        else:
            if self.actions[action] == "movenorth":
                if self.arena[self.player_y +
                              1][self.player_x] in self.passable:
                    self.player_y += 1
            elif self.actions[action] == "movesouth":
                if self.arena[self.player_y -
                              1][self.player_x] in self.passable:
                    self.player_y -= 1
            elif self.actions[action] == "movewest":
                if self.arena[self.player_y][self.player_x +
                                             1] in self.passable:
                    self.player_x += 1
            elif self.actions[action] == "moveeast":
                if self.arena[self.player_y][self.player_x -
                                             1] in self.passable:
                    self.player_x -= 1
            elif self.actions[action] == "attack 0":
                self.attacking = False
            elif self.actions[action] == "use 0":
                self.using = False
            elif self.actions[action] == "attack 1":
                self.attacking = True
                self.using = False
            elif self.actions[action] == "use 1":
                self.using = True
                self.attacking = False
            elif self.actions[action] == "equip pickaxe":
                self.equip("pickaxe_item")
            elif self.actions[action] == "equip axe":
                self.equip("axe_item")            
            elif self.actions[action] == "equip hoe":
                self.equip("hoe_item")
            elif self.actions[action] == "equip bucket":
                self.equip("bucket_item")
        if self.arena[self.player_y][self.player_x] in self.collectable:
            if self.insert_inv(self.arena[self.player_y][self.player_x]):
                self.arena[self.player_y][self.player_x] = 0
        if self.attacking:
            if self.arena[self.player_y +
                          1][self.player_x] == self.object_2_index["stone"]:
                if self.equipped_item == self.object_2_index["pickaxe_item"]:
                    self.arena[self.player_y + 1][
                        self.player_x] = self.object_2_index["cobblestone_item"]
            if self.arena[self.player_y +
                          1][self.player_x] == self.object_2_index["log"]:
                if self.equipped_item == self.object_2_index["axe_item"]:
                    self.arena[self.player_y + 1][
                        self.player_x] = self.object_2_index["log_item"]
            if self.arena[self.player_y +
                          1][self.player_x] == self.object_2_index["dirt"]:
                self.arena[self.player_y +
                           1][self.player_x] = self.object_2_index["dirt_item"]
            if self.arena[self.player_y +
                          1][self.player_x] == self.object_2_index["farmland"]:
                self.arena[self.player_y + 1][
                    self.player_x] = self.object_2_index["farmland_item"]
        if self.using:
            if self.arena[self.player_y +
                          1][self.player_x] == self.object_2_index["dirt"]:
                if self.equipped_item == self.object_2_index["hoe_item"]: #self.inventory[self.selected_inv_item] == self.object_2_index["hoe_item"]:
                    self.arena[self.player_y + 1][
                        self.player_x] = self.object_2_index["farmland"]
            if self.arena[self.player_y +
                          1][self.player_x] == self.object_2_index["water"]:
              #  print("USING IN FRONT OF WATER")
                if self.equipped_item == self.object_2_index["bucket_item"]: #self.inventory[self.selected_inv_item] == self.object_2_index["bucket_item"]:
                    #	print("using bucket in inventory")
               #     print("BUCKET USED")
                    self.arena[self.player_y + 1][self.player_x] = 0
                    locations = np.where(self.inventory == self.equipped_item)
                    self.equipped_item = self.object_2_index["water_bucket_item"]
                    self.inventory[locations[0]] = self.equipped_item

        self.attacking = False
        self.using = False
        goal = self.check_reached_goal()
        reward = self.goal_reward if goal else self.step_cost
        #if goal: print("SUCCEEDED")

        if self.steps >= self.max_steps:
            terminated = True
        else:
            terminated = goal
        self.steps += 1

        #	if self.object_2_index["bucket_item"] in self.inventory: print("Bucket item in iventory")
        #TODO:
        # - item health
        # - other items can destroy log, but it taskes longer
        # - other items can destroy stone, but no cobblesotne
        # - inventory switching
        obs = self.arena_obs(add_selected_item=True, add_goal=True)
        #print(obs)
        #print(self.goal)
        return obs, reward, terminated, {"mission": self.current_mission}

    def __init__(self, random, mission=False):
        self.random = random
        if random == False:
            assert mission is not None
            self.current_mission = mission

        #self.actions = [
        #    "movenorth", "movesouth", "movewest", "moveeast", "attack 0",
        #    "attack 1", "use 0", "use 1","equip pickaxe","equip axe","equip hoe","equip bucket"
        #]
        self.actions = [
            "movenorth", "movesouth", "movewest", "moveeast","attack 1", "use 1","equip pickaxe","equip axe","equip hoe","equip bucket"
        ]
        self.action_space = Discrete(len(self.actions))
        self.observation_space = (9,9)

        self.mission_types = [
            "pickaxe_stone", "axe_log", "hoe_farmland", "bucket_water"
        ]
        self.step_cost = -0.1
        self.goal_reward = 1000.0
        self.max_steps = 1000.0
        self.object_2_index = {
            "air": 0,
            "bedrock": 1,
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



        self.index_2_object = {v: k for k, v in self.object_2_index.items()}
        self.collectable = {
            v for k, v in self.object_2_index.items() if "item" in k
        }
        self.passable = set(
            list(self.collectable) + [0] + [self.object_2_index["water"]])
        self.poss_spawn_loc = []
        for y in range(5, 8+1):
             for x in range(4, 8+1):
                 self.poss_spawn_loc.append((y,x))

        self.reset()
        


if __name__ == "__main__":

    env = MalmoEnvSpecial(True) #False,"bucket_water")
    obs = env.reset()
    print("reset")
    for step in range(100):
        print("\n", step)
        try:
           command = int(input())
        except ValueError:
           command = 8
        obs, reward, done, info = env.step(command)
        print(obs)
        print(reward)

# 1111111111111
# 1111111111111
# 1111111111111
# 1111111111111
# 1111000001111
# 1111000001111
# 1111000001111
# 1111000001111
# 1111000001111
# 1111111111111
# 1111111111111
# 1111111111111
# 1111111111111
