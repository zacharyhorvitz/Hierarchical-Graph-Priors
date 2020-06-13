import numpy as np
import random
import sys
import gym
from gym.spaces import Discrete


class MalmoEnvSpecial(gym.Env):
    def __init__(self, random, mission=False):
        self.random = random
        if random == False:
            assert mission is not None
            self.current_mission = mission
        self.observation_space = gym.spaces.Box(0, 32, shape=(9, 9))
        self.actions = [
            "movenorth", "movesouth", "movewest", "moveeast",
            "chop 0", "chop 1", # axe 
            "mine 0", "mine 1", # pickaxe
            "farm 0", "farm 1", # hoe
            "fill 0", "fill 1", # bucket 
            # "attack 0", "attack 1",
            # "use 0", "use 1",
        ]
        self.action_space = Discrete(len(self.actions))
        self.mission_types = [
            "pickaxe_stone", "axe_log", "hoe_farmland", "bucket_water"
        ]
        self.step_cost = -0.1
        self.goal_reward = 100.0
        self.max_steps = 100.0
        self.object_2_index = {
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
        self.index_2_object = {v: k for k, v in self.object_2_index.items()}
        self.collectable = {
            v for k, v in self.object_2_index.items() if "item" in k
        }
        self.passable = set(
            list(self.collectable) + [0] + [self.object_2_index["water"]])
        self.reset()

    def init_map(self, mission):
        arena = np.ones((13, 13))
        arena[4:9, 4:9] = 0
        arena_shift = 6
        item_x = 8
        item_y = 8
        blocK_y = random.randint(1, 3) - 2 + arena_shift
        block_x = random.randint(0, 4) - 2 + arena_shift

        if mission == "pickaxe_stone":
            item = self.object_2_index["pickaxe_item"]
            block = self.object_2_index["stone"]
        elif mission == "axe_log":
            item = self.object_2_index["axe_item"]
            block = self.object_2_index["log"]
        elif mission == "bucket_water":
            item = self.object_2_index["bucket_item"]
            block = self.object_2_index["water"]
        elif mission == "hoe_farmland":
            item = self.object_2_index["hoe_item"]
            block = self.object_2_index["dirt"]

        arena[item_x][item_y] = item
        arena[blocK_y][block_x] = block
        return arena

    def arena_obs(self, add_selected_item=True, add_goal=True):
        cur_arena = self.arena.copy()
        if add_selected_item:
            cur_arena[self.player_y][self.player_x] = self.inventory[
                self.selected_inv_item]
        obs = cur_arena[self.player_y - 4:self.player_y + 5,
                        self.player_x - 4:self.player_x + 5]
        if add_goal:
            obs = np.concatenate((np.ones(
                (9, 1)) * self.object_2_index[self.goal], obs),
                                 axis=1)

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
        self.mining = False
        self.chopping = False
        self.filling = False
        self.farming = False
        self.steps = 0
        self.inventory = np.zeros((10))
        self.selected_inv_item = 0  #Cannot change!
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

            elif self.actions[action] == "chop 0":
                self.chopping = False
            elif self.actions[action] == "mine 0":
                self.mining = False
            elif self.actions[action] == "farm 0":
                self.farming = False
            elif self.actions[action] == "fill 0":
                self.filling = False

            elif self.actions[action] == "chop 1":
                self.chopping = True 
                self.mining = False
                self.farming = False
                self.filling = False
            elif self.actions[action] == "mine 1":
                self.chopping = False
                self.mining = True
                self.farming = False
                self.filling = False
            elif self.actions[action] == "farm 1":
                self.chopping = False
                self.mining = False
                self.farming = True
                self.filling = False
            elif self.actions[action] == "fill 1":
                self.chopping = False
                self.mining = False
                self.farming = False
                self.filling = True

        if self.arena[self.player_y][self.player_x] in self.collectable:
            if self.insert_inv(self.arena[self.player_y][self.player_x]):
                self.arena[self.player_y][self.player_x] = 0

        if self.mining:
            if self.arena[self.player_y + 1][self.player_x] == self.object_2_index["stone"]:
                if self.inventory[0] == self.object_2_index["pickaxe_item"]:
                    self.arena[self.player_y + 1][self.player_x] = self.object_2_index["cobblestone_item"]
        if self.chopping:
            if self.arena[self.player_y + 1][self.player_x] == self.object_2_index["log"]:
                if self.inventory[0] == self.object_2_index["axe_item"]:
                    self.arena[self.player_y + 1][self.player_x] = self.object_2_index["log_item"]
        if self.mining or self.chopping:
            if self.arena[self.player_y + 1][self.player_x] == self.object_2_index["dirt"]:
                self.arena[self.player_y + 1][self.player_x] = self.object_2_index["dirt_item"]
        if self.mining or self.chopping:
            if self.arena[self.player_y + 1][self.player_x] == self.object_2_index["farmland"]:
                self.arena[self.player_y + 1][self.player_x] = self.object_2_index["farmland_item"]

        if self.farming:
            if self.arena[self.player_y +
                          1][self.player_x] == self.object_2_index["dirt"]:
                if self.inventory[0] == self.object_2_index["hoe_item"]:
                    self.arena[self.player_y + 1][
                        self.player_x] = self.object_2_index["farmland"]
        if self.filling:
            if self.arena[self.player_y +
                          1][self.player_x] == self.object_2_index["water"]:
                if self.inventory[0] == self.object_2_index["bucket_item"]:
                    #	print("using bucket in inventory")
                    self.arena[self.player_y + 1][self.player_x] = 0
                    self.inventory[0] = self.object_2_index["water_bucket_item"]

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

if __name__ == "__main__":

    env = MalmoEnvSpecial()
    obs = env.reset()
    print("reset")
    for step in range(100):
        print("\n", step)
        command = int(input())
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
