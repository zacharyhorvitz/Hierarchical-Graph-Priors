import numpy as np
import random
import sys
import gym
from gym.spaces import Discrete


class MalmoEnvSpecial(gym.Env):

    def generate_game_properties(self,num_tools,num_blocks):
        self.tools = np.arange(2,num_tools+2)

        block_start = np.max(self.tools)+1
        block_end = block_start + num_blocks

        self.blocks = np.arange(block_start,block_end)
        self.drops = np.arange(block_end,block_end+num_blocks)

        block_properties_dict = {}
        tool_dict = {t:[] for t in self.tools}

        np.random.shuffle(self.blocks)
        np.random.shuffle(self.drops)

        for b,d in zip(self.blocks,self.drops):
            tool_choice = random.choice(self.tools)
            block_properties_dict[b] = {"drop":d,"tool":tool_choice}
            tool_dict[tool_choice].append(b)

        self.tool_dict = tool_dict
        self.block_properties_dict = block_properties_dict


    def init_map(self, block):
        arena = np.ones((2+2, 4+2))
        arena[1:-1,1:-1] = 0
      
        spawn_entities = []
 
        all_missions = [block]
        for m in all_missions:
            spawn_entities+=[self.block_properties_dict[block]["tool"],block]

        spawn_entities += [random.choice(self.tools),random.choice(self.blocks)]

        locations = np.random.choice(np.arange(len(self.poss_spawn_loc)), size=len(spawn_entities), replace=False)
        for ent,l in zip(spawn_entities,locations):
            ent_id = ent
            coords = self.poss_spawn_loc[l]
            arena[coords[0]][coords[1]] = ent_id

        return arena

    def arena_obs(self):
        cur_arena = self.arena.copy()

    
        cur_arena[self.player_y][self.player_x] = 1 #self.equipped_item

        obs = cur_arena[1:-1,1:-1]
         
        obs = np.concatenate((np.ones(
            (2, 1)) * self.equipped_item, obs),
                             axis=1)
    
        obs = np.concatenate((np.ones(
            (2, 1)) * self.goal, obs),
                             axis=1)

         
        return obs.reshape(1, 2, -1)

    def reset(self):
        
        self.current_mission = random.choice(self.mission_types)
        self.goal = self.block_properties_dict[ self.current_mission]["drop"]
    
        self.player_x = 1
        self.player_y = 1
        self.arena = self.init_map(self.current_mission)
        self.attacking = False
        self.using = False
        self.steps = 0
        self.inventory = np.zeros(8) #.array([3,6,9,11,0,0,0,0,0])
#         self.selected_inv_item = 0 
        self.equipped_item = 0
        return self.arena_obs()

    def check_reached_goal(self):
        if self.block_properties_dict[self.current_mission]["drop"] in self.inventory:
            return True
            
        return False

    def insert_inv(self, item):
        for i, value in enumerate(self.inventory):
            if value == 0:
                self.inventory[i] = item
                return True
        return False

    # def equip(self,value):
    #     if self.object_2_index[value] in self.inventory:
    #         self.equipped_item = self.object_2_index[value]
    #     else:
    #         self.equipped_item = 0

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
            elif self.actions[action] == "attack 1":
                self.attacking = True

            #elif self.actions[action] == "use 1":
            #    self.using = True
            #    self.attacking = False

        if self.arena[self.player_y][self.player_x] in self.collectable:
            if self.insert_inv(self.arena[self.player_y][self.player_x]):
                self.equipped_item = self.arena[self.player_y][self.player_x]
                self.arena[self.player_y][self.player_x] = 0

        if self.attacking:
            if self.equipped_item in self.tool_dict:
                if self.arena[self.player_y +
                          1][self.player_x] in self.tool_dict[self.equipped_item]:
                           
                        self.arena[self.player_y + 1][self.player_x] = self.block_properties_dict[self.arena[self.player_y + 1][self.player_x]]["drop"]
                
        self.attacking = False
        goal = self.check_reached_goal()
        reward = self.goal_reward if goal else self.step_cost
        if goal: print("SUCCEEDED")

        if self.steps >= self.max_steps:
            terminated = True
        else:
            terminated = goal
        self.steps += 1

    
        obs = self.arena_obs()

        return obs, reward, terminated, {"mission": self.current_mission}

    def __init__(self, num_tools,num_blocks):
       
        self.actions = [
            "movenorth", "movesouth", "movewest", "moveeast","attack 1"
        ]
        self.generate_game_properties(num_tools,num_blocks)
        self.action_space = Discrete(len(self.actions))
        self.observation_space = (2,4)
        self.mission_types = self.blocks
        self.step_cost = -0.1
        self.goal_reward = 10.0
        self.max_steps = 25.0
      
        self.collectable = set(self.drops).union(self.tools)
        self.passable = self.collectable.union([0]).union(self.tools)
        self.poss_spawn_loc = []
        for y in range(2,3):
             for x in range(1, 5):
                 self.poss_spawn_loc.append((y,x))

        self.reset()
        


if __name__ == "__main__":

    env = MalmoEnvSpecial(4,4)
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
