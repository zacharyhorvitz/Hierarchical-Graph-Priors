# from __future__ import print_function

# import malmo.minecraftbootstrap

# from future import standard_library
# standard_library.install_aliases()
# from builtins import input
# from builtins import range
# from builtins import object
# from  malmo import MalmoPython
import json
# import logging
import math
import os
import random
import gym
import sys
import socket
import struct
import time
# import math
# from malmo import malmoutils
import numpy as np

# import tensorflow as tf
# import math
from malmoenv.version import malmo_version
from malmoenv import comms

from lxml import etree
import malmoenv
import malmoenv.bootstrap

import argparse
import time

class MalmoEnvSpecial(gym.Env):
    def checkInventoryForItem(self,obs, requested):
        for i in range(0,9):#39): #Checks primary inventory
            key = 'InventorySlot_'+str(i)+'_item'
            if key in obs:
                item = obs[key]
                if item == requested:
                    return True
        return False

    def checkBlockExists(self,obs, requested):
        return 'floor9x9' in obs and requested in obs['floor9x9']

    def obs_to_vector(self,observation,use_entities=True,flatten=True,expand_dims=True,add_inv=True,add_goal=True):

        state_data_raw = observation #.observations[-1].text)
        
        if 'floor9x9' in state_data_raw:
            state_data_raw = state_data_raw['floor9x9']
        else:
            print("FAILED") 
            return np.zeros((1,1,9,9)) #self.observation_space

        state_data = [self.state_map[block] if block in self.state_map else 0.0 for block in state_data_raw]

        state_data =np.reshape(np.array(state_data,dtype=np.float64),(1,9,9))
        if use_entities: 
            entity_data = self.obs_to_ent_vector(observation,self.relevant_entities)
            if add_inv:
                entity_data[0][entity_data.shape[1] // 2][entity_data.shape[2] // 2] = self.add_inventory(observation)


            if flatten:
                # print(entity_data)
                state_data[np.nonzero(entity_data)] = 0.0
                state_data = state_data + entity_data
            else:
                state_data = np.concatenate((state_data,entity_data),axis=0)

        if add_goal:
            state_data = np.concatenate((np.ones((1,9,1))*self.entity_map[self.goal],state_data),axis=2)


        state_out = np.expand_dims(state_data,0) if expand_dims else state_data

        # print("SHAPE:",state_out.shape)
        return state_out

 
    # def fix_player_location(self,world_state):

    #     if len(world_state) > 0:
    #         observation = json.loads(world_state)
    #         entity_data = observation['entities']
    #         player_data = [x for x in entity_data if x["name"]=="agent"][0]
    #         player_loc = (player_data['x'],player_data['z'])

    #         if abs(math.floor(player_loc[0])-player_loc[0]) != 0.5:
    #             new_x = round(player_loc[0]-0.5)+0.5
    #             self.env.sendCommand("tpx {}".format(new_x))
    #             print("FIXED X")
    #         if abs(math.floor(player_loc[1])-player_loc[1]) != 0.5:
    #             new_z = round(player_loc[1]-0.5)+0.5
    #             self.env.sendCommand("tpz {}".format(new_z))
    #             print("FIXED Z")


    def add_inventory(self,observation):
        key = 'InventorySlot_0_item'
        if observation[key] in self.relevant_entities:
            return self.entity_map[observation[key] ]
        else: return 0



    def obs_to_ent_vector(self,observation,relevant_entities):

        entity_data = observation['entities']
        # print(entity_data)
        player_data = [x for x in entity_data if x["name"]=="agent"][0]
        # print(entity_data)
        entities =  [x for x in entity_data if x["name"] in relevant_entities]
        
        entity_states = np.zeros((1,9,9))
        # print(self.observation_space.shape)
        zero_x = entity_states.shape[2]
        zero_x = zero_x // 2 
        zero_z = entity_states.shape[1]
        zero_z = zero_z // 2 

        player_loc = player_data['x'],player_data['z']

        # print("shifts",zero_x,zero_z)

        for e in entities:
            entity_loc = e['x'],e['z']
            relative_x = entity_loc[0] - player_loc[0]  + zero_x
            relative_z = entity_loc[1] - player_loc[1]  +  zero_z
            # print("coords",relative_x,relative_z)
            entity_states[0][math.floor(relative_z)][math.floor(relative_x)] = float(self.entity_map[e["name"]])
        return entity_states #np.transpose(entity_states,(0,2,1))


    def load_mission_param(self,mission_type):
        mission_dict = {}
        mission_dict["step_cost"] = -0.1
        mission_dict["goal_reward"] = 100.0
        mission_dict["max_steps"] = 100

        if mission_type == "pickaxe_stone":
            mission_dict["state_map"] = {"air":0,"bedrock":1,"stone":2}
            mission_dict["entity_map"] = {"diamond_pickaxe":3,"cobblestone":4}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "cobblestone"
            

        elif mission_type == "axe_log":
            mission_dict["state_map"] = {"air":0,"bedrock":1,"log":5}
            mission_dict["entity_map"] = {"diamond_axe":6,"log":5}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "log"
            
        # elif mission_type == "shovel_clay":
        #     mission_dict["state_map"] = {"air":0,"bedrock":1,"clay":2}
        #     mission_dict["entity_map"] = {"diamond_shovel":3,"clay":4}
        #     mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
        #     mission_dict["goal"] = "clay"
            

        elif mission_type == "hoe_farmland":
            mission_dict["state_map"] = {"air":0,"bedrock":1,"dirt":7,"farmland":8}
            mission_dict["entity_map"] = {"diamond_hoe":9,"dirt":7,"farmland":8}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "farmland"
            

        elif mission_type == "bucket_water":
            mission_dict["state_map"] = {"air":0,"bedrock":1,"water":10}
            mission_dict["entity_map"] = {"bucket":11,"water_bucket":12}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "water_bucket"
            
        # elif mission_type == "sword_pig":
        #     mission_dict["state_map"] = {"air":0,"bedrock":1}
        #     mission_dict["entity_map"] = {"diamond_sword":1,"Pig":2}
        #     mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
        #     mission_dict["goal"] = "porkchop"
            

        # elif mission_type == "sword_cow":
        #     mission_dict["state_map"] = {"air":0,"bedrock":1}
        #     mission_dict["entity_map"] = {"diamond_sword":2,"Cow":3}
        #     mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
        #     mission_dict["goal"] = "beef"
            

        # elif mission_type == "shears_sheep":
        #     mission_dict["state_map"] = {"air":0,"bedrock":1}
        #     mission_dict["entity_map"] = {"shears":2,"Sheep":3}
        #     mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
        #     mission_dict["goal"] = "wool"
    
        # entities = list(mission_dict["relevant_entities"]) + ["water", "stone"]
        # entity_to_idx = {"bucket": 0, "soil": 1, "abstraction": 2, "pig": 3, "dirt": 4, "unpleasant_person": 5, "log": 6, "tool": 7, "pickaxe": 8, "shears": 9, "wool": 10, "solid": 11, "shovel": 12, "meat": 13, "matter": 14, "instrument": 15, "cobblestone": 16, "beef": 17, "artifact": 18, "material": 19, "water_bucket": 20, "sheep": 21, "substance": 22, "instrumentality": 23, "stone": 24, "object": 25, "entity": 26, "hoe": 27, "containerful": 28, "container": 29, "speech_act": 30, "cow": 31, "edge_tool": 32, "bovid": 33, "even-toed_ungulate": 34, "cattle": 35, "water": 36, "porkchop": 37, "physical_entity": 38, "body_waste": 39, "natural_object": 40, "clay": 41, "farmland": 42, "sword": 43, "person": 44, "event": 45, "device": 46, "whole": 47, "axe": 48}
        # env_to_kg_entity = {"diamond_pickaxe" : "pickaxe", "diamond_axe" : "axe", "diamond_shovel" : "shovel", "diamond_hoe" : "hoe", "diamond_sword" : "sword", "Pig" : "pig", "Cow" : "cow", "Sheep" : "sheep"}
        # kg_entities = [env_to_kg_entity[entity] for entity in entities]
        # kg_to_env_entity = {}
        # for k,v in env_to_kg_entity.items(): kg_to_env_entity[v] = k
        # for entity in kg_entities: mission_dict["entity_map"][kg_to_env_entity[entity]] = entity_to_idx[entity]

        if len(mission_dict) == 0:
            print("Invalid mission name:",mission_type)


        return mission_dict

    def build_arena(self):
        arena = ""
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="bedrock" />'.format(-3,203,-3,3,203,3)
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="bedrock" />'.format(-30,203,-10,-3,227,10)
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="bedrock" />'.format(3,203,-10,30,227,10)
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="bedrock" />'.format(-3,203,-10,3,227,-3)
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="bedrock" />'.format(-3,203,3,3,227,10)
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="air" />'.format(-2,204,-2,2,270,2)
         
        return arena

    def get_mission_xml(self,mission_type):
         arena_xml = self.build_arena()

         if mission_type == "pickaxe_stone":  

             mission_xml = self.make_env_string(mission_type,[arena_xml,'<DrawBlock x="{}" y="{}" z="{}" type="stone" />'.format(random.randint(0,4)-2,204,random.randint(1,3)-2),'<DrawItem x="{}" y="{}" z="{}" type="diamond_pickaxe" />'.format(2,206,2) ])
 
         elif  mission_type == "axe_log":   
             mission_xml = self.make_env_string(mission_type,[arena_xml,'<DrawBlock x="{}" y="{}" z="{}" type="log" />'.format(random.randint(0,4)-2,204,random.randint(1,3)-2),'<DrawItem x="{}" y="{}" z="{}" type="diamond_axe" />'.format(2,206,2) ])

         # elif  mission_type == "shovel_clay":   

         #     mission_xml = self.make_env_string(mission_type,[arena_xml,'<DrawBlock x="{}" y="{}" z="{}" type="clay" />'.format(random.randint(0,4)-2,204,random.randint(1,3)-2),'<DrawItem x="{}" y="{}" z="{}" type="diamond_shovel" />'.format(2,206,2) ])

         elif  mission_type == "hoe_farmland":   
             mission_xml = self.make_env_string(mission_type,[arena_xml,'<DrawBlock x="{}" y="{}" z="{}" type="dirt" />'.format(random.randint(0,4)-2,204,random.randint(1,3)-2),'<DrawItem x="{}" y="{}" z="{}" type="diamond_hoe" />'.format(2,206,2) ])

         elif  mission_type == "bucket_water":   
             mission_xml = self.make_env_string(mission_type,[arena_xml,'<DrawBlock x="{}" y="{}" z="{}" type="water" />'.format(random.randint(0,4)-2,204,random.randint(1,3)-2),'<DrawItem x="{}" y="{}" z="{}" type="bucket" />'.format(2,206,2) ])

         # elif  mission_type == "sword_pig":     
         #     pig_pos = (random.randint(1,3)-2,random.randint(1,3)-2)
         #     mission_xml = self.make_env_string(mission_type,[arena_xml,'<DrawEntity x="{}" y="{}" z="{}" type="Pig" />'.format(cow_pos[0],204,cow_pos[1]),'<DrawItem x="{}" y="{}" z="{}" type="diamond_sword" />'.format(2,206,2) ])

         # elif  mission_type == "sword_cow":     
         #     cow_pos = (random.randint(1,3)-2,random.randint(1,3)-2)
         #     mission_xml = self.make_env_string(mission_type,[arena_xml,'<DrawEntity x="{}" y="{}" z="{}" type="Cow" />'.format(cow_pos[0],204,cow_pos[1]),'<DrawItem x="{}" y="{}" z="{}" type="diamond_sword" />'.format(2,206,2) ])

         # elif  mission_type == "shears_sheep":     
         #     sheep_pos = (random.randint(1,3)-2,random.randint(1,3)-2)
         #     mission_xml = self.make_env_string(mission_type,[arena_xml,'<DrawEntity x="{}" y="{}" z="{}" type="Sheep" />'.format(sheep_pos[0],204,sheep_pos[1]),'<DrawItem x="{}" y="{}" z="{}" type="shears" />'.format(2,206,2) ])

         return mission_xml

    def __init__(self,mission_type,port, addr):
        # malmoutils.fix_print()
        # metadata = {'render.modes': ['human']}
        self.env = malmoenv.make()
        self.mission_type = mission_type
        mission_param = self.load_mission_param(self.mission_type)
     #   print(mission_param)
        self.actions =["movenorth","movesouth", "movewest", "moveeast","attack","use"] #,"strafe 1","strafe -1"] #,"attack 1","attack 0"]

        self.observation_space = np.zeros((1,1,9,9))
        self.state_map = mission_param["state_map"]
        self.entity_map = mission_param["entity_map"]
        self.relevant_entities =  mission_param["relevant_entities"]
        self.goal = mission_param["goal"]
        self.step_cost =  mission_param["step_cost"]
        self.goal_reward  =  mission_param["goal_reward"]
        self.max_steps =  mission_param["max_steps"]
        self.port = port
        self.addr = addr
        self.episode = 0
        mission = self.get_mission_xml(self.mission_type)
        self.env.init(mission,server=addr,port=self.port,exp_uid="test",role=0,episode=self.episode,action_filter=self.actions) #, args.port,
        self.action_space = self.env.action_space
      
    def step(self,action):
        _ , _ , done, info = self.env.step(action)
        # print("INFO",len(info))
        reward = 0
        if info is not None and info != "":
            observation = json.loads(info)
            if self.mission_type == "hoe_farmland":
                 reached_goal = self.checkBlockExists(observation,self.goal)
            else:
                 reached_goal = self.checkInventoryForItem(observation,self.goal)

            if reached_goal:
                 done = True
                 reward = self.goal_reward
            else:
                 reward = self.step_cost

            # self.fix_player_location(info)
            obs = self.obs_to_vector(observation)
        else:
            obs = self.observation_space
            obs = np.concatenate((np.ones((1,1,9,1))*self.entity_map[self.goal],obs),axis=3)

        if self.num_steps >= self.max_steps:
           done=True
           
        self.num_steps+=1
        # print(self.num_steps)
        # print(obs)


        return obs[0], reward, done, _ #{"goal":self.goal}
    
    def re_init_mission(self,xml):

        if self.env.resync_period > 0 and (self.env.resets + 1) % self.env.resync_period == 0:
            self.env.exit_resync()

        while not self.env.done:
            self.env.done = self.env._quit_episode()
            if not self.env.done:
                time.sleep(0.1)

        self.env.last_obs = None
        self.env.resets += 1
        if self.env.role != 0:
            self.env._find_server()
        if not self.env.client_socket:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # print("connect " + self.server2 + ":" + str(self.port2))
            sock.connect((self.env.server2, self.env.port2))
            self.env._hello(sock)
            self.env.client_socket = sock  # Now retries will use connected socket.
        self.init_miss(xml)
        self.env.done = False
        return self.env._peek_obs()

    def init_miss(self,raw_xml):
        ok = 0
        while ok != 1:
            # print(xml)
            # new_xml = etree.parse(xml)
            xml = etree.tostring(self.build_xml(raw_xml)) #self.env.xml)
            # print(xml)

            token = (self.env._get_token() + ":" + str(self.env.agent_count)).encode()
            # print(xml.decode())
            comms.send_message(self.env.client_socket, xml)
            comms.send_message(self.env.client_socket, token)

            reply = comms.recv_message(self.env.client_socket)
            ok, = struct.unpack('!I', reply)
            self.turn_key = comms.recv_message(self.env.client_socket).decode('utf-8')
            if ok != 1:
                time.sleep(1)
        


    def build_xml(self,xml):
        # if action_filter is None:
        #     action_filter = {"move", "turn", "use", "attack"}

        # if not xml.startswith('<Mission'):
        #     i = xml.index("<Mission")
        #     if i == -1:
        #         raise EnvException("Mission xml must contain <Mission> tag.")
        #     xml = xml[i:]

        xml = etree.fromstring(xml)
        # self.role = role
        # if exp_uid is None:
        #     self.exp_uid = str(uuid.uuid4())
        # else:
        #     self.exp_uid = exp_uid

        # command_parser = CommandParser(action_filter)
        # commands = command_parser.get_commands_from_xml(self.xml, self.role)
        # actions = command_parser.get_actions(commands)
        # print("role " + str(self.role) + " actions " + str(actions)

        # if action_space:
        #     self.action_space = action_space
        # else:
        #     self.action_space = ActionSpace(actions)

        # self.port = port
        # if server is not None:
        #     self.server = server
        # if server2 is not None:
        #     self.server2 = server2
        # else:
        #     self.server2 = self.server
        # if port2 is not None:
        #     self.port2 = port2
        # else:
        #     self.port2 = self.port + self.role

        # self.agent_count = len(self.xml.findall(self.ns + 'AgentSection'))
        # turn_based = self.xml.find('.//' + self.ns + 'TurnBasedCommands') is not None
        # if turn_based:
        #     self.turn_key = 'AKWozEre'
        # else:
        #     self.turn_key = ""
        # if step_options is None:
        #     self.step_options = 0 if not turn_based else 2
        # else:
        #     self.step_options = step_options
        # self.done = True
        # # print("agent count " + str(self.agent_count) + " turn based  " + turn_based)
        # self.resync_period = resync
        # self.resets = episode

        e = etree.fromstring("""<MissionInit xmlns="http://ProjectMalmo.microsoft.com" 
                                xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
                                SchemaVersion="" PlatformVersion=""" + '\"' + malmo_version + '\"' +
                             f""">
                                <ExperimentUID></ExperimentUID>
                                <ClientRole>0</ClientRole>
                                <ClientAgentConnection>
                                    <ClientIPAddress>{self.addr}</ClientIPAddress>
                                    <ClientMissionControlPort>0</ClientMissionControlPort>
                                    <ClientCommandsPort>0</ClientCommandsPort>
                                    <AgentIPAddress>127.0.0.1</AgentIPAddress>
                                    <AgentMissionControlPort>0</AgentMissionControlPort>
                                    <AgentVideoPort>0</AgentVideoPort>
                                    <AgentDepthPort>0</AgentDepthPort>
                                    <AgentLuminancePort>0</AgentLuminancePort>
                                    <AgentObservationsPort>0</AgentObservationsPort>
                                    <AgentRewardsPort>0</AgentRewardsPort>
                                    <AgentColourMapPort>0</AgentColourMapPort>
                                    </ClientAgentConnection>
                                </MissionInit>""")
        e.insert(0, xml)
        xml = e
        xml.find(self.env.ns + 'ClientRole').text = str(self.env.role)
        xml.find(self.env.ns + 'ExperimentUID').text = self.env.exp_uid

        # if self.role != 0 and self.agent_count > 1:
        #     e = etree.Element(self.ns + 'MinecraftServerConnection',
        #                       attrib={'address': self.server,
        #                               'port': str(0)
        #                               })
        #     self.xml.insert(2, e)

        return xml


    def reset(self,remake_mission=True,random_mission = True):
        self.num_steps = 0
       
        if remake_mission:
            if random_mission:
                new_mission = random.choice(["pickaxe_stone","axe_log","hoe_farmland","bucket_water"])
                self.mission_type = new_mission
                mission_param = self.load_mission_param(self.mission_type)
                self.state_map = mission_param["state_map"]
                self.entity_map = mission_param["entity_map"]
                self.relevant_entities =  mission_param["relevant_entities"]
                self.goal = mission_param["goal"]
                self.step_cost =  mission_param["step_cost"]
                self.goal_reward  =  mission_param["goal_reward"]
                self.max_steps =  mission_param["max_steps"]

            self.re_init_mission(self.get_mission_xml(self.mission_type))

           # self.env.close()
           # self.env.xml = self.get_mission_xml(self.mission_type)
           # self.env._quit_episode()

           # self.env = malmoenv.make()
           # self.env.init(self.get_mission_xml(self.mission_type),server='127.0.0.1',port=self.port,exp_uid="test",role=0,episode=self.episode,action_filter=self.actions) #, args.port,
        else:
            self.env.reset()
        self.episode+=1
        # self.env.reset()
        time.sleep(1)
        obs, _, _, _= self.step(0)
        return obs

    def make_env_string(self,mission_type,draw_entities=[]):
      #  '<?xml version="1.0" standalone="no" ?>
        base = '<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'

        # base = '<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        base+='<About><Summary>Running {}...</Summary></About>'.format(mission_type)
        base+= '<ModSettings><MsPerTick>1</MsPerTick></ModSettings>' #1
        base+= '<ServerSection><ServerInitialConditions><Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime>'
        base+= '</Time><Weather>clear</Weather><AllowSpawning>false</AllowSpawning></ServerInitialConditions><ServerHandlers><FlatWorldGenerator />' 
        base+= '<DrawingDecorator>'

        for entity_info in draw_entities:
            base+=entity_info

        base+='</DrawingDecorator>'
        base+= '<ServerQuitFromTimeUp timeLimitMs="10000000"/><ServerQuitWhenAnyAgentFinishes/></ServerHandlers></ServerSection>'
        base+= '<AgentSection mode="Survival"><Name>agent</Name><AgentStart><Placement x="-1.5" y="204" z="-1.5" pitch="50" yaw="0"/>' #50
        base+= '<Inventory></Inventory>'

        base+='</AgentStart>'
        base+='<AgentHandlers>'
        base+='<ObservationFromGrid> <Grid name="floor9x9"> <min x="-4" y="0" z="-4"/> <max x="4" y="0" z="4"/> </Grid> </ObservationFromGrid>'

        base+='<ObservationFromNearbyEntities><Range name="entities" xrange="5" yrange="5" zrange="5"/></ObservationFromNearbyEntities>'
        base+='<ObservationFromFullInventory/><ObservationFromFullStats/><VideoProducer want_depth="false"><Width>640</Width><Height>480</Height></VideoProducer>'
        base+='<DiscreteMovementCommands><ModifierList type="deny-list"><command>attack</command><command>use</command></ModifierList></DiscreteMovementCommands>'
        base+='<ContinuousMovementCommands><ModifierList type="allow-list"><command>attack</command><command>use</command></ModifierList>'
        base+='</ContinuousMovementCommands><MissionQuitCommands quitDescription="done"/>'
        base+='<AbsoluteMovementCommands><ModifierList type="deny-list"></ModifierList></AbsoluteMovementCommands>'
        base+='</AgentHandlers></AgentSection></Mission>'
        return base


if __name__ == "__main__":
    print("starting server...")

    if len(sys.argv) > 1 and sys.argv[1] == "RUN_SERVER":
        print("Launching on port " + sys.argv[2])
        malmoenv.bootstrap.launch_minecraft(int(sys.argv[2]), installdir=sys.argv[3])
        exit()

    print("initializing environment...")
    env = MalmoEnvSpecial("pickaxe_stone",port=9000, addr=sys.argv[4])
    obs = env.reset()
    print("reset")
    for step in range(100):
        print("\n",step)
        command = int(input())
        obs, reward, done, info = env.step(command)
        print(obs)
        print(reward)
#
