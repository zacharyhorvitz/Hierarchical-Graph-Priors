import gym 

class FakeActions(gym.Wrapper):

    def __init__(self, env, num_new_actions):
        gym.Wrapper.__init__(self, env)
        self.old_actions = len(env.actions)
        assert num_new_actions > self.old_actions, f"New actions must be larger than {old_actions}"
        self.action_space = gym.spaces.Discrete(num_new_actions)
        self.mission_types = env.mission_types

    def step(self, action):
        if action >= self.old_actions:
            return self.env.step(len(self.env.actions) - 1) # noop action
        else:
            return self.env.step(action)

