from gym import spaces
from robosuite.environments.panda_open_door import PandaOpenDoor as RawPandaOpenDoor
import numpy as np
from functools import reduce

class PandaOpenDoor(RawPandaOpenDoor):
    observation_space = spaces.Box(low=-10., high=10., shape=(52,))
    action_space = spaces.Box(low=-1., high=1., shape=(8,)) # todo: magic numbers

    def _get_observation(self):
        obs_ordered_dic = super()._get_observation()
        arrays = map(np.array, obs_ordered_dic.values())
        return reduce(lambda x,y:np.concatenate((x,y)), arrays)