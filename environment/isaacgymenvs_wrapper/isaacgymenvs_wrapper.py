from numpy import ndarray
from torch import Tensor

class IsaacGymEnvWrapper:
    def __init__(self, env) -> None:
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        state_dict = self.env.reset()
        state_gpu_tensor: Tensor = state_dict['obs']
        return state_gpu_tensor.detach().cpu().numpy()

    def step(self, action: ndarray):
        action = Tensor(action).cuda()
        next_state, reward, done, info = self.env.step(action)

        next_state = next_state['obs'].detach().cpu().numpy()
        reward = reward.detach().cpu().numpy()
        done = done.detach().cpu().numpy()

        return next_state, reward, done, info