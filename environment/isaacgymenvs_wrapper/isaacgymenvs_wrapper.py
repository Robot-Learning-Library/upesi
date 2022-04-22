from numpy import ndarray
from torch import Tensor, ones_like

class IsaacGymEnvWrapper:
    def __init__(self, env) -> None:
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.env.max_episode_length = 1000000000 # now all resets are invoked manually by .reset()

    def reset(self):
        # .reset() of the unwrapped env is not the actually reset function.
        # .reset_buf is the essential variable
        # now all resets are invoked manually
        action = self.action_space.sample()
        if hasattr(self.env, 'reset_goal_buf'):
            self.env.reset_goal_buf = ones_like(self.env.reset_goal_buf)
        self.env.reset_buf = ones_like(self.env.reset_buf)
        return self.step(action)[0]

    def step(self, action: ndarray):
        action = Tensor(action).cuda()
        next_state, reward, done, info = self.env.step(action)

        next_state = next_state['obs'].detach().cpu().numpy()
        reward = reward.detach().cpu().numpy()
        done = done.detach().cpu().numpy()

        return next_state, reward, done, info

    def render(self):
        # render process is inside self.env.step()
        pass