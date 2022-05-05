from collections import defaultdict

from gym.core import Env
from gym import make
from gym import envs
envs = envs.registry.all()
from omegaconf import DictConfig

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict

from .gym_envs.acrobot import AcrobotEnv
from .gym_envs.cartpole import CartPoleEnv
from .gym_envs.mountaincar import MountainCarEnv
from .gym_envs.pendulum import PendulumEnv
from .gym_envs.inverteddoublependulumdisc import InvertedDoublePendulumDiscreteEnv
from .gym_envs.inverteddoublependulum import InvertedDoublePendulumEnv
from .gym_envs.halfcheetah import HalfCheetahEnv
from .isaacgymenvs_wrapper.isaacgymenvs_wrapper import IsaacGymEnvWrapper

our_envs = {
    'acrobot': AcrobotEnv,
    'cartpole': CartPoleEnv,
    'mountaincar': MountainCarEnv,
    'pendulum': PendulumEnv,
    'inverteddoublependulumdisc': InvertedDoublePendulumDiscreteEnv,
    'inverteddoublependulum': InvertedDoublePendulumEnv,
    'halfcheetah': HalfCheetahEnv,
}

env_type_info = {
    'our_mujoco': {
        'env_names': our_envs.keys(),
        'create_func': lambda cfg: our_envs[cfg.name]()
    },
    'gym_mujoco': {
        'env_names': [env_spec.id for env_spec in envs],
        'create_func': lambda cfg: make(cfg.name).unwrapped
    },
    'isaac': {
        'env_names': isaacgym_task_map.keys(),
        'create_func': lambda cfg: IsaacGymEnvWrapper(
        isaacgym_task_map[cfg.name](cfg=omegaconf_to_dict(cfg.task), sim_device=cfg.sim_device,
                                graphics_device_id=cfg.graphics_device_id, headless=cfg.headless)
        )
    }
}

env_name2env_type = defaultdict(str)
for env_type, info in env_type_info.items():
    for env_name in info['env_names']:
        assert not env_name2env_type[env_name], f'repeated environment name: {env_name}'
        env_name2env_type[env_name] = env_type

def create_env(cfg: DictConfig, verbose=False) -> Env:
    env_type = env_name2env_type[cfg.name]
    if verbose:
        print(f'name: {cfg.name}, type: {env_type}')
    return env_type_info[env_type]['create_func'](cfg)