from gym.core import Env
from gym import make
from gym import envs
envs = envs.registry.all()
from omegaconf import DictConfig

from .gym_envs.acrobot import AcrobotEnv
from .gym_envs.cartpole import CartPoleEnv
from .gym_envs.mountaincar import MountainCarEnv
from .gym_envs.pendulum import PendulumEnv
from .gym_envs.inverteddoublependulumdisc import InvertedDoublePendulumDiscreteEnv
from .gym_envs.inverteddoublependulum import InvertedDoublePendulumEnv
from .gym_envs.inverteddoublependulumdynamics import InvertedDoublePendulumDynamicsEnv
from .gym_envs.inverteddoublependulumdynamicsembedding import InvertedDoublePendulumDynamicsEmbeddingEnv
from .gym_envs.halfcheetah import HalfCheetahEnv
from .gym_envs.halfcheetahdynamics import HalfCheetahDynamicsEnv
from .gym_envs.halfcheetahdynamicsembedding import HalfCheetahDynamicsEmbeddingEnv
from .isaacgymenvs_wrapper.isaacgymenvs_wrapper import IsaacGymEnvWrapper

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict

our_envs = {
    'acrobot': AcrobotEnv,
    'cartpole': CartPoleEnv,
    'mountaincar': MountainCarEnv,
    'pendulum': PendulumEnv,
    'inverteddoublependulumdisc': InvertedDoublePendulumDiscreteEnv,
    'inverteddoublependulum': InvertedDoublePendulumEnv,
    'inverteddoublependulumdynamics': InvertedDoublePendulumDynamicsEnv,
    'inverteddoublependulumdynamicsembedding': InvertedDoublePendulumDynamicsEmbeddingEnv,
    'halfcheetah': HalfCheetahEnv,
    'halfcheetahdynamics': HalfCheetahDynamicsEnv,
    'halfcheetahdynamicsembedding': HalfCheetahDynamicsEmbeddingEnv,
}

MUJOCO_OUR_ENVS_LIST = our_envs.keys()
MUJOCO_GYM_ENVS_LIST = [env_spec.id for env_spec in envs]
ISAAC_GYM_ENVS_LIST = isaacgym_task_map.keys()

def create_env(cfg: DictConfig, verbose=False) -> Env:
    if cfg.name in MUJOCO_OUR_ENVS_LIST:
        if verbose:
            print(f'name: {cfg.name}, type: ours')
        return our_envs[cfg.name]()
    elif cfg.name in MUJOCO_GYM_ENVS_LIST:
        if verbose:
            print(f'name: {cfg.name}, type: gym mujoco')
        return make(cfg.name).unwrapped
    elif cfg.name in ISAAC_GYM_ENVS_LIST:
        if verbose:
            print(f'name: {cfg.name}, type: isaacgym')
        env = isaacgym_task_map[cfg.name](
            cfg=omegaconf_to_dict(cfg.task),
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless
        )
        return IsaacGymEnvWrapper(env)