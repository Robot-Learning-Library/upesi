import isaacgym # must be imported before torch

from omegaconf import OmegaConf
# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

from upesi_utils.load_params import load_default_training_params
from environment import create_env, env_name2env_type

if __name__ == '__main__':
    cfg = OmegaConf.merge(OmegaConf.load('cfg/config.yaml'), OmegaConf.from_cli())
    cfg.train = OmegaConf.create(load_default_training_params(cfg.basic.alg, cfg.basic.env_name))
    if env_name2env_type[cfg.basic.env_name] == 'isaac':
        cfg.env.using_isaacgym = True
        cfg.env.raw_env_cfg = OmegaConf.load(cfg.basic.main_yaml_path)
        cfg.env.raw_env_cfg.task = OmegaConf.load(cfg.basic.task_yaml_path)

    env = create_env(cfg.env, verbose=True)
    print(f'Observation space: {env.observation_space}')
    print(f'Action space: {env.action_space}')

    if cfg.basic.alg=='td3':
        from rl.td3.train_td3 import train_td3
        train_td3(env, cfg)
    elif cfg.basic.alg=='ppo':
        from rl.ppo.train_ppo import train_ppo
        raise NotImplemented
    else:
        print("Algorithm type is not implemented!")