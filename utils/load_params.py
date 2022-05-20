# for consistence in comparison among different methods
from importlib import import_module  # dynamic module importing

def load_default_training_params(alg_name, env_name):
    module = import_module('default_params_' + alg_name)
    print(f'Loading default training params from: {module}')
    return module.get_hyperparams(env_name)