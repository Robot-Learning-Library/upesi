import isaacgym # must be imported before torch

from omegaconf import DictConfig, OmegaConf
# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

import numpy as np
import random
import torch
import sys, os
sys.path.append('..')
from dynamics_networks import DynamicsNetwork, SINetwork, DynamicsParamsOptimizer, EncoderDynamicsNetwork, EncoderDecoderDynamicsNetwork, VAEDynamicsNetwork
from rl.policy_networks import DPG_PolicyNetwork
from upesi_utils.load_params import load_default_training_params
from upesi_utils.common_func import rand_params
from environment import create_env, env_name2env_type
from defaults import DYNAMICS_PARAMS, HYPER_PARAMS
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

torch.manual_seed(1234)  # Reproducibility
np.random.seed(1234)
random.seed(1234)


def load_policy(env, load_from, params, policy_class=DPG_PolicyNetwork):
    """
    Load a model-free reinforcement learning policy.
    """
    policy = policy_class(env.observation_space, env.action_space, params['hidden_dim'], params['action_range'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    policy = policy.to(device)
    
    if load_from:
        policy.load_state_dict(torch.load(load_from, map_location=device))
    return policy

def collect_train_data(env_name: str, env_cfg: DictConfig, load_from=None, save_to=None, episodes=10000):
    """
    Collect the dataset for training the dynamics forward prediction model.
    """
    env = create_env(env_cfg, verbose=True)
    train_param_dict = load_default_training_params('td3', env_name)
    policy = load_policy(env, load_from, train_param_dict)
    if not policy.action_range:
        policy.action_range = env.action_space.high[0]  # mujoco env gives the range of action and it is symmetric

    data = []
    for ep in range(episodes):
        print(ep)
        ep_r = 0
        if env_cfg.using_isaacgym:
            pass
        else:
            rand_param_dict, param_vec = rand_params(env, DYNAMICS_PARAMS[env.name+'dynamics'])
            s = env.reset(**rand_param_dict)
        for step in range(train_param_dict['max_steps']):
            a = policy.get_action(s, env_cfg.sim_batch_size, noise_scale=0.0)
            s_, r, d, _ = env.step(a)
            # env.render()
            data.append([[s, a, param_vec], s_])
            s=s_
            ep_r+=r
            if d:
                break
    if save_to: 
        os.makedirs(save_to, exist_ok=True)
        np.save(save_to+'/dynamics.npy', data)


def collect_si_train_data(Env, load_from=None, save_to=None, frame_stack=5, episodes=10000, env_settings={}, default_params={},):
    """
    Collect the dataset for training the system identification model.
    """
    env = Env(**env_settings, **default_params)
    params_key  = ['max_steps', 'hidden_dim', 'action_range']
    params = {k:v for k,v in zip (params_key, load_params('td3', env.__class__.__name__.lower(), params_key))}
    policy = load_policy(env, load_from, params)
    if not policy.action_range:
        policy.action_range = env.action_space.high[0]  # mujoco env gives the range of action and it is symmetric

    data = []
    for ep in range(episodes):
        print(ep)
        ep_r = 0
        param_dict, param_vec = rand_params(env, DYNAMICS_PARAMS[env.name+'dynamics'])
        s = env.reset(**param_dict)
        sample=[]
        for step in range(params['max_steps']):
            a = policy.get_action(s, noise_scale=0.0)
            s_, r, d, _ = env.step(a)
            # env.render()
            sample.append(np.concatenate([s,a]))

            s=s_
            ep_r+=r
            if d:
                break

        for i in range(len(sample)-frame_stack):
            data.append([np.array(sample[i:i+frame_stack]).reshape(-1), param_vec])  # shape: [(s,a,s,a,..s,a), param]
    
    if save_to: 
        os.makedirs(save_to, exist_ok=True)
        np.save(save_to+'/data.npy', data)

def process_si_train_data(Env, load_from=None, save_to=None, frame_stack=5):
    data = np.load(load_from, allow_pickle=True)
    proc_data=[]
    sa=[]
    param=[]
    data_l = len(data)
    for i in range(data_l):
        sa.append(np.concatenate(data[i][0][:2])) # sa
        param.append(data[i][0][-1])
    num=0
    for i in range(data_l-frame_stack):
        if param[i] == param[i+frame_stack-1]:
            sample = [np.array(sa[i:i+frame_stack]).reshape(-1), param[i]]
            proc_data.append(sample)
            num+=1

    print('all: ', data_l, '  processed: ', num)
    if save_to: 
        os.makedirs(save_to, exist_ok=True)
        np.save(save_to+'/data.npy', proc_data)


def collect_test_data(Env, load_from=None, save_to=None, num_params=10, episodes_per_param=100, env_settings={}, default_params={},):
    """
    Collect the test dataset.
    """
    env = Env(**env_settings, **default_params)
    params_key  = ['max_steps', 'hidden_dim', 'action_range']
    params = {k:v for k,v in zip (params_key, load_params('td3', env.__class__.__name__.lower(), params_key))}
    policy = load_policy(env, load_from, params)
    if not policy.action_range:
        policy.action_range = env.action_space.high[0]  # mujoco env gives the range of action and it is symmetric

    data = []
    
    for n in range(num_params):
        print(n)
        sal = []
        s_l = []
        param_dict, param_vec = rand_params(env, DYNAMICS_PARAMS[env.name+'dynamics'])
        for epi in range(episodes_per_param):
            s = env.reset(**param_dict)
            for step in range(params['max_steps']):
                a = policy.get_action(s, noise_scale=0.0)
                s_, r, d, _ = env.step(a)
                # env.render()
                sal.append(np.concatenate((s,a)))
                s_l.append(s_)
                s=s_
                if d:
                    break
        data.append({
        'params': param_vec,
        'sa': sal,
        's_': s_l,
        })
    if save_to: 
        os.makedirs(save_to, exist_ok=True)
        np.save(save_to+'/test_dynamics.npy', data)

def train_dynamics(Env, param_dim, epoch=20000, batch_size=64, batch_per_epoch=30, env_settings={}, default_params={}, 
    save_to=None, load_from=None, data_path='./', report_loss=lambda ep, r: print(ep, r), print_interval=10, save_interval=200):
    """
    Train the dynamics forward prediction model with a collected dataset.
    Model input: state, action, dynamics parameters.
    Model output: next state.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    env = Env(**env_settings, **default_params)
    model = DynamicsNetwork(env.observation_space, env.action_space, param_dim)

    if load_from:
        model.load_state_dict(torch.load(load_from, map_location=device))
    if save_to:
        os.makedirs(save_to, exist_ok=True)
        torch.save(model.state_dict(), save_to + '/0.pth')

    data = np.load(data_path, allow_pickle=True)
    data = data.tolist()
    split = int(len(data)*0.8)
    train_data, test_data = data[:split], data[split:]
    running_sum_loss = 0.

    for ep in range(epoch):
        running_loss = 0.

        for n_batch in range(batch_per_epoch):
            x_train, y_train = zip(*random.sample(train_data, min(len(train_data), batch_size)))
            x_train = model.saparm_to_inputs(x_train, device)
            y_train = torch.FloatTensor(y_train).to(device)
            # print(x_train.shape, y_train.shape)
            y_pred = model(x_train)
            model.optimizer.zero_grad()
            loss = model.criterion(y_pred, y_train)
            loss.backward()
            model.optimizer.step()
            running_loss += loss.item()

        running_sum_loss += running_loss / batch_per_epoch  # loss per sample

        if (ep + 1) % print_interval == 0:
            loss_train = running_sum_loss / print_interval
            report_loss(ep + 1, loss_train)
            running_sum_loss = 0.

            x_test, y_test = zip(*random.sample(test_data, min(len(test_data), 100*batch_size)))
            x_test = model.saparm_to_inputs(x_test, device)
            y_test = torch.FloatTensor(y_test).to(device)
            y_pred_ = model(x_test)
            loss_test = model.criterion(y_pred_, y_test).item()
            print('test loss: ', loss_test)

            writer.add_scalars('Loss', {'train': loss_train, 'test': loss_test}, ep)

        if (ep + 1) % save_interval == 0:
            if save_to:
                torch.save(model.state_dict(),
                            save_to + '/{}.pth'.format(ep + 1))

    env.close()

def train_params(Env, embedding, epoch=1000, dynamics_model_path='./', env_settings={}, default_params={}, data_path='./'):
    """
    Train the dynamics parameters with a trained/fixed dynamics prediction model, using the real-world dataset.
    Model input: state, action, dynamics parameters (learnable).
    Model output: next state.
    """
    env = Env(**env_settings, **default_params)
    if embedding:
        param_dim = HYPER_PARAMS[Env.name+'dynamics']['latent_dim']
        param_ini_v = np.zeros(param_dim)
        model = DynamicsParamsOptimizer(env.observation_space, env.action_space, param_dim, param_ini_v)
    else:
        param_dim = len(DYNAMICS_PARAMS[Env.name+'dynamics'])
        param_ini_v = [np.mean(v) for k, v in env.parameters_spec.items() if k in DYNAMICS_PARAMS[Env.name+'dynamics']]
        model = DynamicsParamsOptimizer(env.observation_space, env.action_space, param_dim, param_ini_v)
    model.dynamics_model.load_state_dict(torch.load(dynamics_model_path))
    model.dynamics_model.eval()

    sa = np.load(data_path+'/sa.npy')
    s_ = np.load(data_path+'/s_.npy')
    sa = torch.Tensor(sa)
    s_ = torch.Tensor(s_)

    for ep in range(epoch):
        s_pred = model.forward(sa)
        model.optimizer.zero_grad()
        loss = model.criterion(s_pred, s_)
        loss.backward()
        model.optimizer.step()
        print('epoch: {}, loss: {}'.format(ep, loss.item()))
 
    print(model.params)


def train_si(Env, param_dim, epoch=10000, data_path='./', save_to='./', batch_size=64, batch_per_epoch=30, \
    env_settings={}, default_params={}, print_interval=10, save_interval=10, ):
    """
    Train the system identification model.
    Model input: (state, action, ...., state, action).
    Model output: param.
    """
    env = Env(**env_settings, **default_params)

    data = np.load(data_path, allow_pickle=True)
    data = data.tolist()
    split = int(len(data)*0.8)
    train_data, test_data = data[:split], data[split:]
    model = SINetwork(env.observation_space, env.action_space, param_dim)
    running_sum_loss = 0.

    for ep in range(epoch):
        running_loss = 0.

        for n_batch in range(batch_per_epoch):
            x_train, y_train = zip(*random.sample(train_data, min(len(train_data), batch_size)))
            x_train = torch.FloatTensor(np.vstack(x_train)) # array of list to 2d array
            y_train = torch.FloatTensor(np.vstack(y_train))  # array of list to 2d array)

            pre_param = model(x_train)
            model.optimizer.zero_grad()
            loss= model.criterion(pre_param, y_train)
            loss.backward()
            model.optimizer.step()
            running_loss += loss.item()
        running_sum_loss += running_loss / batch_per_epoch  # loss per sample

        if (ep + 1) % print_interval == 0:
            loss_train = running_sum_loss / print_interval
            running_sum_loss = 0
            x_test, y_test = zip(*random.sample(test_data, min(len(test_data), batch_size)))
            x_test = torch.FloatTensor(np.vstack(x_test)) # array of list to 2d array
            y_test = torch.FloatTensor(np.vstack(y_test))  # array of list to 2d array)

            pre_param = model(x_test)
            model.optimizer.zero_grad()
            loss= model.criterion(pre_param, y_test)
            loss.backward()
            model.optimizer.step()
            loss_test = loss.item()

            print("Epoch:  {}   Loss Train:  {}   Test: {}".format(ep+1, loss_train, loss_test))
            writer.add_scalars('Loss', {'train': loss_train, 'test': loss_test}, ep)

        if (ep + 1) % save_interval == 0:
            if save_to:
                os.makedirs(save_to, exist_ok=True)
                torch.save(model.state_dict(), save_to+'/si')
    env.close()

def train_dynamics_embedding(Env, param_dim, Type = 'EncoderDynamicsNetwork', epoch=20000, batch_size=64, batch_per_epoch=30, env_settings={}, default_params={}, 
    save_to=None, load_from=None, data_path='./', print_interval=2, save_interval=100):
    """
    Train the dynamics forward prediction model and dynamics encoder-decoder with a collected dataset.
    Model input: state, action, dynamics parameters.
    Model output: next state.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    env = Env(**env_settings, **default_params)

    assert Type in ['EncoderDynamicsNetwork', 'EncoderDecoderDynamicsNetwork', 'VAEDynamicsNetwork']
    model = eval(Type)(env.observation_space, env.action_space, param_dim, latent_dim=HYPER_PARAMS[env.name+'dynamics']['latent_dim'])

    if load_from:
        model.load_state_dict(torch.load(load_from, map_location=device))
    if save_to:
        os.makedirs(save_to, exist_ok=True)
        torch.save(model.state_dict(), save_to + '/0.pth')

    data = np.load(data_path, allow_pickle=True)
    data = data.tolist()
    split = int(len(data)*0.8)
    train_data, test_data = data[:split], data[split:]
    running_sum_loss1 = 0.
    running_sum_loss2 = 0.
    running_sum_loss = 0.

    for ep in range(epoch):
        running_loss1 = 0.
        running_loss2 = 0.
        running_loss = 0.

        for n_batch in range(batch_per_epoch):
            x_train, y_train = zip(*random.sample(train_data, min(len(train_data), batch_size)))
            s,a,param = np.stack(x_train, axis=1)   # separate three types of data
            param = np.vstack(param)  # array of list to 2d array

            sa = torch.FloatTensor(np.concatenate((np.vstack(s),np.vstack(a)), axis=-1))   
            param = torch.FloatTensor(param)            
            s_ = torch.FloatTensor(y_train).to(device)
            pre_param, pre_s_ = model(sa, param)
            model.optimizer1.zero_grad()
            model.optimizer2.zero_grad()
            loss1= model.loss_dynamics(pre_s_, s_)
            if Type == 'EncoderDynamicsNetwork':
                loss2=torch.tensor(0.)
            elif Type == 'EncoderDecoderDynamicsNetwork':
                loss2= model.loss_recon(pre_param, param)
            elif Type == 'VAEDynamicsNetwork':
                loss2, _, _= model.loss_vae(pre_param, param)
            loss = loss1+loss2
            loss.backward()
            model.optimizer1.step()
            model.optimizer2.step()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss += loss1.item() + loss2.item()

        running_sum_loss1 += running_loss1 / batch_per_epoch  # loss per sample
        running_sum_loss2 += running_loss2 / batch_per_epoch  # loss per sample
        running_sum_loss += running_loss / batch_per_epoch  # loss per sample
        # model.scheduler.step()


        if (ep + 1) % print_interval == 0:
            loss_train = running_sum_loss / print_interval
            loss1_train = running_sum_loss1 / print_interval
            loss2_train = running_sum_loss2 / print_interval

            print('train loss: {}, loss dynamics: {}, loss encoder-decoder: {}'.format(loss_train, loss1_train, loss2_train))
            running_sum_loss, running_sum_loss1, running_sum_loss2 = 0., 0., 0.

            x_test, y_test = zip(*random.sample(test_data, min(len(test_data), 100*batch_size)))
            s,a,param = np.stack(x_test, axis=1) 
            param = np.vstack(param)  # array of list to 2d array

            sa = torch.FloatTensor(np.concatenate((np.vstack(s),np.vstack(a)), axis=-1))   
            param = torch.FloatTensor(param)            
            s_ = torch.FloatTensor(y_test).to(device)
            pre_param, pre_s_ = model(sa, param)
            loss1_test= model.loss_dynamics(pre_s_, s_).item()
            if Type == 'EncoderDynamicsNetwork':
                loss2_test = torch.tensor(0.)
            elif Type == 'EncoderDecoderDynamicsNetwork':
                loss2_test = model.loss_recon(pre_param, param)
            elif Type == 'VAEDynamicsNetwork':
                loss2_test, loss2_rec, loss2_kld = model.loss_vae(pre_param, param)
            loss2_test = loss2_test.item()
            loss_test = loss1_test+loss2_test
            print('test loss: {}, loss dynamics: {}, loss encoder-decoder: {}'.format(loss_test, loss1_test, loss2_test))

            writer.add_scalars('Loss', {'train': loss_train, 'test': loss_test}, ep)
            writer.add_scalars('Loss Dynamics', {'train': loss1_train, 'test': loss1_test}, ep)
            writer.add_scalars('Loss Encoder-Decoder', {'train': loss2_train, 'test': loss2_test}, ep)
            if Type == 'VAEDynamicsNetwork':
                writer.add_scalars('Test Loss VAE', {'Reconstruction': loss2_rec.item(), 'KL-divergence': loss2_kld.item()}, ep)

        if (ep + 1) % save_interval == 0:
            if save_to:
                model.save_model(save_to)

    env.close()

if __name__ == '__main__':
    cfg = OmegaConf.merge(OmegaConf.load('../cfg/dynamics.yaml'), OmegaConf.from_cli())
    assert os.path.exists(cfg.basic.load_path)
    print(f'current path: {cfg.basic.load_path}, env: {cfg.basic.env_name}, parameters dimension: {len(cfg.dynamics_params_list.current)}')

    if env_name2env_type[cfg.basic.env_name] == 'isaac':
        cfg.env.using_isaacgym = True
        cfg.env.raw_env_cfg = OmegaConf.load(cfg.basic.main_yaml_path)
        cfg.env.raw_env_cfg.task = OmegaConf.load(cfg.basic.task_yaml_path)

    if cfg.basic.command == 'collect_train_data':
        collect_train_data(cfg.basic.env_name, cfg.env, cfg.basic.load_path, cfg.basic.save_dir, cfg.basic.episodes)
    
    if args.CollectTestData:
        if args.env == 'pandapushik2dsimple':
            collect_test_data(Env, load_from=path+'/data/weights/20210119_2007/4000'+'_td3_policy', save_to=path+'/data/dynamics_data/'+args.env, episodes_per_param=10)
        if args.env == 'pandapushfk':
            collect_test_data(Env, load_from=path+'/data/weights/20210301_222527/4999'+'_td3_policy', save_to=path+'/data/dynamics_data/'+args.env, episodes_per_param=30)
        elif args.env == 'inverteddoublependulum':
            collect_test_data(Env, load_from=path+'/data/weights/20201230_1735/1950'+'_td3_policy', save_to=path+'/data/dynamics_data/'+args.env, episodes_per_param=1000)
        elif args.env == 'halfcheetah':
            collect_test_data(Env, load_from=path+'/data/weights/20210203_153134/22000'+'_td3_policy', save_to=path+'/data/dynamics_data/'+args.env, episodes_per_param=100)

    if args.ProcessSITrainData:
        if args.env == 'pandapushik2dsimple':
            process_si_train_data(Env, load_from=path+'/data/dynamics_data/{}/norm_dynamics.npy'.format(args.env), save_to=path+'/data/si_data/'+args.env)
        if args.env == 'pandapushfk':
            process_si_train_data(Env, load_from=path+'/data/dynamics_data/{}/norm_dynamics.npy'.format(args.env), save_to=path+'/data/si_data/'+args.env)
        elif args.env == 'inverteddoublependulum':
            process_si_train_data(Env, load_from=path+'/data/dynamics_data/{}/norm_dynamics.npy'.format(args.env), save_to=path+'/data/si_data/'+args.env)
        elif args.env == 'halfcheetah':
            process_si_train_data(Env, load_from=path+'/data/dynamics_data/{}/norm_dynamics.npy'.format(args.env), save_to=path+'/data/si_data/'+args.env)
    
    # if args.CollectSITestData:
    #     if args.env == 'pandapushik2dsimple':
    #         collect_si_test_data(Env, load_from=path+'/data/weights/20210119_2007/4000'+'_td3_policy', save_to=path+'/data/si_data/'+args.env, episodes_per_param=10)
    #     if args.env == 'pandapushfk':
    #         collect_si_test_data(Env, load_from=path+'/data/weights/20210301_222527/4999'+'_td3_policy', save_to=path+'/data/si_data/'+args.env, episodes_per_param=30)
    #     elif args.env == 'inverteddoublependulum':
    #         collect_si_test_data(Env, load_from=path+'/data/weights/20201230_1735/1950'+'_td3_policy', save_to=path+'/data/si_data/'+args.env, episodes_per_param=1000)
    #     elif args.env == 'halfcheetah':
    #         collect_si_test_data(Env, load_from=path+'/data/weights/20210203_153134/22000'+'_td3_policy', save_to=path+'/data/si_data/'+args.env, episodes_per_param=100)
 
    if args.TrainSI:
        train_si(Env, param_dim=param_dim, data_path=path+'/data/si_data/{}/data.npy'.format(args.env), save_to=path+'/data/si_data/{}/model/dim{}'.format(args.env, str(param_dim)))
    
    file_name = ['dynamics', 'norm_dynamics'][1]
    Type = ['EncoderDynamicsNetwork', 'EncoderDecoderDynamicsNetwork', 'VAEDynamicsNetwork'][2]
    
    if args.TrainDynamics:  # train dynamics forward prediction model without embedding
        train_dynamics(Env, param_dim=param_dim, data_path=path+'/data/dynamics_data/{}/{}.npy'.format(args.env, file_name), save_to=path+'/data/dynamics_data/{}/model/dim{}'.format(args.env, str(param_dim)))

    if args.TrainParams: # train parameters with a pre-trained dynamics prediction model and real dataset
        embedding = True
        if args.env == 'pandapushik2dsimple':
            data_path = path+'/data/real/processed_traj'
        elif args.env == 'inverteddoublependulum':
            print("No real data for env: ", args.env)
        
        train_params(Env, embedding, dynamics_model_path=path+'/data/dynamics_data/{}/model/{}_dim{}/dynamics'.format(args.env, Type, str(param_dim)), data_path=data_path)

    if args.TrainEmbedding:  # train dynamics forward prediction model and the embedding model
        train_dynamics_embedding(Env, Type = Type, param_dim=param_dim, data_path=path+'/data/dynamics_data/{}/{}.npy'.format(args.env, file_name), \
            save_to=path+'/data/dynamics_data/{}/model/{}_dim{}/'.format(args.env, Type, str(param_dim)))

        
