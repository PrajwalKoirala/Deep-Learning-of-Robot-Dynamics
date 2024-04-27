import urllib.request
from tqdm import tqdm
import os
import h5py
from infos import *
import gym
import numpy as np



def filepath_from_url(dataset_url):
    DATASET_PATH = "./datasets/"
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(DATASET_PATH, dataset_name)
    return dataset_filepath

def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print('Downloading dataset:', dataset_url, 'to', dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath





def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys



def get_dataset(dataset_name):
    dataset_url = DATASET_URLS[dataset_name]
    h5path = download_dataset_from_url(dataset_url)
    dslst = dataset_name.split('-')
    dataset_to_env = {
        "halfcheetah": "HalfCheetah-v3",
        "hopper": "Hopper-v2",
        "walker2d": "Walker2d-v3",
    }
    env_name = dataset_to_env[dslst[0]]
    env = gym.make(env_name)

    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in ['observations', 'actions', 'rewards', 'terminals']:
        assert key in data_dict, 'Dataset is missing key %s' % key
    N_samples = data_dict['observations'].shape[0]
    if env.observation_space.shape is not None:
        assert data_dict['observations'].shape[1:] == env.observation_space.shape, \
            'Observation shape does not match env: %s vs %s' % (
                str(data_dict['observations'].shape[1:]), str(env.observation_space.shape))
    assert data_dict['actions'].shape[1:] == env.action_space.shape, \
        'Action shape does not match env: %s vs %s' % (
            str(data_dict['actions'].shape[1:]), str(env.action_space.shape))
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:, 0]
    assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    env.close()
    env_info = {
        "env_name": env_name,
        "state_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
        "max_action": env.action_space.high[0],
        "max_score": REF_MAX_SCORE[dataset_name],
        "min_score": REF_MIN_SCORE[dataset_name],
        "max_episode_steps": env._max_episode_steps,
        "dataset_name": dataset_name,
    }
    return data_dict, env_info


def get_normalized_reward(dataset_name):
    max_score = REF_MAX_SCORE[dataset_name]
    min_score = REF_MIN_SCORE[dataset_name]
    return lambda r: (r - min_score) / (max_score - min_score)

def undo_normalized_reward(dataset_name):
    max_score = REF_MAX_SCORE[dataset_name]
    min_score = REF_MIN_SCORE[dataset_name]
    return lambda r: r * (max_score - min_score) + min_score

def discount_cumsum(x, gamma=1):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(len(x) - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

def sum_of_rewards(x, gamma=1):
    total_reward = np.zeros_like(x)
    total_reward += np.sum(x)
    return total_reward

