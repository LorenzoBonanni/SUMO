import argparse
import os
import ast
import random

import torch
import numpy as np
import importlib
from sklearn.neighbors import KDTree
import faiss

device = None
logger = None


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_device_and_logger(gpu_id, logger_ent):
    global device, logger
    if gpu_id < 0 or torch.cuda.is_available() == False:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(gpu_id))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print("setting device:", device)
    logger = logger_ent


def relative_path_to_module_path(relative_path):
    path = relative_path.replace(".py", "").replace(os.path.sep,'.')
    return path


def load_config(config_path, update_args):
    default_config_path_elements = config_path.split(os.sep)
    default_config_path_elements[-1] = "default.py"
    default_config_path = os.path.join(*default_config_path_elements)
    default_args_module = importlib.import_module(relative_path_to_module_path(default_config_path))
    overwrite_args_module = importlib.import_module(relative_path_to_module_path(config_path))
    default_args_dict = getattr(default_args_module, 'default_args')
    args_dict = getattr(overwrite_args_module, 'overwrite_args')
    assert type(default_args_dict) == dict, "default args file should be default_args=\{...\}"
    assert type(args_dict) == dict, "args file should be default_args=\{...\}"

    #update args is tpule type, convert to dict type
    update_args_dict = {}
    for update_arg in update_args:
        key, val = update_arg.split("=")
        update_args_dict[key] = ast.literal_eval(val)
    
    #update env specific args to default 
    args_dict = merge_dict(default_args_dict, args_dict)
    default_args_dict = update_parameters(default_args_dict, update_args_dict)
    if 'common' in args_dict:
        for sub_key in args_dict:
            if type(args_dict[sub_key]) == dict:
                args_dict[sub_key] = merge_dict(args_dict[sub_key], default_args_dict['common'], "common")
    return args_dict


def merge_dict(source_dict, update_dict, ignored_dict_name=""):
    for key in update_dict:
        if key == ignored_dict_name:
            continue
        if key not in source_dict:
            #print("\033[32m new arg {}: {}\033[0m".format(key, update_dict[key]))
            source_dict[key] = update_dict[key]
        else:
            if type(update_dict[key]) == dict:
                source_dict[key] = merge_dict(source_dict[key], update_dict[key], ignored_dict_name)
            else:
                print("updated {} from {} to {}".format(key, source_dict[key], update_dict[key]))
                source_dict[key] = update_dict[key]
    return source_dict


def update_parameters(source_args, update_args):
    print("updating args", update_args)
    #command line overwriting case, decompose the path and overwrite the args
    for key_path in update_args:
        target_value = update_args[key_path]
        print("key:{}\tvalue:{}".format(key_path, target_value))
        source_args = overwrite_argument_from_path(source_args, key_path, target_value)
    return source_args


def overwrite_argument_from_path(source_dict, key_path, target_value):
    key_path = key_path.split("/")
    curr_dict = source_dict
    for key in key_path[:-1]:
        if not key in curr_dict:
            #illegal path
            return source_dict
        curr_dict = curr_dict[key]
    final_key = key_path[-1] 
    curr_dict[final_key] = target_value
    return source_dict


def second_to_time_str(remaining:int):
    dividers = [86400, 3600, 60, 1]
    names = ['day', 'hour', 'minute', 'second']
    results = []
    for d in dividers:
        re = int(np.floor(remaining / d))
        results.append(re)
        remaining -= re * d
    time_str = ""
    for re, name in zip(results, names):
        if re > 0 :
            time_str += "{} {}  ".format(re, name)
    return time_str


class KD_tree:
    def __init__(self, data, k=1):
        assert isinstance(data, np.ndarray)
        self.data = data
        self.kd_tree = KDTree(self.data)
        self.k = k

    def query(self,query_points):
        assert isinstance(query_points, np.ndarray)
        assert len(query_points.shape) == 2
        distances, _ = self.kd_tree.query(query_points, k=self.k, return_distance=True)
        
        return distances[:,-1][:,None] #(batch,1)
    
class Faiss:
    def __init__(self, data, k=1):
        assert isinstance(data, np.ndarray)
        self.data = data
        self.dim = data.shape[-1]
        self.k = k

        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(self.data)

    def query(self, query_points):
        D, I = self.index.search(query_points, self.k)    
        #D:[num,k]
        return D[:,-1][:,None] 


class Scaler:
    # class variable
    scale_value = -1
    r_max = 1

    @classmethod
    def __call__(cls, penalty):
        assert isinstance(penalty, np.ndarray)
        if abs(np.max(penalty)) > cls.scale_value:
            cls.scale_value = abs(np.max(penalty))
        
        return penalty / cls.scale_value * cls.r_max

def calc_uncertainty_score_genShen(means_first_gaussian: np.ndarray, vars_first_gaussian: np.ndarray,
                                      means_second_gaussian: np.ndarray, vars_second_gaussian: np.ndarray) -> np.ndarray:
    al = 0.5

    t1 = (1 - al) * means_first_gaussian / vars_first_gaussian * means_first_gaussian
    t1S = np.sum(t1, axis=1)

    t2 = al * means_second_gaussian / vars_second_gaussian * means_second_gaussian
    t2S = np.sum(t2, axis=1)

    sigAL = 1 / ((1 - al) / vars_first_gaussian + al / vars_second_gaussian)
    muAl = sigAL * ((1 - al) / vars_first_gaussian * means_first_gaussian + al / vars_second_gaussian * means_second_gaussian)

    t3 = muAl / sigAL * muAl
    t3S = np.sum(t3, axis=1)

    log_det_S1 = np.sum(np.log(vars_first_gaussian), axis=1)
    log_det_S2 = np.sum(np.log(vars_second_gaussian), axis=1)
    log_det_SSum = np.sum(np.log(sigAL), axis=1)

    log_term = (1 - al) * log_det_S1 + al * log_det_S2 - log_det_SSum

    shanon = 0.5 * (t1S + t2S - t3S + log_term)
    return shanon

def calc_pairwise_symmetric_uncertainty_for_measure_function(means_of_all_ensembles: np.ndarray,
                                                                vars_of_all_ensembles: np.ndarray,
                                                                ensemble_size: int, measure_func):
    counter_u = 1
    sum_uncertainty = measure_func(means_of_all_ensembles[0],
                                   vars_of_all_ensembles[0],
                                   means_of_all_ensembles[1],
                                   vars_of_all_ensembles[1])

    for j in range(2, ensemble_size):
        for k in range(j):
            counter_u = counter_u + 1
            sum_uncertainty += measure_func(means_of_all_ensembles[j],
                                            vars_of_all_ensembles[j],
                                            means_of_all_ensembles[k],
                                            vars_of_all_ensembles[k])
    sum_uncertainty = sum_uncertainty / counter_u
    return sum_uncertainty

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="mopo")
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=True)
    parser.add_argument('--target-entropy', type=int, default=-3)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)

    # dynamics model's arguments
    parser.add_argument("--dynamics-lr", type=float, default=0.001)
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=1.0)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)
    parser.add_argument("--uncertainty",type=str, default='faiss', choices=["kd", "faiss",  "max_pair_diff", "max_aleatoric", "gjsd"])

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()