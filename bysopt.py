def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
# load library
import sys
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import hyperopt as hpt
from sklearn.model_selection import RepeatedKFold

from model import model as EfnBoost
from base import survival_dmat

global Logval, eval_cnt, time_start
global train_data
global max_iters, k_fold, n_repeats 
global T_col, E_col

def args_trans(args):
    params = {}
    ##########################################################
    params["eta"] =  args["eta"] * 0.01 + 0.01
    params["nrounds"] =  args["nrounds"] * 10 + 80
    params['max_depth'] = 2 + args["max_depth"]
    params['min_child_weight'] = args['min_child_weight']
    params['subsample'] = args['subsample'] * 0.1 + 0.4
    # params['reg_lambda'] = args['reg_lambda']
    # params['reg_gamma'] = args['reg_gamma']
    ##########################################################
    return params

# def estimate_time():
#     global time_start, eval_cnt
#     time_now = time.clock()
#     total = (time_now - time_start) / eval_cnt * (max_iters - eval_cnt)
#     th = int(total / 3600)
#     tm = int((total - th * 3600) / 60)
#     ts = int(total - th * 3600 - tm * 60)
#     print('Estimate the remaining time: %dh %dm %ds' % (th, tm, ts))

def invoke_xgb(data_train, data_test, params):
    dtrain = survival_dmat(data_train, t_col=T_col, e_col=E_col, label_col="Y")
    dtest = survival_dmat(data_test, t_col=T_col, e_col=E_col, label_col="Y")
    # params
    params_model = {
        'eta': params['eta'],
        'max_depth': params['max_depth'], 
        'min_child_weight': params['min_child_weight'],
        'subsample': params['subsample'],
        # 'colsample_bytree': params['colsample_bytree'],
        # 'lambda': params['reg_lambda'],
        # 'gamma': params['reg_gamma'],
        'verbosity': 1
    }
    # Build and train model
    model = EfnBoost(params_model)
    eval_result = model.train(
        dtrain,
        num_rounds=params['nrounds'],
        plot=False
    )
    # Evaluation
    return model.evals(dtest)

def train_model(args):
    global Logval, eval_cnt, time_start
    global train_data

    params = args_trans(args)
    # Repeated KFold cross validation
    rskf = RepeatedKFold(n_splits=k_fold, n_repeats= n_repeats, random_state=64)
    metrics = []
    for train_index, test_index in rskf.split(train_data):
        data_train = train_data.iloc[train_index, :]
        data_validate = train_data.iloc[test_index,  :]
        metrics.append(invoke_xgb(data_train, data_validate, params))
    metrics_mean = np.array(metrics).mean()
    # Write log
    Logval.append({'params': params, 'ci': metrics_mean})
    eval_cnt += 1
    # Estimate time left
    if eval_cnt % 1 == 0:
        print(params, metrics_mean)
        #estimate_time()
    
    return 1.0 - metrics_mean

def search_params(max_evals=100):
    global Logval
    ###################################################################
    # Parameters' space
    space = {
        "eta": hpt.hp.randint("eta", 10),  # [0.01, 0.10] = 0.01 * [0, 9] + 0.01
        "nrounds": hpt.hp.randint("nrounds", 8),  # [80, 150] = 10 * [0, 7] + 80
        "max_depth": hpt.hp.randint("max_depth", 5), # [2, 6] = [0, 4] + 2
        # "reg_gamma":  hpt.hp.uniform("reg_gamma", 0.0, 1.0), # [0.0, 1.0]
        "min_child_weight": hpt.hp.uniform("min_child_weight", 0.0, 1.0), # [0.0, 1.0]
        "subsample": hpt.hp.randint("subsample", 7), # [0.4, 1.0] = 0.1 * [0, 6] + 0.5
        # "colsample_bytree": hpt.hp.randint("colsample_bytree", 7),# [0.4, 1.0] = 0.1 * [0, 6] + 0.5
        # "reg_lambda" : hpt.hp.uniform("reg_lambda", 0.0, 1.0) # [0.0, 1.0]
    }
    ####################################################################
    # Minimize
    best = hpt.fmin(train_model, space, algo=hpt.tpe.suggest, max_evals=max_evals)
    print("hyper-parameters Searching Finished !")
    print("\tBest params :", args_trans(best))
    print("\tBest metrics:", 1.0 - train_model(best))

def write_file(message, filepath):
    """Write message into the specified file formatted as JSON"""
    # unified format for method `json.dump`
    # convert it back before passing it to models
    for m in message:
        for k in ["nrounds", "max_depth"]:
            m["params"][k] = 1.0 * m["params"][k]
    
    with open(filepath, 'w') as f:
        json.dump(message, f)

if __name__ == "__main__":
    #### Set file name of input and output ###
    train_data = pd.read_csv('train.csv')
    print("No. Rows:", len(train_data))
    print("ID. Cols:", train_data.columns)
    ##########################################
    ###    Initialize global variables     ###
    Logval = []
    eval_cnt = 0
    max_iters = 1
    k_fold = 10
    n_repeats = 1
    T_col = 'T'
    E_col = 'E'
    time_start = time.perf_counter()
    ##########################################
    search_params(max_evals=max_iters)
    #write_file(Logval, output_file)
