"""
    IMPORTING LIBS
"""
import torch 
import argparse, json

import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import networkx as nx
import os
import scipy.io
import shutil
import matplotlib.pyplot as plt
import random

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from torch_geometric.utils import to_networkx, to_dense_adj
from tqdm import tqdm
from copy import copy


"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from data import *
from spectrumGCN import *
from train import *
from utils import *


"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def main():    
    """
        USER CONTROLS
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/GCN_Cora.json' ,help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--script_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--use_cache', help="Please give a value for dataset name")
    parser.add_argument('--data_dir', help="Please give a value for data_dir")
    parser.add_argument('--plt_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--trainVal_percent_perClass', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--train_percent_perClass', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--train_num_perClass', help="Please give a value for min_lr")
    parser.add_argument('--val_num_perClass', help="Please give a value for weight_decay")
    parser.add_argument('--train_percent_allClasses', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--val_percent_allClasses', help="Please give a value for L")
    parser.add_argument('--train_num_allClasses', help="Please give a value for hidden_dim")
    parser.add_argument('--val_num_allClasses', help="Please give a value for out_dim")
    parser.add_argument('--mask_type', help="Please give a value for residual")
    parser.add_argument('--graph_less', help="Please give a value for edge_feat")
    parser.add_argument('--data_verobse', help="Please give a value for readout")
    parser.add_argument('--train_verbose', help="Please give a value for in_feat_dropout")
    parser.add_argument('--train_keep', help="Please give a value for dropout")
    parser.add_argument('--test_keep', help="Please give a value for layer_norm")
    parser.add_argument('--test_verbose', help="Please give a value for batch_norm")
    parser.add_argument('--plt_sh', help="Please give a value for max_time")
    parser.add_argument('--plt_keep', help="Please give a value for pos_enc_dim")
    parser.add_argument('--maskInd', help="Please give a value for pos_enc")
    parser.add_argument('--imagetype', help="Please give a value for alpha_loss")
    parser.add_argument('--num_linear', help="Please give a value for lambda_loss")
    parser.add_argument('--add_relu', help="Please give a value for pe_init")
    parser.add_argument('--conv1_out', help="Please give a value for sign inv net")
    parser.add_argument('--conv_bias', help="Please give a value for sign inv layers")
    parser.add_argument('--p_dropout', help="Please give a value for sign inv activation function")
    args = parser.parse_args()
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    with open(f'{ROOT_DIR}/{args.config}') as f:
        config = json.load(f)
    print(config['params'])
    print(config['net_params'])
    print('CONFIG FILE', args.config)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.script_id is not None:
        script_id = args.script_id
    else:
        script_id = config['script_id']
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = config['data_dir']
    if args.plt_dir is not None:
        plt_dir = args.plt_dir
    else:
        plt_dir = config['plt_dir']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    if args.use_cache is not None:
        use_cache = args.use_cache
    else:
        use_cache = config['use_cache']

    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    

    # parameters 
    params = config['params']
    # network parameters
    net_params = config['net_params']

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
        torch.cuda.manual_seed_all(params['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ######################################################################################################################### results df
    epochResults = epochPerformanceDF()  # detailed of each epoch for train and validation set, both accuracy and loss
    summaryResults = TrainValidationTestDF() # summary of trained model for train, validation, and test, both accuracy and loss
    modelResults = {} # final model
    optimizerResults = {} # final optimizer
    graphs = {}
    final = {}

    graph = data_prepare(DATASET_NAME, maskInd=params['maskInd'], data_dir=data_dir)

    graph = spectral_embedding(data=graph,dataset_name = DATASET_NAME, ncol=2*graph.num_classes, drp_first=True, use_cache = use_cache)
    graph = trainValidationTest_splitPerClass(data=graph, trainVal_percent=params['trainVal_percent_perClass'],
                                                      train_percent=params['train_percent_perClass'],
                                                      train_num=params['train_num_perClass'], val_num=params['val_num_perClass'],
                                                      verbose=params['data_verobse'])

    graph = trainValidationTest_splitAllClasses(graph, params['train_percent_allClasses'], params['train_num_allClasses'],
                                                    params['val_percent_allClasses'], params['val_num_allClasses'],
                                                      verbose=params['data_verobse'])
    
    ##################################################### model: spectrumGCN
    gt = time.time()
    torchStatus()
    torch.cuda.seed_all()
    print("\n")

    ##################################################### initialization
    temp = "WithOutGraph" if params['graph_less'] else 'WithGraph'
    mdl_name = "spectrumGCN" + '+' + str(graph.graph_name) + '+' + str(temp)

    print("graph is \n", graph.graph_name)
    graph = graph.to(device)

    st = time.time()
    mdl = spectrumGCN(graph_less=params['graph_less'],
                 spec_in=2*graph.num_classes, spec_out=graph.num_classes,
                 conv1_in_dim=graph.num_node_features, conv1_out_dim=net_params['conv1_out'],
                 conv2_out_dim=graph.num_classes,
                 num_linear=net_params['num_linear'], add_relu=net_params['add_relu'],
                 conv_bias=net_params['conv_bias'],
                 pdrop=net_params['dropout']).to(device)


    opt = torch.optim.Adam(mdl.parameters(),
                           lr=params['lr'])

    print("\n")
    ##################################################### training phase
    print("entering training phase...\n")
    mdl, opt, epochDF = train(model=mdl, model_name=mdl_name, optimizer=opt, mask_type=params['mask_type'],
                                num_epoch=params['epochs'], data=graph, keepResult=params['train_keep'], verbose=params['train_verbose'])
    t_train = time.time() - st
    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, graph_mode=params['graph_less'], plt_dir=plt_dir,
               col="loss", keep=params['plt_keep'], sh=params['plt_sh'], imagetype=params['imagetype'])
    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, graph_mode=params['graph_less'], plt_dir=plt_dir,
               col="accuracy", keep=params['plt_keep'], sh=params['plt_sh'], imagetype=params['imagetype'])

    epochResults = pd.concat([epochResults, epochDF], ignore_index=True)

    print("\n")

    ########################################## end of training phase

    ##################################################### test phase
    print("entring test phase...\n")
    st = time.time()

    mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type=params['mask_type'], keepResult=params['test_keep'],
                      verbose=params['test_verbose'])
    t_test = time.time() - st
    summaryResults = pd.concat([summaryResults, sumDF], ignore_index=True)

    ft = time.time() - gt
    ##################################################### saving results
    print("saving results...")
    res = {'model_name':mdl_name, 'model':mdl, 'optimizer':opt,
           'epochResults':epochResults, 'summaryResults':summaryResults,
           "t_train":t_train, "t_test":t_test, "t_all":gt}
    #final_result = {str(mdl_name):res}

    ##########################################################################################  save the result

    # try:
    #     final = pickle.load(open("dic.pickle", "rb"))
    # except (OSError, IOError) as e:
    #     final = {}

    final[str(graph.graph_name) + str(script_id)] = res
    #pickle.dump(final, open("dic.pickle", "wb"))


    # try:
    #     graphs = pickle.load(open("graphs.pickle", "rb"))
    # except (OSError, IOError) as e:
    #     graphs = {}
    #
    # graphs[str(graph.graph_name) + str(script_number)] = graph
    # pickle.dump(graphs, open("graphs.pickle", "wb"))

    del graph

    print("\n\n\n\n")


main()  