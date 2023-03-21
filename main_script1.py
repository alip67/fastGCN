r"""


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""

globals().clear()
######################################################################################################################## libraries
import pickle

from data import *
from spectrumGCN import *
from train import *
from utils import *

######################################################################################################################### hyper-parameter initialization
torch.manual_seed(123)

#
trainVal_percent_perClass = 0.01
train_percent_perClass = 0.80
train_num_perClass = None
val_num_perClass = None

#
train_percent_allClasses = 0.65
val_percent_allClasses = 0.15
train_num_allClasses = None
val_num_allClasses = None
#

learning_rate = 0.01
num_epoch = 7
#
mask_type = "original" # either "perClass" or "allClasses" or "original"
#
graph_less = True
# spec_in=2*graph.num_classes
# spec_out= graph.num_classes
num_linear = 2
add_relu = True
conv1_out = 128
conv_bias = True
p_dropout = 0.5
#
data_verobse = True
#
train_verbose = True
train_keep = True

test_keep = True
test_verbose = True

plt_sh = False
plt_keep = True
maskInd =0
imagetype="svg"

#data_dir = "/home/n/na396/fastGCN/DataSets/" # directory to store the data
#data_dir = "D:/Niloo's Project/DataSets/"
# data_dir = "Y:/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/DataSets/"
data_dir = "C:\\Users\\ALIP\\Desktop\\fastGCN-main\\fastGCN-main\\datasets"

#plt_dir = "/home/n/na396/fastGCN/Script1/Plots/" # directory to store the plots
#plt_dir = "D:/Niloo's Project/Plots/Script1/"
# plt_dir = "Y:/Root/Study/PhD - All/Contributions/Paper 4 - ICML - GNN/Code/Plots/Script1/"
plt_dir = "C:\\Users\\ALIP\\Desktop\\fastGCN-main\\fastGCN-main\\Plots\\Script1"


script_number = "1"
#########################################################################################################################

#dataset_name = ["Cora", "CiteSeer", "PubMed", "WikiCs", "Arxiv", "Products"] # dataset name
dataset_name = ["PubMed"]

######################################################################################################################### results df
epochResults = epochPerformanceDF()  # detailed of each epoch for train and validation set, both accuracy and loss
summaryResults = TrainValidationTestDF() # summary of trained model for train, validation, and test, both accuracy and loss
modelResults = {} # final model
optimizerResults = {} # final optimizer
graphs = {}
final = {}

######################################################################################################################### data loading and preperation
# graph: a list of prepared graph datasets

for ds in dataset_name:
    graph = data_prepare(dataset_name=ds, maskInd=maskInd, data_dir=data_dir)

    graph = spectral_embedding(data=graph, ncol=2*graph.num_classes, drp_first=True)
    graph = trainValidationTest_splitPerClass(data=graph, trainVal_percent=trainVal_percent_perClass,
                                                      train_percent=train_percent_perClass,
                                                      train_num=train_num_perClass, val_num=val_num_perClass,
                                                      verbose=data_verobse)

    graph = trainValidationTest_splitAllClasses(graph, train_percent_allClasses, train_num_allClasses,
                                                    val_percent_allClasses, val_num_allClasses,
                                                      verbose=data_verobse)

    ##################################################### model: spectrumGCN
    gt = time.time()
    torchStatus()
    torch.cuda.seed_all()
    print("\n")

    ##################################################### initialization
    temp = "WithOutGraph" if graph_less else 'WithGraph'
    mdl_name = "spectrumGCN" + '+' + str(graph.graph_name) + '+' + str(temp)

    print("graph is \n", graph.graph_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)

    st = time.time()
    mdl = spectrumGCN(graph_less=graph_less,
                 spec_in=2*graph.num_classes, spec_out=graph.num_classes,
                 conv1_in_dim=graph.num_node_features, conv1_out_dim=conv1_out,
                 conv2_out_dim=graph.num_classes,
                 num_linear=2, add_relu=True,
                 conv_bias=True,
                 pdrop=p_dropout).to(device)


    opt = torch.optim.Adam(mdl.parameters(),
                           lr=learning_rate)

    print("\n")
    ##################################################### training phase
    print("entering training phase...\n")
    mdl, opt, epochDF = train(model=mdl, model_name=mdl_name, optimizer=opt, mask_type=mask_type,
                                num_epoch=num_epoch, data=graph, keepResult=train_keep, verbose=train_verbose)
    t_train = time.time() - st
    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, graph_mode=graph_less, plt_dir=plt_dir,
               col="loss", keep=plt_keep, sh=plt_sh, imagetype=imagetype)
    plot_epoch(df=epochDF, model_name=mdl_name, data_name=graph.graph_name, graph_mode=graph_less, plt_dir=plt_dir,
               col="accuracy", keep=plt_keep, sh=plt_sh, imagetype=imagetype)

    epochResults = pd.concat([epochResults, epochDF], ignore_index=True)

    print("\n")

    ########################################## end of training phase

    ##################################################### test phase
    print("entring test phase...\n")
    st = time.time()

    mdl, sumDF = test(model=mdl, model_name=mdl_name, data=graph, mask_type=mask_type, keepResult=test_keep,
                      verbose=test_verbose)
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

    final[str(graph.graph_name) + str(script_number)] = res
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

