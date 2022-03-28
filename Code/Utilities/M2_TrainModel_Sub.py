########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################




########################################    Import libraries    ########################################################
import csv
import argparse
import math
import ast
import numpy as np
import logging
import os
import copy
import pickle
import Utility_Functions
from Utility_Functions import HitCluster
import torch
# import torch_geometric
# from torch_geometric.utils import train_test_split_edges

# import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
########################## Visual Formatting #################################################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

########################## Setting the parser ################################################
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--Mode',help="Please enter the mode: Create/Test/Train", default='Test')
parser.add_argument('--ClusterSet',help="Please enter the image set", default='1')
parser.add_argument('--DNA',help="Please enter the model dna", default='[[4, 4, 1, 2, 2, 2, 2], [5, 4, 1, 1, 2, 2, 2], [5, 4, 2, 1, 2, 2, 2], [5, 4, 2, 1, 2, 2, 2], [], [3, 4, 2], [3, 4, 2], [2, 4, 2], [], [], [7, 1, 1, 4]]')
parser.add_argument('--AFS',help="Please enter the user afs directory", default='.')
parser.add_argument('--EOS',help="Please enter the user eos directory", default='.')
parser.add_argument('--LR',help="Please enter the value of learning rate", default='Default')
parser.add_argument('--Epoch',help="Please enter the number of epochs per cluster", default='10')
parser.add_argument('--ModelName',help="Name of the model", default='1T_MC_1_model')
parser.add_argument('--ModelNewName',help="Name of the model", default='1T_MC_1_model')
parser.add_argument('--f',help="Image set location (for test)", default='')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
ClusterSet=args.ClusterSet
Mode=args.Mode
Epoch=int(args.Epoch)
DNA=ast.literal_eval(args.DNA)
HiddenLayerDNA=[]
FullyConnectedDNA=[]
OutputDNA=[]
for gene in DNA:
    if DNA.index(gene)<=4 and len(gene)>0:
        HiddenLayerDNA.append(gene)
    elif DNA.index(gene)<=9 and len(gene)>0:
        FullyConnectedDNA.append(gene)
    elif DNA.index(gene)>9 and len(gene)>0:
        OutputDNA.append(gene)

act_fun_list=['N/A','linear','exponential','elu','relu', 'selu','sigmoid','softmax','softplus','softsign','tanh']
ValidModel=True
# Accuracy=0.0
# Accuracy0=0.0
##################################   Loading Directory locations   ##################################################
AFS_DIR=args.AFS
EOS_DIR=args.EOS
import sys
sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF
#Load data configuration
EOSsubDIR=EOS_DIR+'/'+'EDER-GNN'
EOSsubModelDIR=EOSsubDIR+'/'+'Models'
flocation=EOS_DIR+'/EDER-GNN/Data/TRAIN_SET/M1_M2_SelectedTrainClusters_'+ClusterSet+'.pkl'

##############################################################################################################################
######################################### Starting the program ################################################################
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising     EDER-GNN   model creation module   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)

# #Calculate number of batches used for this job
# TrainBatchSize=(OutputDNA[0][1]*4)

def get_link_labels(pos_edge_index, neg_edge_index):
     # returns a tensor:
     # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
     # and the number of zeros is equal to the length of neg_edge_index
     E = pos_edge_index.size(1) + neg_edge_index.size(1)
     link_labels = toGCNConvrch.zeros(E, dtype=torch.float, device=device)
     link_labels[:pos_edge_index.size(1)] = 1.
     return link_labels
#
#
def train(sample):
     model.train()

     neg_edge_index = negative_sampling(
         edge_index=sample.train_pos_edge_index, #positive edges
         num_nodes=sample.num_nodes, # number of nodes
         num_neg_samples=sample.train_pos_edge_index.size(1)) # number of neg_sample equal to number of pos_edges
#
     optimizer.zero_grad()
#
     z = model.encode() #encode
     link_logits = model.decode(z, sample.train_pos_edge_index, neg_edge_index) # decode

     link_labels = get_link_labels(sample.train_pos_edge_index, neg_edge_index)
     loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
     loss.backward()
     optimizer.step()

     return loss
#
#
@torch.no_grad()
def test(sample):
     model.eval()
     perfs = []
     for prefix in ["val", "test"]:
         pos_edge_index = sample[f'{prefix}_pos_edge_index']
         neg_edge_index = sample[f'{prefix}_neg_edge_index']

         z = model.encode() # encode train
         link_logits = model.decode(z, pos_edge_index, neg_edge_index) # decode test or val
         link_probs = link_logits.sigmoid() # apply sigmoid

         link_labels = get_link_labels(pos_edge_index, neg_edge_index) # get link

         perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu())) #compute roc_auc score
     return perfs
if Mode!='Test':
    print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+flocation+bcolors.ENDC)
    train_file=open(flocation,'rb')
    TrainClusters=pickle.load(train_file)
    train_file.close()

#
print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been loaded successfully..."+bcolors.ENDC)
#
# NValBatches=math.ceil(float(len(ValImages))/float(TrainBatchSize))
#
# print(UF.TimeStamp(),'Loading the model...')
# ##### This but has to be converted to a part that interprets DNA code  ###################################
# if args.LR=='Default':
#   LR=10**(-int(OutputDNA[0][3]))
#   opt = adam(learning_rate=10**(-int(OutputDNA[0][3])))
# else:
#     LR=float(args.LR)
#     opt = adam(learning_rate=float(args.LR))
#if Mode=='Train':
            # model_name=EOSsubModelDIR+'/'+args.ModelName
            # model=tf.keras.models.load_model(model_name)
            # K.set_value(model.optimizer.learning_rate, LR)
            # model.summary()
            # print(model.optimizer.get_config())
if Mode!='Train' and Mode!='Test':
#            #try:
#              model = Sequential()
#              for HL in HiddenLayerDNA:
#                  Nodes=HL[0]*16
#                  KS=(HL[2]*2)+1
#                  PS=HL[3]
#                  DR=float(HL[6]-1)/10.0
#                  if HiddenLayerDNA.index(HL)==0:
#                     model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS,KS,KS),kernel_initializer='he_uniform', input_shape=(TrainImages[0].H,TrainImages[0].W,TrainImages[0].L,1)))
#                  else:
#                     model.add(Conv3D(Nodes, activation=act_fun_list[HL[1]],kernel_size=(KS,KS,KS),kernel_initializer='he_uniform'))
#                  if PS>1:
#                     model.add(MaxPooling3D(pool_size=(PS, PS, PS)))
#                  model.add(BatchNormalization(center=HL[4]>1, scale=HL[5]>1))
#                  model.add(Dropout(DR))
#              model.add(Flatten())
#              for FC in FullyConnectedDNA:
#                      Nodes=4**FC[0]
#                      DR=float(FC[2]-1)/10.0
#                      model.add(Dense(Nodes, activation=act_fun_list[FC[1]], kernel_initializer='he_uniform'))
#                      model.add(Dropout(DR))
#              model.add(Dense(2, activation=act_fun_list[OutputDNA[0][0]]))
               class Net(torch.nn.Module):
                    def __init__(self,NoF):
                        super(Net, self).__init__()
                        self.conv1 = GCNConv(NoF, 128)
                        self.conv2 = GCNConv(128, 64)

                    def encode(self,sample):
                         x = self.conv1(sample.x, sample.train_pos_edge_index) # convolution 1
                         x = x.relu()
                         return self.conv2(x, sample.train_pos_edge_index) # convolution 2

                    def decode(self, z, pos_edge_index, neg_edge_index): # only pos and neg edges
                         edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) # concatenate pos and neg edges
                         logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product
                         return logits

                    def decode_all(self, z):
                         prob_adj = z @ z.t() # get adj NxN
                         return (prob_adj > 0).nonzero(as_tuple=False).t() # get predicted edge_list
# Compile the model
               model = Net(5).to(device)
               optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

#            #except:
#            #   print(UF.TimeStamp(), bcolors.FAIL+"Invalid model, aborting the training..."+bcolors.ENDC)
#            #   ValidModel=False
#             #  exit()
# if Mode=='Test':
#            model_name=EOSsubModelDIR+'/'+args.ModelName
#            model=tf.keras.models.load_model(model_name)
#            K.set_value(model.optimizer.learning_rate, LR)
#            model.summary()
#            print(model.optimizer.get_config())
#            for ib in range(0,NValBatches):
#               StartSeed=(ib*TrainBatchSize)+1
#               EndSeed=StartSeed+TrainBatchSize-1
#               BatchImages=UF.LoadRenderImages(ValImages,StartSeed,EndSeed)
#               a=model.test_on_batch(BatchImages[0], BatchImages[1], reset_metrics=False)
#               val_loss=a[0]
#               val_acc=a[1]
#               progress=int(round((float(ib)/float(NValBatches))*100,0))
#               print("Validation in progress ",progress,' %',"Validation loss is:",val_loss,"Validation accuracy is:",val_acc , end="\r", flush=True)
#            print('Test is finished')
#            print("Final Validation loss is:",val_loss)
#            print("Final Validation accuracy is:",val_acc)
#            exit()
# records=[]
print(UF.TimeStamp(),'Starting the training process... ')
for tc in range(0,len(TrainClusters)):
    sample=TrainClusters[tc].ClusterGraph
    sample = sample.to(device)
    best_val_perf = test_perf = 0
    for epoch in range(1, Epoch):
     train_loss = train(sample)
     val_perf, tmp_test_perf = test(sample)
     if val_perf > best_val_perf:
         best_val_perf = val_perf
         test_perf = tmp_test_perf
     log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
     if epoch % 10 == 0:
         print(log.format(ClusterSet,tc,epoch, train_loss, best_val_perf, test_perf))
#     StartSeed=(ib*TrainBatchSize)+1
#     EndSeed=StartSeed+TrainBatchSize-1
#     BatchImages=UF.LoadRenderImages(TrainImages,StartSeed,EndSeed)
#     model.train_on_batch(BatchImages[0],BatchImages[1])
#     progress=int(round((float(ib)/float(NTrainBatches))*100,0))
#     print("Training in progress ",progress,' %', end="\r", flush=True)
# print(UF.TimeStamp(),'Finished with the training... ')
# print(UF.TimeStamp(),'Evaluating this epoch ')
# model.reset_metrics()
# for ib in range(0,NTrainBatches):
#     StartSeed=(ib*TrainBatchSize)+1
#     EndSeed=StartSeed+TrainBatchSize-1
#     BatchImages=UF.LoadRenderImages(TrainImages,StartSeed,EndSeed)
#     t=model.test_on_batch(BatchImages[0], BatchImages[1], reset_metrics=False)
#     train_loss=t[0]
#     train_acc=t[1]
# model.reset_metrics()
# for ib in range(0,NValBatches):
#     StartSeed=(ib*TrainBatchSize)+1
#     EndSeed=StartSeed+TrainBatchSize-1
#     BatchImages=UF.LoadRenderImages(ValImages,StartSeed,EndSeed)
#     a=model.test_on_batch(BatchImages[0], BatchImages[1], reset_metrics=False)
#     val_loss=a[0]
#     val_acc=a[1]
# if ValidModel:
#     model_name=EOSsubModelDIR+'/'+args.ModelNewName
#     model.save(model_name)
#     records.append([int(args.Epoch),ImageSet,len(TrainImages),train_loss,train_acc,val_loss,val_acc])
#     UF.LogOperations(EOSsubModelDIR+'/'+'M5_M5_model_train_log_'+ImageSet+'.csv','StartLog', records)

