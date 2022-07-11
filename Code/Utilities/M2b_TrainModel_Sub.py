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
import torch_geometric

import torch.nn.functional as F
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
parser.add_argument('--LR',help="Please enter the value of learning rate", default='0.01')
parser.add_argument('--Epoch',help="Please enter the number of epochs per cluster", default='10')
parser.add_argument('--ModelName',help="Name of the model", default='1T_MC_1_model')
parser.add_argument('--ModelNewName',help="Name of the model", default='1T_MC_1_model')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
ClusterSet=args.ClusterSet
Mode=args.Mode
Epoch=int(args.Epoch)
DNA=ast.literal_eval(args.DNA)
LR=float(args.LR)
HiddenLayerDNA=[x for x in DNA[:5] if x != []]

OutputDNA=[x for x in DNA[10:] if x != []]
act_fun_list=['N/A','linear','exponential','elu','relu', 'selu','sigmoid','softmax','softplus','softsign','tanh']
ValidModel=True


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
     E = pos_edge_index.size(1) + neg_edge_index.size(1)
     link_labels = torch.zeros(E, dtype=torch.float, device=device)
     link_labels[:pos_edge_index.size(1)] = 1.
     return link_labels
#
#
def train(sample):
     model.train()
     optimizer.zero_grad()
#
     z = model.encode(sample) #encode
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
         z = model.encode(sample) # encode train
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


print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been loaded successfully..."+bcolors.ENDC)

if Mode=='Train':
            class Net(torch.nn.Module):
                    def __init__(self):
                        super(Net, self).__init__()
                        for el in range(0,len(HiddenLayerDNA)):
                            if el==0:
                                Nodes=32*HiddenLayerDNA[el][0]
                                NoF=OutputDNA[0][0]
                                self.conv1 = GCNConv(NoF, Nodes)
                            if el==1:
                                Nodes=32*HiddenLayerDNA[el][0]
                                PNodes=32*HiddenLayerDNA[el-1][0]
                                self.conv2 = GCNConv(PNodes, Nodes)
                            if el==2:
                                Nodes=32*HiddenLayerDNA[el][0]
                                PNodes=32*HiddenLayerDNA[el-1][0]
                                self.conv3 = GCNConv(PNodes, Nodes)
                            if el==3:
                                Nodes=32*HiddenLayerDNA[el][0]
                                PNodes=32*HiddenLayerDNA[el-1][0]
                                self.conv4 = GCNConv(PNodes, Nodes)
                            if el==4:
                                Nodes=32*HiddenLayerDNA[el][0]
                                PNodes=32*HiddenLayerDNA[el-1][0]
                                self.conv5 = GCNConv(PNodes, Nodes)
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
            model_name=EOSsubModelDIR+'/'+args.ModelName
            model = Net().to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
            model.load_state_dict(torch.load(model_name))
if Mode!='Train' and Mode!='Test':
               class MLP(nn.Module):
                    def __init__(self, input_size, output_size, hidden_size):
                        super(MLP, self).__init__()

                        self.layers = nn.Sequential(
                            nn.Linear(input_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, output_size),
                        )

                    def forward(self, C):
                        return self.layers(C)

               class EdgeClassifier(nn.Module):
                    def __init__(self, node_indim, edge_indim):
                        super(EdgeClassifier, self).__init__()
                        self.IN = InteractionNetwork(node_indim, edge_indim,
                                     node_outdim=3, edge_outdim=4,
                                     hidden_size=120)
                        self.W = MLP(4, 1, 40)

                    def forward(self, x: Tensor, edge_index: Tensor,
                        edge_attr: Tensor) -> Tensor:

                        x1, edge_attr_1 = self.IN(x, edge_index, edge_attr)
                        return torch.sigmoid(self.W(edge_attr))

# Compile the model
               model = Net().to(device)
               optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

#            #except:
#            #   print(UF.TimeStamp(), bcolors.FAIL+"Invalid model, aborting the training..."+bcolors.ENDC)
#            #   ValidModel=False
#             #  exit()

print(UF.TimeStamp(),'Starting the training process... ')
records=[]
for tc in range(0,len(TrainClusters)):
    c_sample=TrainClusters[tc].ClusterGraph
    c_sample.train_mask = c_sample.val_mask = c_sample.test_mask = c_sample.y = None
    c_sample = c_sample.to(device)
    best_val_perf = test_perf = 0
    for epoch in range(0, Epoch):
      try:
          train_loss = train(c_sample)
          val_perf, tmp_test_perf = test(c_sample)
      except:
         print('Failed training...')
         break
      if val_perf > best_val_perf:
             best_val_perf = val_perf
             test_perf = tmp_test_perf
      if epoch % 10 == 0:
                records.append([ClusterSet,tc, TrainClusters[tc].ClusterSize,epoch, train_loss.item(), best_val_perf, test_perf])
                print(ClusterSet,tc, TrainClusters[tc].ClusterSize,epoch, train_loss.item(), best_val_perf, test_perf)

# if ValidModel:
model_name=EOSsubModelDIR+'/'+args.ModelNewName
torch.save(model.state_dict(), model_name)
UF.LogOperations(EOSsubModelDIR+'/'+'M2_M2_model_train_log_'+ClusterSet+'.csv','StartLog', records)

