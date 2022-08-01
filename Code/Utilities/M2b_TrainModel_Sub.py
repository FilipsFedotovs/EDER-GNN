########################################################################################################################
#######################################  This simple script prepares data for CNN  #####################################




########################################    Import libraries    ########################################################
import csv
import argparse
import math
import ast
import numpy as np
import random
import logging
import os
import copy
import pickle
import Utility_Functions
from Utility_Functions import HitCluster
import torch
import torch_geometric
from torch import optim
from time import time
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from track_condensation_network import TCN
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
parser.add_argument('--DNA',help="Please enter the model dna", default='[[4, 4, 1, 2, 2, 2, 2], [5, 4, 1, 1, 2, 2, 2], [5, 4, 2, 1, 2, 2, 2], [5, 4, 2, 1, 2, 2, 2], [], [3, 4, 2], [3, 4, 2], [2, 4, 2], [], [], [7, 1, 1, 4]]')
parser.add_argument('--AFS',help="Please enter the user afs directory", default='.')
parser.add_argument('--EOS',help="Please enter the user eos directory", default='.')
parser.add_argument('--LR',help="Please enter the value of learning rate", default='0.01')
parser.add_argument('--Epoch',help="Please enter the number of epochs per cluster", default='10')
parser.add_argument('--ModelName',help="Name of the model", default='1T_MC_1_model')
parser.add_argument('--ModelNewName',help="Name of the model", default='1T_MC_1_model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
########################################     Initialising Variables    #########################################
args = parser.parse_args()
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


##############################################################################################################################
######################################### Starting the program ################################################################
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################  Initialising     EDER-GNN   model creation module   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
def zero_divide(a, b):
    if (b==0): return 0
    return a/b
def binary_classification_stats(output, y, thld):
    TP = torch.sum((y==1) & (output>thld))
    TN = torch.sum((y==0) & (output<thld))
    FP = torch.sum((y==0) & (output>thld))
    FN = torch.sum((y==1) & (output<thld))
    acc = zero_divide(TP+TN, TP+TN+FP+FN)
    TPR = zero_divide(TP, TP+FN)
    TNR = zero_divide(TN, TN+FP)
    return acc, TPR, TNR

def train(model, device, sample, optimizer):
    """ train routine, loss and accumulated gradients used to update
        the model via the ADAM optimizer externally
    """
    model.train()
    losses_w = [] # edge weight loss
    iterator=0
    for HC in sample:

        data = HC.to(device)
        if (len(data.x)==0 or len(data.edge_index)==0): continue

        try:
          iterator+=1
          w = model(data.x, data.edge_index, data.edge_attr)
          y, w = data.y.float(), w.squeeze(1)
        except:
            print('Erroneus data set: ',data.x, data.edge_index, data.edge_attr, 'skipping these samples...')
            continue
        #edge weight loss
        loss_w = F.binary_cross_entropy(w, y, reduction='mean')
        # optimize total loss
        if iterator%4==0:
           optimizer.zero_grad()
           loss_w.backward()
           optimizer.step()

        # store losses
        losses_w.append(loss_w.item())
    loss_w = np.nanmean(losses_w)
    return loss_w,iterator

def validate(model, device, sample):
    model.eval()
    opt_thlds, accs, losses = [], [], []
    for HC in sample[:5]:
        data = HC.to(device)
        if (len(data.x)==0 or len(data.edge_index)==0): continue
        try:
            output = model(data.x, data.edge_index, data.edge_attr)
        except:
            continue

        y, output = data.y.float(), output.squeeze(1)
        try:
          loss = F.binary_cross_entropy(output, y, reduction='mean').item()
        except:
            print('Erroneus data set: ',data.x, data.edge_index, data.edge_attr, 'skipping these samples...')
            continue
        diff, opt_thld, opt_acc = 100, 0, 0
        best_tpr, best_tnr = 0, 0
        for thld in np.arange(0.01, 0.6, 0.01):
            acc, TPR, TNR = binary_classification_stats(output, y, thld)
            delta = abs(TPR-TNR)
            if (delta.item() < diff):
                diff, opt_thld, opt_acc = delta.item(), thld, acc.item()
        opt_thlds.append(opt_thld)
        accs.append(opt_acc)
        losses.append(loss)
    return np.nanmean(opt_thlds),np.nanmean(losses),np.nanmean(accs)

def test(model, device, sample, thld):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for HC in sample:
            data = HC.to(device)
            if (len(data.x)==0 or len(data.edge_index)==0): continue
            try:
               output = model(data.x, data.edge_index, data.edge_attr)
            except:
               continue
            y, output = data.y.float(), output.squeeze(1)
            acc, TPR, TNR = binary_classification_stats(output, y, thld)
            try:
                loss = F.binary_cross_entropy(output, y,reduction='mean')
            except:
                print('Erroneus data set: ',data.x, data.edge_index, data.edge_attr, 'skipping these samples...')
                continue
            accs.append(acc.item())
            losses.append(loss.item())
    return np.nanmean(losses), np.nanmean(accs)
#if Mode!='Test':
DataItrStatus=True
SampleCounter=0
TrainSamples=[]
ValSamples=[]
TestSamples=[]
while DataItrStatus:
    SampleCounter+=1
    try:
        flocation=EOS_DIR+'/EDER-GNN/Data/TRAIN_SET/M1_M2_SelectedTrainClusters_'+str(SampleCounter)+'.pkl'
        print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+flocation+bcolors.ENDC)
        train_file=open(flocation,'rb')
        TrainClusters=pickle.load(train_file)
        TrainFraction=int(math.floor(len(TrainClusters)*0.85))
        ValFraction=int(math.ceil(len(TrainClusters)*0.1))
        for smpl in range(0,TrainFraction):
             TrainSamples.append(TrainClusters[smpl].ClusterGraph)
        for smpl in range(TrainFraction,TrainFraction+ValFraction):
             ValSamples.append(TrainClusters[smpl].ClusterGraph)
        for smpl in range(TrainFraction+ValFraction,len(TrainClusters)):
             TestSamples.append(TrainClusters[smpl].ClusterGraph)
        train_file.close()
    except:
        break
num_nodes_ftr=TrainSamples[0].num_node_features
num_edge_ftr=TrainSamples[0].num_edge_features
print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has loaded and analysed successfully..."+bcolors.ENDC)

def main(self):
    print(UF.TimeStamp(),'Starting the training process... ')
    State_Save_Path=EOSsubModelDIR+'/'+args.ModelNewName+'_State_Save'
    device = torch.device("cpu")
    if Mode!='Train' and Mode!='Test':
        model = TCN(num_nodes_ftr, num_edge_ftr).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=0.1,
                       gamma=0.1)
        StEpoch=1
        FinEpoch=11
    if Mode=='Train':
        model_name=EOSsubModelDIR+'/'+args.ModelName
        model = TCN(num_nodes_ftr, num_edge_ftr).to(device)
        model.load_state_dict(torch.load(model_name))
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=0.1,
                       gamma=0.1)
        checkpoint = torch.load(State_Save_Path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        StEpoch=Epoch
        FinEpoch=Epoch+10
    # instantiate optimizer with scheduled learning rate decay


    records=[]
    for epoch in range(StEpoch, FinEpoch):
        train_loss, itr= train(model, device,TrainSamples, optimizer)
        thld, val_loss,val_acc = validate(model, device, ValSamples)
        test_loss, test_acc = test(model, device,TestSamples, thld)
        scheduler.step()
        print('Epoch ',epoch, ' is completed')
        records.append([epoch,itr,train_loss,thld,val_loss,val_acc,test_loss,test_acc])

    torch.save({    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),    # HERE IS THE CHANGE
                    }, State_Save_Path)
    model_name=EOSsubModelDIR+'/'+args.ModelNewName
    torch.save(model.state_dict(), model_name)
    if Mode=='Create':
       Header=[['Epoch','# Samples','Train Loss','Optimal Threshold','Validation Loss','Validation Accuracy','Test Loss','Test Accuracy']]
       Header+=records
       UF.LogOperations(EOSsubModelDIR+'/'+'Train_Log_'+args.ModelNewName+'.csv','StartLog', Header)
    elif Mode=='Train':
       UF.LogOperations(EOSsubModelDIR+'/'+'Train_Log_'+args.ModelNewName+'.csv','UpdateLog', records)
if __name__ == '__main__':
    main(sys.argv[1:])


