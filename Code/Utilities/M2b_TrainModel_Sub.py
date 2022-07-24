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
from time import time
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from edge_classifier_1 import EdgeClassifier
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

def train(args, model, device, sample, optimizer, epoch):
    model.train()
    losses, t0, N = [], time(), len(sample)
    for batch_idx, (data, fname) in enumerate(sample):
        data = data.to(device)
        if (len(data.x)==0): continue
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze(1)
        loss = F.binary_cross_entropy(output, y, reduction='mean')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            percent_complete = 100. * batch_idx / N
            logging.info(f'Train Epoch: {epoch} [{batch_idx}/{N}' +
                         f'({percent_complete:.0f}%)]\tLoss: {loss.item():.6f}')
            if args.dry_run: break
        losses.append(loss.item())
    logging.info(f'Epoch completed in {time()-t0}s')
    logging.info(f'Train Loss: {np.nanmean(losses)}')
    return np.nanmean(losses)
#
#

def validate(model, device, val_loader):
    model.eval()
    opt_thlds, accs = [], []
    for batch_idx, (data, fname) in enumerate(val_loader):
        data = data.to(device)
        if (len(data.x)==0): continue
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze()
        loss = F.binary_cross_entropy(output, y, reduction='mean').item()
        diff, opt_thld, opt_acc = 100, 0, 0
        best_tpr, best_tnr = 0, 0
        for thld in np.arange(0.01, 0.6, 0.01):
            acc, TPR, TNR = binary_classification_stats(output, y, thld)
            delta = abs(TPR-TNR)
            if (delta.item() < diff):
                diff, opt_thld, opt_acc = delta.item(), thld, acc.item()
        opt_thlds.append(opt_thld)
        accs.append(opt_acc)
    logging.info(f'Validation set accuracy (where TPR=TNR): {np.nanmean(accs)}')
    logging.info(f'Validation set optimal edge weight thld: {np.nanmean(opt_thld)}')
    return np.nanmean(opt_thlds)

def test(model, device, test_loader, thld=0.5):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for batch_idx, (data, fname) in enumerate(test_loader):
            data = data.to(device)
            if (len(data.x)==0): continue
            output = model(data.x, data.edge_index, data.edge_attr)
            y, output = data.y, output.squeeze()
            acc, TPR, TNR = binary_classification_stats(output, y, thld)
            loss = F.binary_cross_entropy(output, data.y,
                                          reduction='mean')
            accs.append(acc.item())
            losses.append(loss.item())
    logging.info(f'Test loss: {np.nanmean(losses):.4f}')
    logging.info(f'Test accuracy: {np.nanmean(accs):.4f}')
    return np.nanmean(losses), np.nanmean(accs)
if Mode!='Test':
    print(UF.TimeStamp(),'Loading data from ',bcolors.OKBLUE+flocation+bcolors.ENDC)
    train_file=open(flocation,'rb')
    TrainClusters=pickle.load(train_file)
    train_file.close()


print(UF.TimeStamp(), bcolors.OKGREEN+"Train data has been loaded successfully..."+bcolors.ENDC)

# if Mode=='Train':
#             class Net(torch.nn.Module):
#                     def __init__(self):
#                         super(Net, self).__init__()
#                         for el in range(0,len(HiddenLayerDNA)):
#                             if el==0:
#                                 Nodes=32*HiddenLayerDNA[el][0]
#                                 NoF=OutputDNA[0][0]
#                                 self.conv1 = GCNConv(NoF, Nodes)
#                             if el==1:
#                                 Nodes=32*HiddenLayerDNA[el][0]
#                                 PNodes=32*HiddenLayerDNA[el-1][0]
#                                 self.conv2 = GCNConv(PNodes, Nodes)
#                             if el==2:
#                                 Nodes=32*HiddenLayerDNA[el][0]
#                                 PNodes=32*HiddenLayerDNA[el-1][0]
#                                 self.conv3 = GCNConv(PNodes, Nodes)
#                             if el==3:
#                                 Nodes=32*HiddenLayerDNA[el][0]
#                                 PNodes=32*HiddenLayerDNA[el-1][0]
#                                 self.conv4 = GCNConv(PNodes, Nodes)
#                             if el==4:
#                                 Nodes=32*HiddenLayerDNA[el][0]
#                                 PNodes=32*HiddenLayerDNA[el-1][0]
#                                 self.conv5 = GCNConv(PNodes, Nodes)
#                     def encode(self,sample):
#                          x = self.conv1(sample.x, sample.train_pos_edge_index) # convolution 1
#                          x = x.relu()
#                          return self.conv2(x, sample.train_pos_edge_index) # convolution 2
#
#                     def decode(self, z, pos_edge_index, neg_edge_index): # only pos and neg edges
#                          edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) # concatenate pos and neg edges
#                          logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product
#                          return logits
#
#                     def decode_all(self, z):
#                          prob_adj = z @ z.t() # get adj NxN
#                          return (prob_adj > 0).nonzero(as_tuple=False).t() # get predicted edge_list
#             model_name=EOSsubModelDIR+'/'+args.ModelName
#             model = Net().to(device)
#             optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
#             model.load_state_dict(torch.load(model_name))
# if Mode!='Train' and Mode!='Test':
#                class MLP(nn.Module):
#                     def __init__(self, input_size, output_size, hidden_size):
#                         super(MLP, self).__init__()
#
#                         self.layers = nn.Sequential(
#                             nn.Linear(input_size, hidden_size),
#                             nn.ReLU(),
#                             nn.Linear(hidden_size, hidden_size),
#                             nn.ReLU(),
#                             nn.Linear(hidden_size, hidden_size),
#                             nn.ReLU(),
#                             nn.Linear(hidden_size, output_size),
#                         )
#
#                     def forward(self, C):
#                         return self.layers(C)
#
#                class EdgeClassifier(nn.Module):
#                     def __init__(self, node_indim, edge_indim):
#                         super(EdgeClassifier, self).__init__()
#                         self.IN = InteractionNetwork(node_indim, edge_indim,
#                                      node_outdim=3, edge_outdim=4,
#                                      hidden_size=120)
#                         self.W = MLP(4, 1, 40)
#
#                     def forward(self, x: Tensor, edge_index: Tensor,
#                         edge_attr: Tensor) -> Tensor:
#
#                         x1, edge_attr_1 = self.IN(x, edge_index, edge_attr)
#                         return torch.sigmoid(self.W(edge_attr))
#
# # Compile the model
#                model = Net().to(device)
#                optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

#            #except:
#            #   print(UF.TimeStamp(), bcolors.FAIL+"Invalid model, aborting the training..."+bcolors.ENDC)
#            #   ValidModel=False
#             #  exit()


def main(args):
    print(UF.TimeStamp(),'Starting the training process... ')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    logging.info(f'Parameter use_cuda={use_cuda}')
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 4}

    loaders = get_dataloaders(args.indir, args.n_train, args.n_test,
                              n_val=args.n_val, shuffle=False,
                              params=params)

    model = EdgeClassifier(3, 4).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Trainable params in network: {total_trainable_params}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size,
                       gamma=args.gamma)

    output = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    for epoch in range(1, args.epochs + 1):
        logging.info(f'Entering epoch {epoch}')
        train_loss = train(args, model, device, loaders['train'], optimizer, epoch)
        thld = validate(model, device, loaders['val'])
        logging.info(f'Sending thld={thld} to test routine.')
        test_loss, test_acc = test(model, device, loaders['test'], thld=thld)
        scheduler.step()
        if args.save_model:
            model_name = join(args.model_outdir,
                              job_name + f'_epoch{epoch}')
            torch.save(model.state_dict(), model_name)
        output['train_loss'].append(train_loss)
        output['test_loss'].append(test_loss)
        output['test_acc'].append(test_acc)
        np.save(join(args.stats_outdir, job_name), output)


if __name__ == '__main__':
    main(sys.argv[1:])
model_name=EOSsubModelDIR+'/'+args.ModelNewName
torch.save(model.state_dict(), model_name)
UF.LogOperations(EOSsubModelDIR+'/'+'M2_M2_model_train_log_'+ClusterSet+'.csv','StartLog', records)
