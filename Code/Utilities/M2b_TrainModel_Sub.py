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
parser.add_argument('--ClusterSet',help="Please enter the image set", default='1')
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

# def train(args, model, device, sample, optimizer, epoch):
#     model.train()
#     losses, t0, N = [], time(), len(sample)
#     for HC in sample:
#         data = HC.ClusterGraph.to(device)
#         if (len(data.x)==0): continue
#         optimizer.zero_grad()
#         print(data.x)
#         print(data.edge_index)
#         print(data.edge_attr)
#         output = model(data.x, data.edge_index, data.edge_attr)
#         print(output)
#         exit()
#         y, output = data.y, output.squeeze(1)
#         loss = F.binary_cross_entropy(output, y, reduction='mean')
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             percent_complete = 100. * batch_idx / N
#             logging.info(f'Train Epoch: {epoch} [{batch_idx}/{N}' +
#                          f'({percent_complete:.0f}%)]\tLoss: {loss.item():.6f}')
#             if args.dry_run: break
#         losses.append(loss.item())
#     logging.info(f'Epoch completed in {time()-t0}s')
#     logging.info(f'Train Loss: {np.nanmean(losses)}')
#     return np.nanmean(losses)
#
#

def train(Predict, model, device, sample, optimizer, epoch):
    """ train routine, loss and accumulated gradients used to update
        the model via the ADAM optimizer externally
    """
    model.train()
    epoch_t0 = time()
    losses = []   # total loss
    losses_w = [] # edge weight loss
    losses_c = [] # condensation loss
    losses_b = [] # background loss
    losses_o = [] # object loss
    for HC in sample:
        data = HC.ClusterGraph.to(device)
        optimizer.zero_grad()
        print(data.x)
        print(data.edge_index)
        print(data.edge_attr)
        if Predict:
            w, xc, beta, p = model(data.x, data.edge_index, data.edge_attr)
        else:
            w, xc, beta = model(data.x, data.edge_index, data.edge_attr)


            print(w, xc, beta)
            exit()

    #     y, w = data.y, w.squeeze(1)
    #     particle_id = data.particle_id
    #     track_params = data.track_params
    #
    #     # edge weight loss
    #     loss_w = F.binary_cross_entropy(w, y, reduction='mean')
    #     loss = loss_w
    #
    #     # condensation loss
    #     loss_c = condensation_loss(beta, xc, particle_id,
    #                                device=device, q_min=args.q_min)
    #     loss_c *= args.loss_c_scale
    #
    #     # background loss
    #     loss_b = background_loss(beta, xc, particle_id,
    #                              device=device, q_min=args.q_min,
    #                              sb=args.sb)
    #     loss_b *= args.loss_b_scale
    #
    #     # object loss
    #     if args.predict_track_params:
    #         loss_o = object_loss(p, beta,
    #                              track_params, particle_id,
    #                              device=device)
    #         loss_o *= args.loss_o_scale
    #         loss += loss_o
    #         losses_o.append(loss_o.item())
    #
    #     # optimize total loss
    #     loss += (loss_c + loss_b)
    #     loss.backward()
    #     optimizer.step()
    #
    #     if batch_idx % args.log_interval == 0:
    #         logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
    #                     .format(epoch, batch_idx, len(train_loader.dataset),
    #                             100. * batch_idx / len(train_loader),
    #                             loss.item()))
    #         logging.info(f'...losses: w={loss_w.item()}, c={loss_c.item()}' +
    #                      f'           b={loss_b.item()}, o={loss_o.item()}')
    #
    #     # store losses
    #     losses.append(loss.item())
    #     losses_w.append(loss_w.item())
    #     losses_c.append(loss_c.item())
    #     losses_b.append(loss_b.item())
    #
    # logging.info(f"Epoch {epoch} Time: {(time()-epoch_t0):.4f}s")
    # loss = np.nanmean(losses)
    # loss_w = np.nanmean(losses_w)
    # loss_c = np.nanmean(losses_c)
    # loss_b = np.nanmean(losses_b)
    # logging.info(f"Epoch {epoch} Train Loss: {loss:.6f}")
    # logging.info(f"Epoch {epoch}: Edge Weight Loss: {loss_w:.6f}")
    # logging.info(f"Epoch {epoch}: Condensation Loss: {loss_c:.6f}")
    # logging.info(f"Epoch {epoch}: Background Loss: {loss_b:.6f}")
    # return loss, loss_w, loss_c, loss_b

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
#if Mode!='Test':
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


def main(self):
    print(UF.TimeStamp(),'Starting the training process... ')

    #use_cuda = not args.no_cuda and torch.cuda.is_available()
    #logging.info(f'Parameter use_cuda={use_cuda}')
    #torch.manual_seed(args.seed)
    device = torch.device("cpu")

    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 4}

    model = TCN(5, 4, 2, predict_track_params=False).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total Trainable Params: {total_trainable_params}')

    # instantiate optimizer with scheduled learning rate decay
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=0.1,
                       gamma=0.1)

    # epoch loop
    output = {'train_loss': [], 'test_loss': [], 'test_acc': [],
              'train_loss_w': [], 'train_loss_c': [], 'train_loss_b': [],
              'test_loss_w': [], 'test_loss_c': [], 'test_loss_b': []}

    for epoch in range(1, 2):
        logging.info(f"---- Epoch {epoch} ----")
        train_loss, tlw, tlc, tlb = train(False, model, device,
                                          TrainClusters, optimizer, epoch)
        print(train_loss)
        exit()
        thld = validate(model, device, val_loader)
        test_loss, te_lw, te_lc, te_lb, te_acc = test(args, model, device,
                                                      test_loader, thld=thld)
        scheduler.step()

        # save output
        output['train_loss'].append(train_loss)
        output['train_loss_w'].append(tlw)
        output['train_loss_c'].append(tlc)
        output['train_loss_b'].append(tlb)
        output['test_loss'].append(test_loss)
        output['test_loss_w'].append(te_lw)
        output['test_loss_c'].append(te_lc)
        output['test_loss_b'].append(te_lb)
        output['test_acc'].append(te_acc)

        if (args.save_models):
            model_out = os.path.join(args.outdir,
                                     f"{args.model_outfile}_epoch{epoch}.pt")
            torch.save(model.state_dict(), model_out)

        stat_out = os.path.join(args.outdir,
                                f"{args.stat_outfile}.csv")
        write_output_file(stat_out, args, pd.DataFrame(output))


if __name__ == '__main__':
    main(sys.argv[1:])
model_name=EOSsubModelDIR+'/'+args.ModelNewName
torch.save(model.state_dict(), model_name)
UF.LogOperations(EOSsubModelDIR+'/'+'M2_M2_model_train_log_'+ClusterSet+'.csv','StartLog', records)

