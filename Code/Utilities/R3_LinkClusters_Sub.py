#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-GNN package
#Made by Filips Fedotovs
#Current version 1.0

########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
from pandas import DataFrame as df
import math #We use it for data manipulation
import os, psutil #helps to monitor the memory
import gc  #Helps to clear memory
import numpy as np
import pickle
import torch
import ast
import torch_geometric
from torch_geometric.nn import GCNConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--Z_ID',help="Enter Z id", default='0')
parser.add_argument('--X_ID',help="Please enter the cluster set", default='1')
parser.add_argument('--Y_ID',help="Please enter the cluster set", default='1')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--Log',help="Logging yes?", default='N')
parser.add_argument('--stepX',help="Enter X step size", default='0')
parser.add_argument('--stepY',help="Enter Y step size", default='0')
parser.add_argument('--stepZ',help="Enter Z step size", default='0')
parser.add_argument('--cut_dt',help="Cut on angle difference", default='1.0')
parser.add_argument('--cut_dr',help="Cut on angle difference", default='4000')
parser.add_argument('--zOffset',help="Data offset on z", default='0.0')
parser.add_argument('--yOffset',help="Data offset on y", default='0.0')
parser.add_argument('--xOffset',help="Data offset on x", default='0.0')
parser.add_argument('--ModelName',help="Name of the model", default='')
parser.add_argument('--DNA',help="Please enter the model dna", default='[[4, 4, 1, 2, 2, 2, 2], [5, 4, 1, 1, 2, 2, 2], [5, 4, 2, 1, 2, 2, 2], [5, 4, 2, 1, 2, 2, 2], [], [3, 4, 2], [3, 4, 2], [2, 4, 2], [], [], [7, 1, 1, 4]]')
######################################## Set variables  #############################################################
args = parser.parse_args()
DNA=ast.literal_eval(args.DNA)
Z_ID=int(args.Z_ID)
X_ID=int(args.X_ID)
Y_ID=int(args.Y_ID)
stepX=float(args.stepX) #The coordinate of the st plate in the current scope
stepZ=float(args.stepZ)
stepY=float(args.stepY)
z_offset=float(args.zOffset)
y_offset=float(args.yOffset)
x_offset=float(args.xOffset)
cut_dt=float(args.cut_dt)
cut_dr=float(args.cut_dr)
HiddenLayerDNA=[x for x in DNA[:5] if x != []]
OutputDNA=[x for x in DNA[10:] if x != []]
#Loading Directory locations
EOS_DIR=args.EOS
AFS_DIR=args.AFS

#import sys
#sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF #This is where we keep routine utility functions

#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-GNN/Data/REC_SET/R2_R2_SelectedClusters_'+str(Z_ID)+'_'+str(X_ID)+'_'+str(Y_ID)+'.pkl'

print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
data_file=open(input_file_location,'rb')
RawClusters=pickle.load(data_file)
data_file.close()
if RawClusters[0].ClusterSize<2 and args.Log=='Y':
    label=['Initial # of combinations','Delete self-permutations','Enforce positive directionality','Cut on delta t', 'Cut on delta x','Track Reconstruction']
    array1=[0,0,0,0,0,0]
    array2=[0,0,0,0,0,0]
    RawClusters[0].RecStats=[label,array1,array2]
    LoadedClusters=[]
    LoadedClusters.append(RawClusters[0])
    output_file_location=EOS_DIR+'/EDER-GNN/Data/REC_SET/R3_R4_LinkedClusters_'+str(Z_ID)+'_'+str(X_ID)+'_'+str(Y_ID)+'.pkl'
    open_file = open(output_file_location, "wb")
    pickle.dump(LoadedClusters, open_file)
    print(UF.TimeStamp(), "Cluster linking is finished...")
else:
    print(UF.TimeStamp(),'Loading the model... ')

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
                             output_matrix=(prob_adj > 0).nonzero(as_tuple=False).t().tolist()
                             strength_matrix=[]
                             for i in range(len(output_matrix[0])):
                                element=prob_adj[output_matrix[0][i]][output_matrix[1][i]].item()
                                strength_matrix.append(element)
                             output_matrix.append(strength_matrix)
                             return output_matrix # get predicted edge_list
    model = Net().to(device)
    model.load_state_dict(torch.load(EOS_DIR+'/EDER-GNN/Models/'+args.ModelName))
    model.eval()
    if args.Log=='Y':
        input_file_location=EOS_DIR+'/EDER-GNN/Data/TEST_SET/E1_HITS.csv'
        MCdata=pd.read_csv(input_file_location,header=0,
                                usecols=["Hit_ID","x","y","z","tx","ty",'MC_Mother_Track_ID'])
        MCdata["x"] = pd.to_numeric(MCdata["x"],downcast='float')
        MCdata["y"] = pd.to_numeric(MCdata["y"],downcast='float')
        MCdata["z"] = pd.to_numeric(MCdata["z"],downcast='float')
        MCdata["Hit_ID"] = MCdata["Hit_ID"].astype(str)
        MCdata['z']=MCdata['z']-z_offset
        MCdata['x']=MCdata['x']-x_offset
        MCdata['y']=MCdata['y']-y_offset
        MCdata.drop(MCdata.index[MCdata['z'] >= ((Z_ID+1)*stepZ)], inplace = True)  #Keeping the relevant z slice
        MCdata.drop(MCdata.index[MCdata['z'] < (Z_ID*stepZ)], inplace = True)  #Keeping the relevant z slice
        MCdata.drop(MCdata.index[MCdata['x'] >= ((X_ID+1)*stepX)], inplace = True)  #Keeping the relevant z slice
        MCdata.drop(MCdata.index[MCdata['x'] < (X_ID*stepX)], inplace = True)  #Keeping the relevant z slice
        MCdata.drop(MCdata.index[MCdata['y'] >= ((Y_ID+1)*stepY)], inplace = True)  #Keeping the relevant z slice
        MCdata.drop(MCdata.index[MCdata['y'] < (Y_ID*stepY)], inplace = True)  #Keeping the relevant z slice
        MCdata_list=MCdata.values.tolist()

    LoadedClusters=[]
    data=RawClusters[0].ClusterGraph
    top=[]
    bottom=[]
    for i in range(RawClusters[0].ClusterSize):
                top.append(i)
                bottom.append(i)
    data.train_pos_edge_index=torch.tensor(np.array([top,bottom]))
    lat_z = model.encode(data)
    if args.Log=='Y':
                RawClusters[0].LinkHits(model.decode_all(lat_z),True,MCdata_list,cut_dt,cut_dr)
    LoadedClusters.append(RawClusters[0])
    output_file_location=EOS_DIR+'/EDER-GNN/Data/REC_SET/R3_R4_LinkedClusters_'+str(Z_ID)+'_'+str(X_ID)+'_'+str(Y_ID)+'.pkl'
    open_file = open(output_file_location, "wb")
    pickle.dump(LoadedClusters, open_file)
    print(UF.TimeStamp(), "Cluster linking is finished...")
    #End of the script



