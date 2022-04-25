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
import torch_geometric
from torch_geometric.nn import GCNConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--set',help="Enter Z id", default='0')
parser.add_argument('--subset',help="Please enter the cluster set", default='1')
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
######################################## Set variables  #############################################################
args = parser.parse_args()

Set=int(args.set)
ClusterSet=int(args.subset)
stepX=float(args.stepX) #The coordinate of the st plate in the current scope
stepZ=float(args.stepZ)
stepY=float(args.stepY)
z_offset=float(args.zOffset)
y_offset=float(args.yOffset)
x_offset=float(args.xOffset)
cut_dt=float(args.cut_dt)
cut_dr=float(args.cut_dr)
#Loading Directory locations
EOS_DIR=args.EOS
AFS_DIR=args.AFS

#import sys
#sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF #This is where we keep routine utility functions

#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-GNN/Data/REC_SET/R2_R2_SelectedClusters_'+str(Set)+'_'+str(ClusterSet)+'.pkl'

print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)
data_file=open(input_file_location,'rb')
RawClusters=pickle.load(data_file)
data_file.close()

print(UF.TimeStamp(),'Loading the model... ')

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
                         output_matrix=(prob_adj > 0).nonzero(as_tuple=False).t().tolist()
                         strength_matrix=[]
                         for i in range(output_matrix[0]):
                            element=prob_adj[output_matrix[0][i]][output_matrix[1][i]].item()
                            strength_matrix.append(element)
                         print(strength_matrix)
                         exit()
                         return (prob_adj > 0).nonzero(as_tuple=False).t() # get predicted edge_list
data=RawClusters[0].ClusterGraph

model = Net(5).to(device)
top=[]
bottom=[]
for i in range(RawClusters[0].ClusterSize):
    top.append(i)
    bottom.append(i)
data.train_pos_edge_index=torch.tensor(np.array([top,bottom]))
model.load_state_dict(torch.load(EOS_DIR+'/EDER-GNN/Models/DefaultModel'))
model.eval()
lat_z = model.encode(data)
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
    MCdata.drop(MCdata.index[MCdata['z'] >= ((Set+1)*stepZ)], inplace = True)  #Keeping the relevant z slice
    MCdata.drop(MCdata.index[MCdata['z'] < (Set*stepZ)], inplace = True)  #Keeping the relevant z slice
    MCdata_list=MCdata.values.tolist()
    RawClusters[0].LinkHits(model.decode_all(lat_z),True,MCdata_list)



print(RawClusters[0].HitLinks)
exit()

# for i in range(0,Xsteps):
#     LoadedClusters=[]
#     progress=round((float(i)/float(Xsteps))*100,2)
#     print(UF.TimeStamp(),"progress is ",progress,' %') #Progress display
#     for j in range(0,Ysteps):
#         HC=UF.HitCluster([i,j,Set],[stepX,stepY,stepZ])
#         HC.LoadClusterHits(data_list)
#         if args.Log=='Y':
#             HC.GiveStats(MCdata_list,cut_dt,cut_dr)
#         LoadedClusters.append(HC)
#     output_file_location=EOS_DIR+'/EDER-GNN/Data/REC_SET/R2_R2_SelectedClusters_'+str(Set)+'_'+str(i)+'.pkl'
#     open_file = open(output_file_location, "wb")
#     pickle.dump(LoadedClusters, open_file)
# print(UF.TimeStamp(), "Cluster generation is finished...")
# #End of the script



