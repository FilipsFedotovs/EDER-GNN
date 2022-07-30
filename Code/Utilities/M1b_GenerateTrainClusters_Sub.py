#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-VIANN package
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
import random

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--set',help="Enter Z id", default='0')
parser.add_argument('--subset',help="Enter X id", default='0')
parser.add_argument('--stepX',help="Enter X step size", default='0')
parser.add_argument('--stepY',help="Enter Y step size", default='0')
parser.add_argument('--stepZ',help="Enter Z step size", default='0')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--zOffset',help="Data offset on z", default='0.0')
parser.add_argument('--yOffset',help="Data offset on y", default='0.0')
parser.add_argument('--xOffset',help="Data offset on x", default='0.0')
parser.add_argument('--valRatio',help="Fraction of validation edges", default='0.1')
parser.add_argument('--testRatio',help="Fraction of test edges", default='0.05')
parser.add_argument('--cut_dt',help="Cut on angle difference", default='1.0')
parser.add_argument('--cut_dr',help="Cut on angle difference", default='4000')

######################################## Set variables  #############################################################
args = parser.parse_args()

Set=int(args.set)
Subset=int(args.subset)
stepX=float(args.stepX) #The coordinate of the st plate in the current scope
stepZ=float(args.stepZ)
stepY=float(args.stepY)
z_offset=float(args.zOffset)
y_offset=float(args.yOffset)
x_offset=float(args.xOffset)
cut_dt=float(args.cut_dt)
cut_dr=float(args.cut_dr)
val_ratio=float(args.valRatio)
test_ratio=float(args.testRatio)
#Loading Directory locations
EOS_DIR=args.EOS
AFS_DIR=args.AFS

#import sys
#sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF #This is where we keep routine utility functions

#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-GNN/Data/REC_SET/R1_HITS.csv'

print(UF.TimeStamp(), "Modules Have been imported successfully...")
print(UF.TimeStamp(),'Loading pre-selected data from ',input_file_location)

data=pd.read_csv(input_file_location,header=0,
            usecols=["Hit_ID","x","y","z","tx","ty"])
data["x"] = pd.to_numeric(data["x"],downcast='float')
data["y"] = pd.to_numeric(data["y"],downcast='float')
data["z"] = pd.to_numeric(data["z"],downcast='float')
data['x']=data['x']-x_offset
data['y']=data['y']-y_offset
data["Hit_ID"] = data["Hit_ID"].astype(str)
data['z']=data['z']-z_offset
x_max=data['x'].max()
y_max=data['y'].max()
print(UF.TimeStamp(),'Creating clusters... ')
data.drop(data.index[data['z'] >= ((Set+1)*stepZ)], inplace = True)  #Keeping the relevant z slice
data.drop(data.index[data['z'] < (Set*stepZ)], inplace = True)  #Keeping the relevant z slice
data.drop(data.index[data['x'] >= ((Subset+1)*stepX)], inplace = True)  #Keeping the relevant z slice
data.drop(data.index[data['x'] < (Subset*stepX)], inplace = True)  #Keeping the relevant z slice
data_list=data.values.tolist()
Ysteps=math.ceil(y_max/stepY)  #Calculating number of cuts



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
MCdata.drop(MCdata.index[MCdata['x'] >= ((Subset+1)*stepX)], inplace = True)  #Keeping the relevant z slice
MCdata.drop(MCdata.index[MCdata['x'] < (Subset*stepX)], inplace = True)  #Keeping the relevant z slice
MCdata_list=MCdata.values.tolist()
LoadedClusters=[]
for j in range(0,Ysteps):
        progress=round((float(j)/float(Ysteps))*100,2)
        print(UF.TimeStamp(),"progress is ",progress,' %') #Progress display
        HC=UF.HitCluster([Subset,j,Set],[stepX,stepY,stepZ])
        HC.LoadClusterHits(data_list)
        GraphStatus = HC.GenerateTrainDatav2(MCdata_list,cut_dt, cut_dr)
        print(GraphStatus)
        if GraphStatus:
            LoadedClusters.append(HC)
print(random.shuffle(LoadedClusters))
output_file_location=EOS_DIR+'/EDER-GNN/Data/TRAIN_SET/M1_M1_SelectedTrainClusters_'+str(Set)+'_' +str(Subset)+'.pkl'
open_file = open(output_file_location, "wb")
pickle.dump(random.shuffle(LoadedClusters), open_file)
print(UF.TimeStamp(), "Cluster generation is finished...")
#End of the script



