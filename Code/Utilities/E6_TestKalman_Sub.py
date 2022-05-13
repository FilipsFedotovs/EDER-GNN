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
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--Z_ID',help="Enter Z id", default='0')
parser.add_argument('--X_ID',help="Please enter the cluster set", default='1')
parser.add_argument('--Y_ID',help="Please enter the cluster set", default='1')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')
parser.add_argument('--stepX',help="Enter X step size", default='0')
parser.add_argument('--stepY',help="Enter Y step size", default='0')
parser.add_argument('--stepZ',help="Enter Z step size", default='0')
parser.add_argument('--zOffset',help="Data offset on z", default='0.0')
parser.add_argument('--yOffset',help="Data offset on y", default='0.0')
parser.add_argument('--xOffset',help="Data offset on x", default='0.0')
######################################## Set variables  #############################################################
args = parser.parse_args()

Z_ID=int(args.Z_ID)
X_ID=int(args.X_ID)
Y_ID=int(args.Y_ID)
stepX=float(args.stepX) #The coordinate of the st plate in the current scope
stepZ=float(args.stepZ)
stepY=float(args.stepY)
z_offset=float(args.zOffset)
y_offset=float(args.yOffset)
x_offset=float(args.xOffset)

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

input_file_location=EOS_DIR+'/EDER-GNN/Data/TEST_SET/E5_HITS.csv'
FEDRAdata=pd.read_csv(input_file_location,header=0,
                            usecols=["Hit_ID","x","y","z","tx","ty",'FEDRA_Track_ID'])
FEDRAdata["x"] = pd.to_numeric(FEDRAdata["x"],downcast='float')
FEDRAdata["y"] = pd.to_numeric(FEDRAdata["y"],downcast='float')
FEDRAdata["z"] = pd.to_numeric(FEDRAdata["z"],downcast='float')
FEDRAdata["Hit_ID"] = FEDRAdata["Hit_ID"].astype(str)
FEDRAdata['z']=FEDRAdata['z']-z_offset
FEDRAdata['x']=FEDRAdata['x']-x_offset
FEDRAdata['y']=FEDRAdata['y']-y_offset
FEDRAdata.drop(FEDRAdata.index[FEDRAdata['z'] >= ((Z_ID+1)*stepZ)], inplace = True)  #Keeping the relevant z slice
FEDRAdata.drop(FEDRAdata.index[FEDRAdata['z'] < (Z_ID*stepZ)], inplace = True)  #Keeping the relevant z slice
FEDRAdata.drop(FEDRAdata.index[FEDRAdata['x'] >= ((X_ID+1)*stepX)], inplace = True)  #Keeping the relevant z slice
FEDRAdata.drop(FEDRAdata.index[FEDRAdata['x'] < (X_ID*stepX)], inplace = True)  #Keeping the relevant z slice
FEDRAdata.drop(FEDRAdata.index[FEDRAdata['y'] >= ((Y_ID+1)*stepY)], inplace = True)  #Keeping the relevant z slice
FEDRAdata.drop(FEDRAdata.index[FEDRAdata['y'] < (Y_ID*stepY)], inplace = True)  #Keeping the relevant z slice
FEDRAdata_list=FEDRAdata.values.tolist()
LoadedClusters=[]
RawClusters[0].TestKalmanHits(FEDRAdata_list,MCdata_list)
LoadedClusters.append(RawClusters[0])
output_file_location=EOS_DIR+'/EDER-GNN/Data/TEST_SET/E6_LinkedClusters_'+str(Z_ID)+'_'+str(X_ID)+'_'+str(Y_ID)+'.pkl'
open_file = open(output_file_location, "wb")
pickle.dump(LoadedClusters, open_file)
print(UF.TimeStamp(), "Fedra reconstruction is finished...")
#End of the script



