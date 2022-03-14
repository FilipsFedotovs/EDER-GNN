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

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='select cut parameters')
parser.add_argument('--set',help="Enter Z id", default='0')
parser.add_argument('--stepX',help="Enter X step size", default='0')
parser.add_argument('--stepY',help="Enter Y step size", default='0')
parser.add_argument('--stepZ',help="Enter Z step size", default='0')
parser.add_argument('--EOS',help="EOS directory location", default='.')
parser.add_argument('--AFS',help="AFS directory location", default='.')


######################################## Set variables  #############################################################
args = parser.parse_args()

Set=int(args.set)

stepX=float(args.stepX) #The coordinate of the st plate in the current scope
stepZ=float(args.stepZ)
stepY=float(args.stepY)

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
data["Hit_ID"] = data["Hit_ID"].astype(str)
z_offset=data['z'].min()
data['z']=data['z']-z_offset
print(UF.TimeStamp(),'Creating clusters... ')
data.drop(data.index[data['z'] >= ((Set+1)*stepZ)], inplace = True)  #Keeping the relevant z slice
data.drop(data.index[data['z'] < (Set*stepZ)], inplace = True)  #Keeping the relevant z slice
#data=data.reset_index()
print(data)

x_offset=data['x'].min()
y_offset=data['y'].min()
print(x_offset)
print(y_offset)
data['x']=data['x']-x_offset
data['y']=data['y']-y_offset

data_list=data.values.tolist()

x_max=data['x'].max()
y_max=data['y'].max()

Xsteps=math.ceil(x_max/stepX) #Even if use only a max of 20000 track on the right join we cannot perform the full outer join due to the memory limitations, we do it in a small 'cuts'
Ysteps=math.ceil(y_max/stepY)  #Calculating number of cuts
print(Xsteps)
print(Ysteps)
for i in range(0,Xsteps):
    LoadedClusters=[]
    progress=round((float(i)/float(Xsteps))*100,2)
    print(UF.TimeStamp(),"progress is ",progress,' %') #Progress display
    for j in range(0,Ysteps):
        HC=UF.HitCluster([i,j,Set],[stepX,stepY,stepZ])
        HC.LoadClusterHits(data_list)
        LoadedClusters.append(HC)
    output_file_location=EOS_DIR+'/EDER-GNN/Data/REC_SET/R2_R2_SelectedClusters_'+str(Set)+'_'+str(i)+'.pkl'
    open_file = open(output_file_location, "wb")
    pickle.dump(LoadedClusters, output_file_location)
print(UF.TimeStamp(), "Cluster generation is finished...")
#End of the script



