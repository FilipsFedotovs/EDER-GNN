#This simple script prepares 2-Track seeds for the initial CNN vertexing
# Part of EDER-VIANN package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import numpy as np
import os
import pickle
from tabulate import tabulate

class bcolors:   #We use it for the interface
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script takes preselected 2-track seed candidates from previous step and refines them by applying additional cuts on the parameters such as DOCA, fiducial cute and distance to the possible vertex origin.')
parser.add_argument('--Mode',help="Running Mode: Reset(R)/Continue(C)", default='C')
parser.add_argument('--Sampling',help="How much sampling?", default='1.0')
######################################## Set variables  #############################################################
args = parser.parse_args()
Mode=args.Mode
Sampling=float(args.Sampling)



#Loading Directory locations
csv_reader=open('../config',"r")
config = list(csv.reader(csv_reader))
for c in config:
    if c[0]=='AFS_DIR':
        AFS_DIR=c[1]
    if c[0]=='EOS_DIR':
        EOS_DIR=c[1]
csv_reader.close()
import sys


sys.path.insert(1, AFS_DIR+'/Code/Utilities/')
import Utility_Functions as UF #This is where we keep routine utility functions
import Parameters as PM #This is where we keep framework global parameters
########################################     Preset framework parameters    #########################################
stepX=PM.stepX
stepY=PM.stepY
stepZ=PM.stepZ
cut_dt=PM.cut_dt
cut_dr=PM.cut_dr
testRatio=PM.testRatio
valRatio=PM.valRatio
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-GNN/Data/REC_SET/R1_HITS.csv'
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-GNN Train Cluster Generation module   #####################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Loading preselected data from ',bcolors.OKBLUE+input_file_location+bcolors.ENDC)
data=pd.read_csv(input_file_location,header=0,usecols=['z','x','y'])
print(UF.TimeStamp(),'Analysing data... ',bcolors.ENDC)
z_offset=data['z'].min()
data['z']=data['z']-z_offset
z_max=data['z'].max()
Zsteps=math.ceil(z_max/stepZ)
y_offset=data['y'].min()
x_offset=data['x'].min()
data['x']=data['x']-x_offset
x_max=data['x'].max()
Xsteps=math.ceil(x_max/stepX) #Even if use only a max of 20000 track on the right join we cannot perform the full outer join due to the memory limitations, we do it in a small 'cuts'
if Mode=='R':
   print(UF.TimeStamp(),bcolors.WARNING+'Warning! You are running the script with the "Mode R" option which means that you want to vertex the seeds from the scratch'+bcolors.ENDC)
   print(UF.TimeStamp(),bcolors.WARNING+'This option will erase all the previous Seed vertexing jobs/results'+bcolors.ENDC)
   UserAnswer=input(bcolors.BOLD+"Would you like to continue (Y/N)? \n"+bcolors.ENDC)
   if UserAnswer=='N':
         Mode='C'
         print(UF.TimeStamp(),'OK, continuing then...')

   if UserAnswer=='Y':
      print(UF.TimeStamp(),'Performing the cleanup... ',bcolors.ENDC)
      UF.TrainCleanUp(AFS_DIR, EOS_DIR, 'M1', ['M1_M1','M1_M2'], "SoftUsed == \"EDER-GNN-M1\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      for k in range(0,Zsteps):
            OptionHeader = [' --set ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --cut_dt ', ' --cut_dr ', ' --testRatio ', ' --valRatio ', ' --subset ']
            OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, cut_dt,cut_dr,testRatio,'$1']
            SHName = AFS_DIR + '/HTCondor/SH/SH_M1_' + str(k) + '.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M1_' + str(k) + '.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M1_' + str(k)
            ScriptName = AFS_DIR + '/Code/Utilities/M1_GenerateTrainClusters_Sub.py '
            UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'EDER-GNN-M1', False,False])
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   bad_pop=[]
   print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)
   for k in range(0,Zsteps):
       progress=round((float(k)/float(Zsteps))*100,2)
       print(UF.TimeStamp(),"progress is ",progress,' %') #Progress display
       for i in range(0,Xsteps):
            OptionHeader = [' --set ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --cut_dt ', ' --cut_dr ', ' --testRatio ', ' --valRatio ', ' --subset ']
            OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, cut_dt,cut_dr,testRatio,valRatio, i]
            required_output_file_location=EOS_DIR+'/EDER-GNN/Data/TRAIN_SET/M1_M1_SelectedTrainClusters_'+str(k)+'_'+str(i)+'.pkl'
            SHName = AFS_DIR + '/HTCondor/SH/SH_M1_' + str(k) + '.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_M1_' + str(k) + '.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_M1_' + str(k)
            ScriptName = AFS_DIR + '/Code/Utilities/M1_GenerateTrainClusters_Sub.py '
            if os.path.isfile(required_output_file_location)!=True:
               bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-GNN-M1', False,False])
   if len(bad_pop)>0:
     print(UF.TimeStamp(),bcolors.WARNING+'Warning, there are still', len(bad_pop), 'HTCondor jobs remaining'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to wait and try again later please enter W'+bcolors.ENDC)
     print(bcolors.BOLD+'If you would like to resubmit please enter R'+bcolors.ENDC)
     UserAnswer=input(bcolors.BOLD+"Please, enter your option\n"+bcolors.ENDC)
     if UserAnswer=='W':
         print(UF.TimeStamp(),'OK, exiting now then')
         exit()
     if UserAnswer=='R':
        for bp in bad_pop:
             UF.SubmitJobs2Condor(bp)
        print(UF.TimeStamp(), bcolors.OKGREEN+"All jobs have been resubmitted"+bcolors.ENDC)
        print(bcolors.BOLD+"Please check them in few hours"+bcolors.ENDC)
        exit()
   else:
       print(UF.TimeStamp(),bcolors.OKGREEN+'All HTCondor Seed Creation jobs have finished'+bcolors.ENDC)
       exit()
       print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)
       count=0
       for k in range(0,Zsteps):
           progress=round((float(k)/float(Zsteps))*100,2)
           print(UF.TimeStamp(),"progress is ",progress,' %') #Progress display
           for i in range(0,Xsteps):
                count+=1
                source_output_file_location=EOS_DIR+'/EDER-GNN/Data/TRAIN_SET/M1_M1_SelectedTrainClusters_'+str(k)+'_'+str(i)+'.pkl'
                destination_output_file_location=EOS_DIR+'/EDER-GNN/Data/TRAIN_SET/M1_M2_SelectedTrainClusters_'+str(count)+'.pkl'
                os.rename(source_output_file_location, destination_output_file_location)
       print(UF.TimeStamp(),bcolors.OKGREEN+'Files are ready for the model training'+bcolors.ENDC)
       print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script



