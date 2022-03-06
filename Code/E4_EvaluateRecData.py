# Part of EDER-GNN package
#Made by Filips Fedotovs


########################################    Import libraries    #############################################
import csv
import argparse
import pandas as pd #We use Panda for a routine data processing
import math #We use it for data manipulation
import gc  #Helps to clear memory
import numpy as np
import os
import pickle

class bcolors:   #We use it for the interface
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'






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
from Utility_Functions import Track
#Setting the parser - this script is usually not run directly, but is used by a Master version Counterpart that passes the required arguments
parser = argparse.ArgumentParser(description='This script compares the ouput of the previous step with the output of EDER-GNN reconstructed data to calculate reconstruction performance.')
parser.add_argument('--sf',help="Please choose the input file", default=EOS_DIR+'/EDER-GNN/Data/REC_SET/R1_HITS.csv')
parser.add_argument('--of',help="Please choose the evaluation file (has to match the same geometrical domain and type of the track as the subject", default=EOS_DIR+'/EDER-GNN/Data/TEST_SET/E1_HITS.csv')
parser.add_argument('--Track',help="Name of the control track", default='Fedra_Track_ID')
######################################## Set variables  #############################################################
args = parser.parse_args()

########################################     Preset framework parameters    #########################################
 #The Separation bound is the maximum Euclidean distance that is allowed between hits in the beggining of Seed tracks.


print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-GNN Evaluation module              ########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################              Written by Filips Fedotovs              #########################"+bcolors.ENDC)
print(bcolors.HEADER+"#########################                 PhD Student at UCL                   #########################"+bcolors.ENDC)
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(UF.TimeStamp(), bcolors.OKGREEN+"Modules Have been imported successfully..."+bcolors.ENDC)
print(UF.TimeStamp(),'Analysing evaluation data... ',bcolors.ENDC)
input_GNN_eval_file_location=args.of
if os.path.isfile(input_GNN_eval_file_location)!=True:
                     print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",input_eval_file_location,'is missing, please restart the evaluation sequence scripts'+bcolors.ENDC)
eval_data=pd.read_csv(input_GNN_eval_file_location,header=0,usecols=['Hit_ID','MC_Mother_Track_ID'])
print(UF.TimeStamp(),'Evaluating reconstructed set ',bcolors.ENDC)
test_file_location=args.sf
if os.path.isfile(test_file_location)!=True:
        print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",test_file_location,'is missing, please restart the reconstruction sequence scripts'+bcolors.ENDC)
test_data = pd.read_csv(test_file_location, header=0,
                                usecols=['Hit_ID', args.Track])

test_data_no=test_data.drop(['Hit_ID'],axis=1)
test_data_no['Track_No']=test_data_no[args.Track]
test_data_no=test_data_no.groupby([args.Track],as_index=False).count()
test_data_no = test_data_no[test_data_no.Track_No >= PM.MinHitsTrack]
test_data=pd.merge(test_data, test_data_no, how="inner", on=[args.Track])
N_particles_TR=len(eval_data['MC_Mother_Track_ID'].drop_duplicates(keep='first').axes[0])
N_particles_RR=len(test_data[args.Track].drop_duplicates(keep='first').axes[0])
matched_data=pd.merge(test_data, eval_data, how="inner", on=['Hit_ID'])
matched_data=matched_data.drop(['Track_No'],axis=1)
matched_data=matched_data.groupby([args.Track,'MC_Mother_Track_ID'],as_index=False).count()
matched_data = matched_data[matched_data['Hit_ID'] >= PM.MinHitsTrack]
matched_data=matched_data.sort_values(['MC_Mother_Track_ID',args.Track,'Hit_ID'],ascending=[1,1,1])
print(matched_data)
exit()
N_particles_RRM=len(matched_data[args.Track].drop_duplicates(keep='first').axes[0])
matched_data=matched_data.groupby([args.Track,'MC_Mother_Track_ID'],as_index=False).count()
matched_data = matched_data[matched_data['Hit_ID'] >= PM.MinHitsTrack]
N_particles_RRM=len(matched_data[args.Track].drop_duplicates(keep='first').axes[0])
efficiency=round((float(N_particles_RRM)/float(N_particles_TR))*100,2)
try:
  purity=round((float(N_particles_RRM)/float(N_particles_RR))*100,2)
except:
    purity=0
test_data=test_data.drop(['Track_No'],axis=1)
print(test_data)
print(eval_data)
print(matched_data)
exit()
matched_data=matched_data.drop(['Track_No','Hit_ID'],axis=1)
r_matched_data=matched_data.drop_duplicates(subset=[args.Track],keep='first')
e_matched_data=matched_data.drop_duplicates(subset=['MC_Mother_Track_ID'],keep='first')
rec_matched_data=pd.merge(r_matched_data, test_data, how="inner", on=[args.Track])
eval_matched_data=pd.merge(e_matched_data, eval_data, how="inner", on=['MC_Mother_Track_ID'])
rec_matched_data=rec_matched_data.drop_duplicates(subset=[args.Track],keep='first')
eval_matched_data=eval_matched_data.drop_duplicates(subset=['MC_Mother_Track_ID'],keep='first')
try:
  avg_track_purity=round((float(len(matched_data.axes[0]))/float(len(rec_matched_data.axes[0])))*100,2)
except:
   avg_track_purity=0
try:
    avg_track_efficiency=round((float(len(matched_data.axes[0]))/float(len(eval_matched_data.axes[0])))*100,2)
except:
    avg_track_efficiency=0
print('Maximum number of particles according to MC Data that can be reconstructed:',N_particles_TR)
print('Maximum number of particles reconstructed:',N_particles_RR)
print('Overall track reconstruction efficiency:',bcolors.BOLD+str(efficiency), '%'+bcolors.ENDC)
print('Overall track reconstruction purity:',bcolors.BOLD+str(purity), '%'+bcolors.ENDC)
print('Average track reconstruction efficiency:',bcolors.BOLD+str(avg_track_efficiency), '%'+bcolors.ENDC)
print('Average track reconstruction purity:',bcolors.BOLD+str(avg_track_purity), '%'+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script



