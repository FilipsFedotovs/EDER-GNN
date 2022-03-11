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
parser.add_argument('--Track',help="Name of the control track", default='FEDRA_Track_ID')
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


seed_test_data_l=test_data.rename(columns={'Hit_ID': "Left_Hit"})
seed_test_data_r=test_data.rename(columns={'Hit_ID': "Right_Hit"})
seed_test_data=pd.merge(seed_test_data_l, seed_test_data_r, how="inner", on=[args.Track])
seed_test_data["Hit_Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(seed_test_data["Left_Hit"], seed_test_data["Right_Hit"])]
seed_test_data.drop_duplicates(subset="Hit_Seed_ID",keep='first',inplace=True)
seed_test_data.drop(seed_test_data.index[seed_test_data["Left_Hit"] == seed_test_data["Right_Hit"]], inplace = True)
seed_test_data.drop(["Hit_Seed_ID"],axis=1,inplace=True)
Seed_Test_Count=len(seed_test_data.axes[0])

seed_eval_data_l=eval_data.rename(columns={'Hit_ID': "Left_Hit"})
seed_eval_data_r=eval_data.rename(columns={'Hit_ID': "Right_Hit"})
seed_eval_data=pd.merge(seed_eval_data_l, seed_eval_data_r, how="inner", on=['MC_Mother_Track_ID'])
seed_eval_data["Hit_Seed_ID"]= ['-'.join(sorted(tup)) for tup in zip(seed_eval_data["Left_Hit"], seed_eval_data["Right_Hit"])]
seed_eval_data.drop_duplicates(subset="Hit_Seed_ID",keep='first',inplace=True)
seed_eval_data.drop(seed_eval_data.index[seed_eval_data["Left_Hit"] == seed_eval_data["Right_Hit"]], inplace = True)
seed_eval_data.drop(["Hit_Seed_ID"],axis=1,inplace=True)
Seed_Eval_Count=len(seed_eval_data.axes[0])

seed_merge_data=pd.merge(seed_test_data, seed_eval_data, how="inner", on=["Left_Hit","Right_Hit"])
Seed_Merge_Count=len(seed_merge_data.axes[0])

Recall=round((float(Seed_Merge_Count)/float(Seed_Eval_Count))*100,2)
Precision=round((float(Seed_Merge_Count)/float(Seed_Test_Count))*100,2)

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
matched_data=matched_data.sort_values(['MC_Mother_Track_ID','Hit_ID'],ascending=[1,0])
matched_data=matched_data.drop_duplicates(subset=['MC_Mother_Track_ID'],keep='first')
matched_data=matched_data.drop(['Hit_ID'],axis=1)
N_particles_RRM=len(matched_data)
efficiency=round((float(N_particles_RRM)/float(N_particles_TR))*100,2)
try:
  purity=round((float(N_particles_RRM)/float(N_particles_RR))*100,2)
except:
    purity=0
test_data=test_data.drop(['Track_No'],axis=1)
rec_matched_data=pd.merge(matched_data, test_data, how="inner", on=[args.Track])
eval_matched_data=pd.merge(matched_data, eval_data, how="inner", on=['MC_Mother_Track_ID'])
combined_hits=pd.merge(rec_matched_data, eval_matched_data, how="inner", on=['Hit_ID'])
try:
    avg_track_purity=round((float(len(combined_hits))/float(len(rec_matched_data)))*100,2)
except:
    avg_track_purity=0
try:
    avg_track_efficiency=round((float(len(combined_hits))/float(len(eval_matched_data)))*100,2)
except:
    avg_track_efficiency=0

print(bcolors.HEADER+"############################################# Hit combination metrics ################################################"+bcolors.ENDC)
print('Total 2-hit combinations are expected according to Monte Carlo:',Seed_Eval_Count)
print('Total 2-hit combinations were reconstructed:',Seed_Test_Count)
print('Correct combinations were reconstructed:',Seed_Merge_Count)
print('Therefore the recall of the current model is',bcolors.BOLD+str(Recall), '%'+bcolors.ENDC)
print('And the precision of the current model is',bcolors.BOLD+str(Precision), '%'+bcolors.ENDC)
print(bcolors.HEADER+"############################################# Track reconstruction metrics ################################################"+bcolors.ENDC)
print('Maximum number of particles according to MC Data that can be reconstructed:',N_particles_TR)
print('Maximum number of particles reconstructed:',N_particles_RR)
print('Overall track reconstruction efficiency:',bcolors.BOLD+str(efficiency), '%'+bcolors.ENDC)
print('Overall track reconstruction purity:',bcolors.BOLD+str(purity), '%'+bcolors.ENDC)
print('Track hit utilisation efficiency:',bcolors.BOLD+str(avg_track_efficiency), '%'+bcolors.ENDC)
print('Track hit purity:',bcolors.BOLD+str(avg_track_purity), '%'+bcolors.ENDC)
print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script



