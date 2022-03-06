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
test_data['Track_No']=test_data[args.Track]
test_data=test_data.groupby([args.Track],as_index=False).count()
test_data = test_data[test_data.Track_No >= PM.MinHitsTrack]
test_data['Track_No']=test_data[args.Track]
N_particles_TR=len(eval_data['MC_Mother_Track_ID'].drop_duplicates(keep='first').axes[0])
N_particles_RR=len(test_data[args.Track].drop_duplicates(keep='first').axes[0])
matched_data=pd.merge(test_data, eval_data, how="inner", on=['Hit_ID'])
N_particles_RRM=len(matched_data[args.Track].drop_duplicates(keep='first').axes[0])
N_particles_TRM=len(matched_data['MC_Mother_Track_ID'].drop_duplicates(keep='first').axes[0])
matched_data=matched_data.groupby([args.Track,'MC_Mother_Track_ID'],as_index=False).count()
print(matched_data)
matched_data = matched_data[matched_data.Track_No >= PM.MinHitsTrack]
print(matched_data)
exit()
#     seg_data['FEDRA_Seg_No']=seg_data['FEDRA_Seg_ID']
#     seg_data=seg_data.groupby(by=['MC_Mother_Track_ID','FEDRA_Seg_ID'])['FEDRA_Seg_No'].count().reset_index()
#     seg_data=seg_data.drop(seg_data.index[seg_data['FEDRA_Seg_No'] < 2])
#     seg_data_segm_kpi=seg_data.drop(['FEDRA_Seg_No'],axis=1)
#     seg_data_segm_kpi=seg_data_segm_kpi.groupby(by=['MC_Mother_Track_ID'])['FEDRA_Seg_ID'].count().reset_index()
#     TotalFullMCTracks=seg_data_segm_kpi['MC_Mother_Track_ID'].nunique()
#     TotalFullFEDRATracks=seg_data_segm_kpi['FEDRA_Seg_ID'].sum()
#     Segmentation=seg_data_segm_kpi['FEDRA_Seg_ID'].mean()
#     output_file_location = EOS_DIR + '/EDER-TSU/Data/TEST_SET/E4_MC_TRACK_SEGMENTATION_STATS.csv'
#     seg_data_segm_kpi.to_csv(output_file_location,index=False)
#     print(UF.TimeStamp(), bcolors.OKGREEN+"Stats have on MC Track segmentation has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
#     mc_min = rec_data.groupby('MC_Mother_Track_ID')['z'].min()
#     mc_min=mc_min.reset_index()
#     rec_data_mc_min=pd.merge(mc_min, rec_data, how="inner", on=['MC_Mother_Track_ID','z'])
#     rec_data_mc_min.drop_duplicates(subset='MC_Mother_Track_ID',keep='first',inplace=True)
#     rec_data_mc_min=rec_data_mc_min.drop(['FEDRA_Seg_ID'],axis=1)
#     rec_data_mc_min=rec_data_mc_min.rename(columns={"x": "mc_s_x"})
#     rec_data_mc_min=rec_data_mc_min.rename(columns={"y": "mc_s_y"})
#     rec_data_mc_min=rec_data_mc_min.rename(columns={"z": "mc_s_z"})
#     mc_max = rec_data.groupby('MC_Mother_Track_ID')['z'].max()
#     mc_max=mc_max.reset_index()
#     rec_data_mc_max=pd.merge(mc_max, rec_data, how="inner", on=['MC_Mother_Track_ID','z'])
#     rec_data_mc_max.drop_duplicates(subset='MC_Mother_Track_ID',keep='first',inplace=True)
#     rec_data_mc_max=rec_data_mc_max.drop(['FEDRA_Seg_ID'],axis=1)
#     rec_data_mc_max=rec_data_mc_max.rename(columns={"x": "e_x"})
#     rec_data_mc_max=rec_data_mc_max.rename(columns={"y": "e_y"})
#     rec_data_mc_max=rec_data_mc_max.rename(columns={"z": "e_z"})
#     rec_data_mc=pd.merge(rec_data_mc_min, rec_data_mc_max, how="inner", on=['MC_Mother_Track_ID'])
#     rec_data_mc['MC_Track_Rec_Len']=np.sqrt((rec_data_mc['mc_s_x']-rec_data_mc['e_x'])**2+((rec_data_mc['mc_s_y']-rec_data_mc['e_y'])**2)+((rec_data_mc['mc_s_z']-rec_data_mc['e_z'])**2))
#     rec_data_mc=rec_data_mc.drop(['e_x'],axis=1)
#     rec_data_mc=rec_data_mc.drop(['e_y'],axis=1)
#     rec_data_mc=rec_data_mc.drop(['e_z'],axis=1)
#     fedra_min = rec_data.groupby('FEDRA_Seg_ID')['z'].min()
#     fedra_min=fedra_min.reset_index()
#     rec_data_fedra_min=pd.merge(fedra_min, rec_data, how="inner", on=['FEDRA_Seg_ID','z'])
#     rec_data_fedra_min=rec_data_fedra_min.drop(['MC_Mother_Track_ID'],axis=1)
#     rec_data_fedra_min=rec_data_fedra_min.rename(columns={"x": "fedra_s_x"})
#     rec_data_fedra_min=rec_data_fedra_min.rename(columns={"y": "fedra_s_y"})
#     rec_data_fedra_min=rec_data_fedra_min.rename(columns={"z": "fedra_s_z"})
#     fedra_max = rec_data.groupby('FEDRA_Seg_ID')['z'].max()
#     fedra_max=fedra_max.reset_index()
#     rec_data_fedra_max=pd.merge(fedra_max, rec_data, how="inner", on=['FEDRA_Seg_ID','z'])
#     rec_data_fedra_max=rec_data_fedra_max.drop(['MC_Mother_Track_ID'],axis=1)
#     rec_data_fedra_max=rec_data_fedra_max.rename(columns={"x": "e_x"})
#     rec_data_fedra_max=rec_data_fedra_max.rename(columns={"y": "e_y"})
#     rec_data_fedra_max=rec_data_fedra_max.rename(columns={"z": "e_z"})
#     rec_data_fedra=pd.merge(rec_data_fedra_min, rec_data_fedra_max, how="inner", on=['FEDRA_Seg_ID'])
#     rec_data_fedra['FEDRA_Track_Rec_Len']=np.sqrt((rec_data_fedra['fedra_s_x']-rec_data_fedra['e_x'])**2+((rec_data_fedra['fedra_s_y']-rec_data_fedra['e_y'])**2)+((rec_data_fedra['fedra_s_z']-rec_data_fedra['e_z'])**2))
#     rec_data_fedra=rec_data_fedra.drop(['e_x'],axis=1)
#     rec_data_fedra=rec_data_fedra.drop(['e_y'],axis=1)
#     rec_data_fedra=rec_data_fedra.drop(['e_z'],axis=1)
#     seg_data=seg_data.sort_values(['MC_Mother_Track_ID','FEDRA_Seg_No'],ascending=[1,0])
#     seg_data.drop_duplicates(subset='MC_Mother_Track_ID',keep='first',inplace=True)
#     seg_data=pd.merge(seg_data, rec_data_fedra, how="inner", on=['FEDRA_Seg_ID'])
#     seg_data=pd.merge(seg_data, rec_data_mc, how="inner", on=['MC_Mother_Track_ID'])
#     output_file_location = EOS_DIR + '/EDER-TSU/Data/TEST_SET/E4_FEDRA_TRACK_PROPERTY_STATS.csv'
#     seg_data['Displacement']=np.sqrt((seg_data['fedra_s_x']-seg_data['mc_s_x'])**2+((seg_data['fedra_s_y']-seg_data['mc_s_y'])**2)+((seg_data['fedra_s_z']-seg_data['mc_s_z'])**2))
#     seg_data.to_csv(output_file_location,index=False)
#     print(UF.TimeStamp(), bcolors.OKGREEN+"Stats have on FEDRA Track properties has been written to"+bcolors.ENDC, bcolors.OKBLUE+output_file_location+bcolors.ENDC)
print('Maximum number of particles according to MC Data that can be reconstructed:',N_particles_TR)
print('Maximum number of particles reconstructed:',N_particles_RR)
#     print('Therefore the segmentation is:',round(Segmentation,3))
#     print('The average discrepency between MC and FEDRA track start position is',bcolors.BOLD+str(int(seg_data['Displacement'].mean())), 'microns'+bcolors.ENDC)
#     print('On average MC Track is ',bcolors.BOLD+str(int(seg_data['MC_Track_Rec_Len'].mean())), 'microns in length'+bcolors.ENDC)
#     print('On average the largest segment of MC track that was reconstructed by FEDRA is:',bcolors.BOLD+str(int(seg_data['FEDRA_Track_Rec_Len'].mean())), 'microns in length'+bcolors.ENDC)
#     print('On average only  ',bcolors.BOLD+str(round(seg_data['FEDRA_Track_Rec_Len'].sum()/seg_data['MC_Track_Rec_Len'].sum(),2)*100), '% of MC track is reconstructed'+bcolors.ENDC)
#     print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#     #End of the script



