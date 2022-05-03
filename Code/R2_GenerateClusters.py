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
parser.add_argument('--Log',help="Would you like to enable logging? Y/N", default='N')
######################################## Set variables  #############################################################
args = parser.parse_args()
Mode=args.Mode




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
#Specifying the full path to input/output files
input_file_location=EOS_DIR+'/EDER-GNN/Data/REC_SET/R1_HITS.csv'
print(bcolors.HEADER+"########################################################################################################"+bcolors.ENDC)
print(bcolors.HEADER+"######################     Initialising EDER-GNN Cluster Generation module       ########################"+bcolors.ENDC)
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
      UF.RecCleanUp(AFS_DIR, EOS_DIR, 'R3', ['R2_R3','R2_R2'], "SoftUsed == \"EDER-GNN-R2\"")
      print(UF.TimeStamp(),'Submitting jobs... ',bcolors.ENDC)
      for k in range(0,Zsteps):
            OptionHeader = [' --set ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --Log ', ' --cut_dt ', ' --cut_dr ', ' --subset ']
            OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, args.Log, cut_dt,cut_dr, '$1']
            SHName = AFS_DIR + '/HTCondor/SH/SH_R2_' + str(k) + '.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_R2_' + str(k) + '.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_R2_' + str(k)
            ScriptName = AFS_DIR + '/Code/Utilities/R2_GenerateClusters_Sub.py '
            UF.SubmitJobs2Condor([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, Xsteps, 'EDER-GNN-R2', True,False])
      print(UF.TimeStamp(), bcolors.OKGREEN+'All jobs have been submitted, please rerun this script with "--Mode C" in few hours'+bcolors.ENDC)
if Mode=='C':
   bad_pop=[]
   print(UF.TimeStamp(),'Checking jobs... ',bcolors.ENDC)
   for k in range(0,Zsteps):
       progress=round((float(k)/float(Zsteps))*100,2)
       print(UF.TimeStamp(),"progress is ",progress,' %') #Progress display
       for i in range(0,Xsteps):
            OptionHeader = [' --set ', ' --stepX ',' --stepY ',' --stepZ ', ' --EOS ', " --AFS ", " --zOffset ", " --xOffset ", " --yOffset ", ' --Log ', ' --cut_dt ', ' --cut_dr ',' --subset ']
            OptionLine = [k, stepX,stepY,stepZ, EOS_DIR, AFS_DIR, z_offset, x_offset, y_offset, args.Log, cut_dt,cut_dr, i]
            required_output_file_location=EOS_DIR+'/EDER-GNN/Data/REC_SET/R2_R2_SelectedClusters_'+str(k)+'_'+str(i)+'.pkl'
            SHName = AFS_DIR + '/HTCondor/SH/SH_R2_' + str(k) + '.sh'
            SUBName = AFS_DIR + '/HTCondor/SUB/SUB_R2_' + str(k) + '.sub'
            MSGName = AFS_DIR + '/HTCondor/MSG/MSG_R2_' + str(k)
            ScriptName = AFS_DIR + '/Code/Utilities/R2_GenerateClusters_Sub.py '
            if os.path.isfile(required_output_file_location)!=True:
               bad_pop.append([OptionHeader, OptionLine, SHName, SUBName, MSGName, ScriptName, 1, 'EDER-GNN-R2', True,False])
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
       if args.Log!='Y':
           print(UF.TimeStamp(), bcolors.OKGREEN+"Cluster generation is completed, you can start applying GNN on them now"+bcolors.ENDC)
       else:
            fake_results_1=[]
            fake_results_2=[]
            fake_results_3=[]
            fake_results_4=[]
            fake_results_5=[]
            truth_results_1=[]
            truth_results_2=[]
            truth_results_3=[]
            truth_results_4=[]
            truth_results_5=[]

            precision_results_1=[]
            precision_results_2=[]
            precision_results_3=[]
            precision_results_4=[]
            precision_results_5=[]
            recall_results_1=[]
            recall_results_2=[]
            recall_results_3=[]
            recall_results_4=[]
            recall_results_5=[]
            for k in range(0,Zsteps):
               progress=round((float(k)/float(Zsteps))*100,2)
               print(UF.TimeStamp(),"progress is ",progress,' %') #Progress display

               for i in range(0,Xsteps):
                    required_output_file_location=EOS_DIR+'/EDER-GNN/Data/REC_SET/R2_R2_SelectedClusters_'+str(k)+'_'+str(i)+'.pkl'
                    if os.path.isfile(required_output_file_location)!=True:
                       print(UF.TimeStamp(), bcolors.FAIL+"Critical fail: file",required_output_file_location,'is missing, please restart the script with the option "--Mode R"'+bcolors.ENDC)
                    elif os.path.isfile(required_output_file_location):
                        cluster_data_file=open(required_output_file_location,'rb')
                        cluster_data=pickle.load(cluster_data_file)
                        for cd in cluster_data:
                            result_temp=cd.Stats
                            fake_results_1.append(int(result_temp[1][0]))
                            fake_results_2.append(int(result_temp[1][1]))
                            fake_results_3.append(int(result_temp[1][2]))
                            fake_results_4.append(int(result_temp[1][3]))
                            fake_results_5.append(int(result_temp[1][4]))
                            fake_results_6.append(int(result_temp[1][5]))
                            truth_results_1.append(int(result_temp[2][0]))
                            truth_results_2.append(int(result_temp[2][1]))
                            truth_results_3.append(int(result_temp[2][2]))
                            truth_results_4.append(int(result_temp[2][3]))
                            truth_results_5.append(int(result_temp[2][4]))
                            truth_results_6.append(int(result_temp[2][5]))
                            try:
                                precision_results_1.append(int(result_temp[2][0])/(int(result_temp[2][0])+int(result_temp[1][0])))
                                precision_results_2.append(int(result_temp[2][1])/(int(result_temp[2][1])+int(result_temp[1][1])))
                                precision_results_3.append(int(result_temp[2][2])/(int(result_temp[2][2])+int(result_temp[1][2])))
                                precision_results_4.append(int(result_temp[2][3])/(int(result_temp[2][3])+int(result_temp[1][3])))
                                precision_results_5.append(int(result_temp[2][4])/(int(result_temp[2][4])+int(result_temp[1][4])))
                                precision_results_6.append(int(result_temp[2][5])/(int(result_temp[2][5])+int(result_temp[1][5])))
                                recall_results_1.append(int(result_temp[2][0])/(int(result_temp[2][2])))
                                recall_results_2.append(int(result_temp[2][1])/(int(result_temp[2][2])))
                                recall_results_3.append(int(result_temp[2][2])/(int(result_temp[2][2])))
                                recall_results_4.append(int(result_temp[2][3])/(int(result_temp[2][2])))
                                recall_results_5.append(int(result_temp[2][4])/(int(result_temp[2][2])))
                                recall_results_6.append(int(result_temp[2][5])/(int(result_temp[2][2])))
                            except:
                                continue
                            label=result_temp[0]
            print(UF.TimeStamp(),bcolors.OKGREEN+'Raw results have been compiled and presented bellow:'+bcolors.ENDC)
            print(tabulate([[label[0], np.average(fake_results_1), np.average(truth_results_1), np.average(precision_results_1), np.std(precision_results_1), np.average(recall_results_1), np.std(recall_results_1)], \
                            [label[1], np.average(fake_results_2), np.average(truth_results_2), np.average(precision_results_2), np.std(precision_results_2), np.average(recall_results_2), np.std(recall_results_2)], \
                            [label[2], np.average(fake_results_3), np.average(truth_results_3), np.average(precision_results_3), np.std(precision_results_3), np.average(recall_results_3), np.std(recall_results_3)], \
                            [label[3], np.average(fake_results_4), np.average(truth_results_4), np.average(precision_results_4), np.std(precision_results_4), np.average(recall_results_4), np.std(recall_results_4)],\
                            [label[4], np.average(fake_results_5), np.average(truth_results_5), np.average(precision_results_5), np.std(precision_results_5), np.average(recall_results_5), np.std(recall_results_5)]], \
                            headers=['Step', 'Avg # Fake edges', 'Avg # of Genuine edges', 'Avg precision', 'Precision std','Avg recall', 'Recall std' ], tablefmt='orgtbl'))
            # print(UF.TimeStamp(),bcolors.OKGREEN+'Performance metrics are presented bellow'+bcolors.ENDC)
            # Precision_Nom=np.average(truth_results_5)
            # Precision_Nom_Err=np.std(truth_results_5)
            # Precision_Den=np.average(fake_results_5)+np.average(truth_results_5)
            # Precision_Den_Err=UF.ErrorOperations(np.average(fake_results_5),np.average(truth_results_5),np.std(fake_results_5),np.std(truth_results_5),'+')
            # Precision=round((Precision_Nom/Precision_Den)*100,2)
            # Precision_Err=round(UF.ErrorOperations(Precision_Nom,Precision_Den,Precision_Nom_Err,Precision_Den_Err,'/')*100,2)
            # print(tabulate([['Precision', Precision, Precision_Err]], headers=['Average [%]', 'Error [%]'], tablefmt='orgtbl'))
       print(bcolors.HEADER+"############################################# End of the program ################################################"+bcolors.ENDC)
#End of the script



