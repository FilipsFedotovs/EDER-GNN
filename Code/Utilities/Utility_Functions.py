###This file contains the utility functions that are commonly used in EDER_VIANN packages

import csv
import math
import os, shutil
import subprocess
#import time as t
import datetime
import ast
import numpy as np
#import scipy
import copy
import random
#from scipy.stats import chisquare

#This utility provides Timestamps for print messages
def TimeStamp():
 return "["+datetime.datetime.now().strftime("%D")+' '+datetime.datetime.now().strftime("%H:%M:%S")+"]"

class HitCluster:
      def __init__(self,ClusterID,Step):
          self.ClusterID=ClusterID
          self.Step=Step
      def __eq__(self, other):
        return ('-'.join(str(self.ClusterID))) == ('-'.join(str(other.ClusterID)))
      def __hash__(self):
        return hash(('-'.join(str(self.ClusterID))))
      def LoadClusterHits(self,RawHits): #Decorate hit information
           self.ClusterHits=[]
           self.ClusterHitIDs=[]
           __ClusterHitsTemp=[]
           for s in RawHits:
               if s[1]>=self.ClusterID[0]*self.Step[0] and s[1]<((self.ClusterID[0]+1)*self.Step[0]):
                   if s[2]>=self.ClusterID[1]*self.Step[1] and s[2]<((self.ClusterID[1]+1)*self.Step[1]):
                       if s[3]>=self.ClusterID[2]*self.Step[2] and s[3]<((self.ClusterID[2]+1)*self.Step[2]):
                          __ClusterHitsTemp.append([(s[1]-(self.ClusterID[0]*self.Step[0])),(s[2]-(self.ClusterID[1]*self.Step[1])), (s[3]-(self.ClusterID[2]*self.Step[2])), s[4], s[5]])
                          self.ClusterHitIDs.append(s[0])
                          self.ClusterHits.append(s)
           self.ClusterSize=len(__ClusterHitsTemp)
           import torch
           import torch_geometric
           from torch_geometric.data import Data
           self.ClusterGraph=Data(x=torch.Tensor(__ClusterHitsTemp), edge_index=None, y=None)
           del __ClusterHitsTemp

      def GenerateTrainData(self, MCHits, val_ratio, test_ratio,cut_dt, cut_dr): #Decorate hit information
           import pandas as pd
           _MCClusterHits=[]
           for s in MCHits:
               if s[1]>=self.ClusterID[0]*self.Step[0] and s[1]<((self.ClusterID[0]+1)*self.Step[0]):
                   if s[2]>=self.ClusterID[1]*self.Step[1] and s[2]<((self.ClusterID[1]+1)*self.Step[1]):
                       if s[3]>=self.ClusterID[2]*self.Step[2] and s[3]<((self.ClusterID[2]+1)*self.Step[2]):
                          _MCClusterHits.append([s[0],s[6]])
           #Preparing Raw and MC combined data 1
           _l_MCHits=pd.DataFrame(_MCClusterHits, columns = ['l_HitID','l_MC_ID'])
           _l_Hits=pd.DataFrame(self.ClusterHits, columns = ['l_HitID','l_x','l_y','l_z','l_tx','l_ty'])
           #Join hits + MC truth
           _l_Tot_Hits=pd.merge(_l_MCHits, _l_Hits, how="right", on=['l_HitID'])
           _l_Tot_Hits['join_key'] = 'join_key'

           #Preparing Raw and MC combined data 2
           _r_MCHits=pd.DataFrame(_MCClusterHits, columns = ['r_HitID','r_MC_ID'])
           _r_Hits=pd.DataFrame(self.ClusterHits, columns = ['r_HitID','r_x','r_y','r_z','r_tx','r_ty'])
           #Join hits + MC truth
           _r_Tot_Hits=pd.merge(_r_MCHits, _r_Hits, how="right", on=['r_HitID'])
           _r_Tot_Hits['join_key'] = 'join_key'

           #Combining data 1 and 2
           _Tot_Hits=pd.merge(_l_Tot_Hits, _r_Tot_Hits, how="inner", on=['join_key'])
           _Tot_Hits.l_MC_ID= _Tot_Hits.l_MC_ID.fillna(_Tot_Hits.l_HitID)
           _Tot_Hits.r_MC_ID= _Tot_Hits.r_MC_ID.fillna(_Tot_Hits.r_HitID)
           #_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_HitID'] == _Tot_Hits['r_HitID']], inplace = True)
           #_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_z'] <= _Tot_Hits['r_z']], inplace = True)
           _Tot_Hits['d_tx'] = _Tot_Hits['l_tx']-_Tot_Hits['r_tx']
           _Tot_Hits['d_tx'] = _Tot_Hits['d_tx'].abs()
           _Tot_Hits['d_ty'] = _Tot_Hits['l_ty']-_Tot_Hits['r_ty']
           _Tot_Hits['d_ty'] = _Tot_Hits['d_ty'].abs()
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_tx'] >= cut_dt], inplace = True)
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_ty'] >= cut_dt], inplace = True)
           _Tot_Hits['d_x'] = (_Tot_Hits['r_x']-(_Tot_Hits['l_x']+(_Tot_Hits['l_tx']*(_Tot_Hits['r_z']-_Tot_Hits['l_z']))))
           _Tot_Hits['d_x'] = _Tot_Hits['d_x'].abs()
           _Tot_Hits['d_y'] = (_Tot_Hits['r_y']-(_Tot_Hits['l_y']+(_Tot_Hits['l_ty']*(_Tot_Hits['r_z']-_Tot_Hits['l_z']))))
           _Tot_Hits['d_y'] = _Tot_Hits['d_y'].abs()
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_x'] >= cut_dr], inplace = True)
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_y'] >= cut_dr], inplace = True)
           _Genuine=_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']])
           _Fakes=_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] == _Tot_Hits['r_MC_ID']])
           _Genuine = _Genuine.drop(['d_tx','d_ty','d_x','d_y','join_key','r_x','r_y','r_z','l_x','l_y','l_z','l_tx','l_ty','r_tx','r_ty','l_MC_ID','r_MC_ID'],axis=1)
           _Fakes = _Fakes.drop(['d_tx','d_ty','d_x','d_y','join_key','r_x','r_y','r_z','l_x','l_y','l_z','l_tx','l_ty','r_tx','r_ty','l_MC_ID','r_MC_ID'],axis=1)
           _min_n=min(len(_Genuine),len(_Fakes))
           _Genuine=_Genuine.sample(n=_min_n)
           _Fakes=_Fakes.sample(n=_min_n)
           _TestSize=int(round(len(_Fakes)*test_ratio))
           _ValSize=int(round(len(_Fakes)*val_ratio))
           _FakeList=_Fakes.values.tolist()
           _GenuineList=_Genuine.values.tolist()
           _FakeTestList=_FakeList[0:_TestSize]
           _GenuineTestList=_GenuineList[0:_TestSize]
           _FakeValList=_FakeList[_TestSize:(_ValSize+_TestSize)]
           _GenuineValList=_GenuineList[_TestSize:(_ValSize+_TestSize)]
           _FakeList=_FakeList[(_ValSize+_TestSize):]
           _GenuineList=_GenuineList[(_ValSize+_TestSize):]
           import torch
           self.ClusterGraph.val_pos_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_GenuineValList,self.ClusterHitIDs)))
           self.ClusterGraph.test_pos_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_GenuineTestList,self.ClusterHitIDs)))
           self.ClusterGraph.val_neg_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_FakeValList,self.ClusterHitIDs)))
           self.ClusterGraph.test_neg_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_FakeTestList,self.ClusterHitIDs)))
           self.ClusterGraph.train_neg_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_FakeList,self.ClusterHitIDs)))
           self.ClusterGraph.train_pos_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_GenuineList,self.ClusterHitIDs)))

      def GenerateTrainDatav2(self, MCHits,cut_dt, cut_dr): #Decorate hit information
           import pandas as pd
           _MCClusterHits=[]
           for s in MCHits:
               if s[1]>=self.ClusterID[0]*self.Step[0] and s[1]<((self.ClusterID[0]+1)*self.Step[0]):
                   if s[2]>=self.ClusterID[1]*self.Step[1] and s[2]<((self.ClusterID[1]+1)*self.Step[1]):
                       if s[3]>=self.ClusterID[2]*self.Step[2] and s[3]<((self.ClusterID[2]+1)*self.Step[2]):
                          _MCClusterHits.append([s[0],s[6]])
           #Preparing Raw and MC combined data 1
           _l_MCHits=pd.DataFrame(_MCClusterHits, columns = ['l_HitID','l_MC_ID'])
           _l_Hits=pd.DataFrame(self.ClusterHits, columns = ['l_HitID','l_x','l_y','l_z','l_tx','l_ty'])
           #Join hits + MC truth
           _l_Tot_Hits=pd.merge(_l_MCHits, _l_Hits, how="right", on=['l_HitID'])
           _l_Tot_Hits['join_key'] = 'join_key'

           #Preparing Raw and MC combined data 2
           _r_MCHits=pd.DataFrame(_MCClusterHits, columns = ['r_HitID','r_MC_ID'])
           _r_Hits=pd.DataFrame(self.ClusterHits, columns = ['r_HitID','r_x','r_y','r_z','r_tx','r_ty'])
           #Join hits + MC truth
           _r_Tot_Hits=pd.merge(_r_MCHits, _r_Hits, how="right", on=['r_HitID'])
           _r_Tot_Hits['join_key'] = 'join_key'

           #Combining data 1 and 2
           _Tot_Hits=pd.merge(_l_Tot_Hits, _r_Tot_Hits, how="inner", on=['join_key'])
           _Tot_Hits.l_MC_ID= _Tot_Hits.l_MC_ID.fillna(_Tot_Hits.l_HitID)
           _Tot_Hits.r_MC_ID= _Tot_Hits.r_MC_ID.fillna(_Tot_Hits.r_HitID)
           #_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_HitID'] == _Tot_Hits['r_HitID']], inplace = True)
           #_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_z'] <= _Tot_Hits['r_z']], inplace = True)
           _Tot_Hits['d_tx'] = _Tot_Hits['l_tx']-_Tot_Hits['r_tx']
           _Tot_Hits['d_tx'] = _Tot_Hits['d_tx'].abs()
           _Tot_Hits['d_ty'] = _Tot_Hits['l_ty']-_Tot_Hits['r_ty']
           _Tot_Hits['d_ty'] = _Tot_Hits['d_ty'].abs()
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_tx'] >= cut_dt], inplace = True)
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_ty'] >= cut_dt], inplace = True)
           _Tot_Hits['d_x'] = (_Tot_Hits['r_x']-(_Tot_Hits['l_x']+(_Tot_Hits['l_tx']*(_Tot_Hits['r_z']-_Tot_Hits['l_z']))))
           _Tot_Hits['d_x'] = _Tot_Hits['d_x'].abs()
           _Tot_Hits['d_y'] = (_Tot_Hits['r_y']-(_Tot_Hits['l_y']+(_Tot_Hits['l_ty']*(_Tot_Hits['r_z']-_Tot_Hits['l_z']))))
           _Tot_Hits['d_y'] = _Tot_Hits['d_y'].abs()
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_x'] >= cut_dr], inplace = True)
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_y'] >= cut_dr], inplace = True)
           _Genuine=_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']])
           _Fakes=_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] == _Tot_Hits['r_MC_ID']])


           _Genuine = _Genuine.drop(['d_tx','d_ty','d_x','d_y','join_key','r_x','r_y','r_z','l_x','l_y','l_z','l_tx','l_ty','r_tx','r_ty','l_MC_ID','r_MC_ID'],axis=1)
           _Fakes = _Fakes.drop(['d_tx','d_ty','d_x','d_y','join_key','r_x','r_y','r_z','l_x','l_y','l_z','l_tx','l_ty','r_tx','r_ty','l_MC_ID','r_MC_ID'],axis=1)
           _Genuine['Connection']=1
           _Fakes['Connection']=0
           _FakeList=_Fakes.values.tolist()
           _GenuineList=_Genuine.values.tolist()
           _FinalList=_FakeList+_GenuineList
           random.shuffle(_FinalList)

           self.ClusterGraph.edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_FinalList,self.ClusterHitIDs)))
           print(self.ClusterGraph.edge_index)
           exit()
           import torch
           self.ClusterGraph.val_pos_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_GenuineValList,self.ClusterHitIDs)))
           self.ClusterGraph.test_pos_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_GenuineTestList,self.ClusterHitIDs)))
           self.ClusterGraph.val_neg_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_FakeValList,self.ClusterHitIDs)))
           self.ClusterGraph.test_neg_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_FakeTestList,self.ClusterHitIDs)))
           self.ClusterGraph.train_neg_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_FakeList,self.ClusterHitIDs)))
           self.ClusterGraph.train_pos_edge_index=torch.tensor(np.array(HitCluster.GenerateLinks(_GenuineList,self.ClusterHitIDs)))

      def GiveStats(self,MCHits,cut_dt, cut_dr): #Decorate hit information
           import pandas as pd
           _MCClusterHits=[]
           StatFakeValues=[]
           StatTruthValues=[]
           StatLabels=['Initial # of combinations','Delete self-permutations','Enforce positive directionality','Cut on delta t', 'Cut on delta x','Track Reconstruction']
           for s in MCHits:
               if s[1]>=self.ClusterID[0]*self.Step[0] and s[1]<((self.ClusterID[0]+1)*self.Step[0]):
                   if s[2]>=self.ClusterID[1]*self.Step[1] and s[2]<((self.ClusterID[1]+1)*self.Step[1]):
                       if s[3]>=self.ClusterID[2]*self.Step[2] and s[3]<((self.ClusterID[2]+1)*self.Step[2]):
                          _MCClusterHits.append([s[0],s[6]])
           #Preparing Raw and MC combined data 1
           _l_MCHits=pd.DataFrame(_MCClusterHits, columns = ['l_HitID','l_MC_ID'])
           #Temp
           #_l_MCHits.to_csv('MC_Output.csv',index=False)

           _l_Hits=pd.DataFrame(self.ClusterHits, columns = ['l_HitID','l_x','l_y','l_z','l_tx','l_ty'])
           #Join hits + MC truth
           _l_Tot_Hits=pd.merge(_l_MCHits, _l_Hits, how="right", on=['l_HitID'])
           _l_Tot_Hits['join_key'] = 'join_key'

           #Preparing Raw and MC combined data 2
           _r_MCHits=pd.DataFrame(_MCClusterHits, columns = ['r_HitID','r_MC_ID'])
           _r_Hits=pd.DataFrame(self.ClusterHits, columns = ['r_HitID','r_x','r_y','r_z','r_tx','r_ty'])
           #Join hits + MC truth
           _r_Tot_Hits=pd.merge(_r_MCHits, _r_Hits, how="right", on=['r_HitID'])
           _r_Tot_Hits['join_key'] = 'join_key'
           #Combining data 1 and 2
           _Tot_Hits=pd.merge(_l_Tot_Hits, _r_Tot_Hits, how="inner", on=['join_key'])

           _Tot_Hits.l_MC_ID= _Tot_Hits.l_MC_ID.fillna(_Tot_Hits.l_HitID)
           _Tot_Hits.r_MC_ID= _Tot_Hits.r_MC_ID.fillna(_Tot_Hits.r_HitID)


           StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
           StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_HitID'] == _Tot_Hits['r_HitID']], inplace = True)
           StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
           StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_z'] <= _Tot_Hits['r_z']], inplace = True)
           StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
           StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

           #_Tot_Hits.to_csv('Stats_Crosslink_0.csv',index=False)
           _Tot_Hits['d_tx'] = _Tot_Hits['l_tx']-_Tot_Hits['r_tx']
           _Tot_Hits['d_tx'] = _Tot_Hits['d_tx'].abs()
           _Tot_Hits['d_ty'] = _Tot_Hits['l_ty']-_Tot_Hits['r_ty']
           _Tot_Hits['d_ty'] = _Tot_Hits['d_ty'].abs()
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_tx'] >= cut_dt], inplace = True)
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_ty'] >= cut_dt], inplace = True)
           #_Tot_Hits.to_csv('Stats_Crosslink_1.csv',index=False)
           StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
           StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

           _Tot_Hits['d_x'] = (_Tot_Hits['r_x']-(_Tot_Hits['l_x']+(_Tot_Hits['l_tx']*(_Tot_Hits['r_z']-_Tot_Hits['l_z']))))
           _Tot_Hits['d_x'] = _Tot_Hits['d_x'].abs()
           
           _Tot_Hits['d_y'] = (_Tot_Hits['r_y']-(_Tot_Hits['l_y']+(_Tot_Hits['l_ty']*(_Tot_Hits['r_z']-_Tot_Hits['l_z']))))
           _Tot_Hits['d_y'] = _Tot_Hits['d_y'].abs()

           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_x'] >= cut_dr], inplace = True)
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_y'] >= cut_dr], inplace = True)
           #_Tot_Hits.to_csv('Stats_Crosslink_2.csv',index=False)
           StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
           StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
           _Tot_Hits=_Tot_Hits[['l_HitID','r_HitID','r_z']]
           _Tot_Hits['link_strength']=1.0
           Trigger=False
           f_result=[]
           while(len(_Tot_Hits)>0):
                    _Tot_Hits_Pool=_Tot_Hits
                    z_ind=_Tot_Hits_Pool.sort_values(by=['r_z'], ascending=True)[['r_z']].drop_duplicates(subset=['r_z'],keep='first').values.tolist()
                    temp_s_hits=_Tot_Hits_Pool.drop(_Tot_Hits_Pool.index[_Tot_Hits_Pool['r_z'] != z_ind[0][0]])
                    temp_s_hits['Segment_0']=temp_s_hits['r_HitID']
                    temp_s_hits['Track_ID']=temp_s_hits['r_HitID']
                    temp_s_hits=temp_s_hits.rename(columns={"link_strength": "Fit"})
                    temp_s_hits=temp_s_hits.drop(["r_HitID",'r_z'], axis=1)
                    temp_s_hits=temp_s_hits.rename(columns={"l_HitID": "r_HitID" })
                    if len(z_ind)>1:
                        for zz in range(1,len(z_ind)):
                            temp_m_hits=_Tot_Hits_Pool.drop(_Tot_Hits_Pool.index[_Tot_Hits_Pool['r_z'] != z_ind[zz][0]])
                            temp_s_hits=pd.merge(temp_s_hits, temp_m_hits, how="left", on=['r_HitID'])
                            temp_s_hits['Segment_'+str(zz)]=temp_s_hits['r_HitID']
                            temp_s_hits.l_HitID= temp_s_hits.l_HitID.fillna(temp_s_hits.r_HitID)
                            temp_s_hits.link_strength= temp_s_hits.link_strength.fillna(0.0)
                            _Tot_Hits_Pool=pd.merge(_Tot_Hits_Pool, temp_s_hits[['r_HitID','l_HitID','Segment_'+str(zz)]], how="left", on=['r_HitID','l_HitID'])
                            _Tot_Hits_Pool=_Tot_Hits_Pool[_Tot_Hits_Pool['Segment_'+str(zz)].isnull()]
                            _Tot_Hits_Pool=_Tot_Hits_Pool.drop(['Segment_'+str(zz)], axis=1)
                            temp_s_hits['Track_ID']+=('-'+temp_s_hits['r_HitID'])
                            temp_s_hits['Fit']+=temp_s_hits['link_strength']
                            if zz==len(z_ind)-1:
                                temp_s_hits['Track_ID']+=('-'+temp_s_hits['l_HitID'])
                                temp_s_hits['Segment_'+str(zz+1)]=temp_s_hits['l_HitID']
                            temp_s_hits=temp_s_hits.drop(["r_HitID",'r_z','link_strength'], axis=1)
                            temp_s_hits=temp_s_hits.rename(columns={"l_HitID": "r_HitID" })
                            if zz==len(z_ind)-1:
                                temp_s_hits=temp_s_hits.drop(["r_HitID"], axis=1)
                    else:
                            temp_s_hits['Track_ID']+=('-'+temp_s_hits['r_HitID'])
                            temp_s_hits['Segment_1']=temp_s_hits['r_HitID']
                            temp_s_hits=temp_s_hits.drop(["r_HitID"], axis=1)
                    columns=[col for col in temp_s_hits.columns if 'Segment' in col]
                    t_count=0
                    for c1 in columns:
                        for c2 in columns:
                            if c1!=c2 and abs(columns.index(c1)-columns.index(c2))>1:
                                t_count+=1
                                t_temp_e_hits=pd.merge(temp_s_hits, _Tot_Hits_Pool[["r_HitID","l_HitID",'link_strength']], how="inner", left_on=[c1,c2], right_on=["r_HitID","l_HitID"])
                                if t_count==1:
                                    temp_e_hits=t_temp_e_hits
                                else:
                                    m_frames=[temp_e_hits,t_temp_e_hits]
                                    temp_e_hits=pd.concat(m_frames)
                    if t_count!=0:
                        temp_e_hits=temp_e_hits.drop_duplicates(subset=["r_HitID","l_HitID",'link_strength'],keep='first')[['Track_ID','link_strength',"r_HitID","l_HitID"]]
                        temp_e_hits=temp_e_hits.groupby(['Track_ID'])['link_strength'].sum().reset_index()
                        temp_e_hits=pd.merge(temp_s_hits, temp_e_hits, how="left", on=['Track_ID'])
                        temp_e_hits.link_strength= temp_e_hits.link_strength.fillna(0.0)
                        temp_dof_hits=temp_e_hits.drop(["Fit",'link_strength',], axis=1)
                        temp_dof_hits=pd.melt(temp_dof_hits, id_vars=['Track_ID'])
                        temp_dof_hits=temp_dof_hits.drop(["variable"], axis=1)
                        temp_dof_hits=temp_dof_hits.drop_duplicates(keep='first')
                        temp_dof_hits=temp_dof_hits.groupby(['Track_ID'])['value'].count().reset_index()
                        temp_dof_hits= temp_dof_hits.rename(columns={'value': "DoF"})
                        temp_e_hits=pd.merge(temp_e_hits, temp_dof_hits, how="inner", on=['Track_ID'])
                        temp_e_hits['Fit/DOF']=(temp_e_hits['link_strength']+temp_e_hits['Fit'])/(temp_e_hits['DoF']-1)
                        temp_e_hits=temp_e_hits.sort_values(by=['Fit/DOF'], ascending=False)
                        print(temp_e_hits)
                        input('Press to continue')
                        temp_e_hits=temp_e_hits.iloc[:1]
                        temp_e_hits=temp_e_hits.drop(["Fit",'DoF','link_strength','Fit/DOF'], axis=1)
                        temp_e_hits=pd.melt(temp_e_hits, id_vars=['Track_ID'])
                        temp_e_hits=temp_e_hits.drop(['variable'], axis=1)
                        temp_e_hits= temp_e_hits.rename(columns={'value': "HitID",'Track_ID': 'Segment_ID'})
                        temp_e_hits=temp_e_hits.drop_duplicates(keep='first')

                    else:
                         temp_e_hits=temp_s_hits.sort_values(by=['Fit'], ascending=False)
                         temp_e_hits=temp_e_hits.iloc[:1]
                         temp_e_hits=temp_e_hits.drop(["Fit"], axis=1)
                         temp_e_hits=pd.melt(temp_e_hits, id_vars=['Track_ID'])
                         temp_e_hits=temp_e_hits.drop(['variable'], axis=1)
                         temp_e_hits= temp_e_hits.rename(columns={'value': "HitID",'Track_ID': 'Segment_ID'})
                         temp_e_hits=temp_e_hits.drop_duplicates(keep='first')
                    if Trigger:
                            f_frames=[f_result,temp_e_hits]
                            f_result=pd.concat(f_frames)
                    else:
                            f_result=temp_e_hits
                            Trigger=True
                    _Tot_Hits=pd.merge(_Tot_Hits, temp_e_hits, how="left", left_on=['l_HitID'], right_on=['HitID'])
                    _Tot_Hits=_Tot_Hits[_Tot_Hits['Segment_ID'].isnull()]
                    _Tot_Hits=_Tot_Hits[['l_HitID','r_HitID','r_z','link_strength']]

                    _Tot_Hits=pd.merge(_Tot_Hits, temp_e_hits, how="left", left_on=['r_HitID'], right_on=['HitID'])
                    _Tot_Hits=_Tot_Hits[_Tot_Hits['Segment_ID'].isnull()]
                    _Tot_Hits=_Tot_Hits[['l_HitID','r_HitID','r_z','link_strength']]
                    _Tot_Hits=_Tot_Hits.drop_duplicates(keep='first')

                    if len(_Tot_Hits)==0:
                        break
           if len(f_result)>0:
               f_result_sl=f_result.groupby(by=['Segment_ID'])['HitID'].count().reset_index()
               f_result_sl=f_result_sl.rename(columns={"HitID": "Segment_Fit"})
               f_result=pd.merge(f_result, f_result_sl, how="inner", on=['Segment_ID'])
               f_result=f_result.sort_values(by=['HitID','Segment_Fit'], ascending=False)
               f_result=f_result.drop_duplicates(subset='HitID',keep='first')
               f_result=f_result[['HitID','Segment_ID']]
               _l_fHits= f_result.rename(columns={"HitID": "l_HitID"})
               _l_Tot_fHits=pd.merge(_l_MCHits, _l_fHits, how="left", on=['l_HitID'])
               _r_fHits= f_result.rename(columns={"HitID": "r_HitID"})

                #Join hits + MC truth
               _r_Tot_fHits=pd.merge(_r_MCHits, _r_fHits, how="right", on=['r_HitID'])
               _r_Tot_fHits.drop_duplicates(subset=['r_HitID'],keep='first', inplace=True)
               _l_Tot_fHits.drop_duplicates(subset=['l_HitID'],keep='first', inplace=True)
               _Tot_fHits=pd.merge(_l_Tot_fHits, _r_Tot_fHits, how="inner",on=["Segment_ID"])
               _Tot_fHits.l_MC_ID= _Tot_fHits.l_MC_ID.fillna(_Tot_fHits.l_HitID)
               _Tot_fHits.r_MC_ID= _Tot_fHits.r_MC_ID.fillna(_Tot_fHits.r_HitID)
               _Tot_fHits.drop(_Tot_fHits.index[_Tot_fHits['l_HitID'] == _Tot_fHits['r_HitID']], inplace = True)
               _Tot_fHits["Pair_ID"]= ['-'.join(sorted(tup)) for tup in zip(_Tot_fHits['l_HitID'], _Tot_fHits['r_HitID'])]
               _Tot_fHits.drop_duplicates(subset="Pair_ID",keep='first',inplace=True)
               StatFakeValues.append(len(_Tot_fHits.axes[0])-len(_Tot_fHits.drop(_Tot_fHits.index[_Tot_fHits['l_MC_ID'] != _Tot_fHits['r_MC_ID']]).axes[0]))
               StatTruthValues.append(len(_Tot_fHits.drop(_Tot_fHits.index[_Tot_fHits['l_MC_ID'] != _Tot_fHits['r_MC_ID']]).axes[0]))
           else:
               StatFakeValues.append(0)
               StatTruthValues.append(0)
           self.Stats=[StatLabels,StatFakeValues,StatTruthValues]
           #Temp
           f_result.to_csv('Stats_Crosslink_Final.csv',index=False)






      def LinkHits(self,hits,GiveStats,MCHits,cut_dt,cut_dr):
          self.HitLinks=hits
          _Map=[]
          for h in range(len(self.HitLinks[0])):
              _Map.append([self.ClusterHitIDs[self.HitLinks[0][h]],self.ClusterHitIDs[self.HitLinks[1][h]],self.HitLinks[2][h]])
              _Map.append([self.ClusterHitIDs[self.HitLinks[1][h]],self.ClusterHitIDs[self.HitLinks[0][h]],self.HitLinks[2][h]])
          import pandas as pd
          _Hits_df=pd.DataFrame(self.ClusterHits, columns = ['_l_HitID','x','y','z','tx','ty'])
          _Hits_df["x"] = pd.to_numeric(_Hits_df["x"],downcast='float')
          _Hits_df["y"] = pd.to_numeric(_Hits_df["y"],downcast='float')
          _Hits_df["z"] = pd.to_numeric(_Hits_df["z"],downcast='float')
          _Hits_df["tx"] = pd.to_numeric(_Hits_df["tx"],downcast='float')
          _Hits_df["ty"] = pd.to_numeric(_Hits_df["ty"],downcast='float')
          _Map_df=pd.DataFrame(_Map, columns = ['_l_HitID','_r_HitID','link_strength'])
          _Tot_Hits_df=pd.merge(_Hits_df, _Map_df, how="inner", on=['_l_HitID'])
          _Tot_Hits_df.drop_duplicates(subset=['_l_HitID','_r_HitID'],keep='first', inplace=True)
          #_Tot_Hits_df.drop(_Tot_Hits_df.index[_Tot_Hits_df['_l_HitID'] == _Tot_Hits_df['_r_HitID']], inplace = True)
          if GiveStats:
            _MCClusterHits=[]
            StatFakeValues=[]
            StatTruthValues=[]

            StatLabels=['Initial # of combinations','Delete self-permutations','Enforce positive directionality','Cut on delta t', 'Cut on delta x','Track Reconstruction']
            for s in MCHits:
               if s[1]>=self.ClusterID[0]*self.Step[0] and s[1]<((self.ClusterID[0]+1)*self.Step[0]):
                   if s[2]>=self.ClusterID[1]*self.Step[1] and s[2]<((self.ClusterID[1]+1)*self.Step[1]):
                       if s[3]>=self.ClusterID[2]*self.Step[2] and s[3]<((self.ClusterID[2]+1)*self.Step[2]):
                          _MCClusterHits.append([s[0],s[6]])
           #Preparing Raw and MC combined data 1
            _l_MCHits=pd.DataFrame(_MCClusterHits, columns = ['_l_HitID','l_MC_ID'])
            _r_MCHits=pd.DataFrame(_MCClusterHits, columns = ['_r_HitID','l_MC_ID'])
            _l_Hits=_Tot_Hits_df.rename(columns={"x": "l_x", "y": "l_y", "z": "l_z", "tx": "l_tx","ty": "l_ty","_r_HitID": "_link_HitID" })
            #Join hits + MC truth
            _l_Tot_Hits=pd.merge(_l_MCHits, _l_Hits, how="right", on=['_l_HitID'])
            #Preparing Raw and MC combined data 2
            _r_MCHits=pd.DataFrame(_MCClusterHits, columns = ['_r_HitID','r_MC_ID'])
            _r_Hits=_Tot_Hits_df[['_l_HitID', 'x', 'y', 'z', 'tx', 'ty']].rename(columns={"x": "r_x", "y": "r_y", "z": "r_z", "tx": "r_tx","ty": "r_ty","_l_HitID": "_r_HitID" })
            #Join hits + MC truth
            _r_Tot_Hits=pd.merge(_r_MCHits, _r_Hits, how="right", on=['_r_HitID'])
            _r_Tot_Hits.drop_duplicates(subset=['_r_HitID'],keep='first', inplace=True)

            _Tot_Hits=pd.merge(_l_Tot_Hits, _r_Tot_Hits, how="inner", left_on=["_link_HitID"], right_on=["_r_HitID"])
            _Tot_Hits.l_MC_ID= _Tot_Hits.l_MC_ID.fillna(_Tot_Hits._l_HitID)
            _Tot_Hits.r_MC_ID= _Tot_Hits.r_MC_ID.fillna(_Tot_Hits._r_HitID)
            #_Tot_Hits.to_csv('LinkStats_Crosslink_0.csv',index=False)
            StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['_l_HitID'] == _Tot_Hits['_r_HitID']], inplace = True)
            StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

            _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_z'] <= _Tot_Hits['r_z']], inplace = True)
            #_Tot_Hits.to_csv('LinkStats_Crosslink_1.csv',index=False)
            StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            _Tot_Hits['d_tx'] = _Tot_Hits['l_tx']-_Tot_Hits['r_tx']
            _Tot_Hits['d_tx'] = _Tot_Hits['d_tx'].abs()
            _Tot_Hits['d_ty'] = _Tot_Hits['l_ty']-_Tot_Hits['r_ty']
            _Tot_Hits['d_ty'] = _Tot_Hits['d_ty'].abs()
            _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_tx'] >= cut_dt], inplace = True)
            _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_ty'] >= cut_dt], inplace = True)

            StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

            _Tot_Hits['d_x'] = (_Tot_Hits['r_x']-(_Tot_Hits['l_x']+(_Tot_Hits['l_tx']*(_Tot_Hits['r_z']-_Tot_Hits['l_z']))))
            _Tot_Hits['d_x'] = _Tot_Hits['d_x'].abs()

            _Tot_Hits['d_y'] = (_Tot_Hits['r_y']-(_Tot_Hits['l_y']+(_Tot_Hits['l_ty']*(_Tot_Hits['r_z']-_Tot_Hits['l_z']))))
            _Tot_Hits['d_y'] = _Tot_Hits['d_y'].abs()

            _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_x'] >= cut_dr], inplace = True)
            _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['d_y'] >= cut_dr], inplace = True)
            #_Tot_Hits.to_csv('LinkStats_Crosslink_2.csv',index=False)
            StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            _Tot_Hits=_Tot_Hits[['_l_HitID','_r_HitID','r_z','link_strength']]
            Trigger=False
            f_result=[]
            while(len(_Tot_Hits)>0):
                    _Tot_Hits_Pool=_Tot_Hits

                    z_ind=_Tot_Hits_Pool.sort_values(by=['r_z'], ascending=True)[['r_z']].drop_duplicates(subset=['r_z'],keep='first').values.tolist()
                    temp_s_hits=_Tot_Hits_Pool.drop(_Tot_Hits_Pool.index[_Tot_Hits_Pool['r_z'] != z_ind[0][0]])
                    temp_s_hits['Segment_0']=temp_s_hits['_r_HitID']
                    temp_s_hits['Track_ID']=temp_s_hits['_r_HitID']
                    temp_s_hits=temp_s_hits.rename(columns={"link_strength": "Fit"})
                    temp_s_hits=temp_s_hits.drop(["_r_HitID",'r_z'], axis=1)
                    temp_s_hits=temp_s_hits.rename(columns={"_l_HitID": "_r_HitID" })
                    if len(z_ind)>1:
                        for zz in range(1,len(z_ind)):
                            temp_m_hits=_Tot_Hits_Pool.drop(_Tot_Hits_Pool.index[_Tot_Hits_Pool['r_z'] != z_ind[zz][0]])
                            temp_s_hits=pd.merge(temp_s_hits, temp_m_hits, how="left", on=['_r_HitID'])
                            temp_s_hits['Segment_'+str(zz)]=temp_s_hits['_r_HitID']
                            temp_s_hits._l_HitID= temp_s_hits._l_HitID.fillna(temp_s_hits._r_HitID)
                            temp_s_hits.link_strength= temp_s_hits.link_strength.fillna(0.0)
                            _Tot_Hits_Pool=pd.merge(_Tot_Hits_Pool, temp_s_hits[['_r_HitID','_l_HitID','Segment_'+str(zz)]], how="left", on=['_r_HitID','_l_HitID'])
                            _Tot_Hits_Pool=_Tot_Hits_Pool[_Tot_Hits_Pool['Segment_'+str(zz)].isnull()]
                            _Tot_Hits_Pool=_Tot_Hits_Pool.drop(['Segment_'+str(zz)], axis=1)
                            temp_s_hits['Track_ID']+=('-'+temp_s_hits['_r_HitID'])
                            temp_s_hits['Fit']+=temp_s_hits['link_strength']
                            if zz==len(z_ind)-1:
                                temp_s_hits['Track_ID']+=('-'+temp_s_hits['_l_HitID'])
                                temp_s_hits['Segment_'+str(zz+1)]=temp_s_hits['_l_HitID']
                            temp_s_hits=temp_s_hits.drop(["_r_HitID",'r_z','link_strength'], axis=1)
                            temp_s_hits=temp_s_hits.rename(columns={"_l_HitID": "_r_HitID" })
                            if zz==len(z_ind)-1:
                                temp_s_hits=temp_s_hits.drop(["_r_HitID"], axis=1)
                    else:
                            temp_s_hits['Track_ID']+=('-'+temp_s_hits['_r_HitID'])
                            temp_s_hits['Segment_1']=temp_s_hits['_r_HitID']
                            temp_s_hits=temp_s_hits.drop(["_r_HitID"], axis=1)
                    columns=[col for col in temp_s_hits.columns if 'Segment' in col]
                    t_count=0
                    for c1 in columns:
                        for c2 in columns:
                            if c1!=c2 and abs(columns.index(c1)-columns.index(c2))>1:
                                t_count+=1
                                t_temp_e_hits=pd.merge(temp_s_hits, _Tot_Hits_Pool[["_r_HitID","_l_HitID",'link_strength']], how="inner", left_on=[c1,c2], right_on=["_r_HitID","_l_HitID"])
                                if t_count==1:
                                    temp_e_hits=t_temp_e_hits
                                else:
                                    m_frames=[temp_e_hits,t_temp_e_hits]
                                    temp_e_hits=pd.concat(m_frames)
                    if t_count!=0:

                        temp_e_hits=temp_e_hits.drop_duplicates(subset=["_r_HitID","_l_HitID",'link_strength'],keep='first')[['Track_ID','link_strength',"_r_HitID","_l_HitID"]]

                        temp_e_hits=temp_e_hits.groupby(['Track_ID'])['link_strength'].sum().reset_index()
                        temp_e_hits=pd.merge(temp_s_hits, temp_e_hits, how="left", on=['Track_ID'])
                        temp_e_hits.link_strength= temp_e_hits.link_strength.fillna(0.0)
                        temp_dof_hits=temp_e_hits.drop(["Fit",'link_strength',], axis=1)
                        temp_dof_hits=pd.melt(temp_dof_hits, id_vars=['Track_ID'])
                        temp_dof_hits=temp_dof_hits.drop(["variable"], axis=1)
                        temp_dof_hits=temp_dof_hits.drop_duplicates(keep='first')
                        temp_dof_hits=temp_dof_hits.groupby(['Track_ID'])['value'].count().reset_index()
                        temp_dof_hits= temp_dof_hits.rename(columns={'value': "DoF"})
                        temp_e_hits=pd.merge(temp_e_hits, temp_dof_hits, how="inner", on=['Track_ID'])
                        temp_e_hits['Fit/DOF']=(temp_e_hits['link_strength']+temp_e_hits['Fit'])/(temp_e_hits['DoF']-1)
                        temp_e_hits=temp_e_hits.sort_values(by=['Fit/DOF'], ascending=False)
                        temp_e_hits=temp_e_hits.iloc[:1]
                        temp_e_hits=temp_e_hits.drop(["Fit",'DoF','link_strength','Fit/DOF'], axis=1)
                        temp_e_hits=pd.melt(temp_e_hits, id_vars=['Track_ID'])
                        temp_e_hits=temp_e_hits.drop(['variable'], axis=1)
                        temp_e_hits= temp_e_hits.rename(columns={'value': "HitID",'Track_ID': 'Segment_ID'})
                        temp_e_hits=temp_e_hits.drop_duplicates(keep='first')

                    else:
                         temp_e_hits=temp_s_hits.sort_values(by=['Fit'], ascending=False)
                         temp_e_hits=temp_e_hits.iloc[:1]
                         temp_e_hits=temp_e_hits.drop(["Fit"], axis=1)
                         temp_e_hits=pd.melt(temp_e_hits, id_vars=['Track_ID'])
                         temp_e_hits=temp_e_hits.drop(['variable'], axis=1)
                         temp_e_hits= temp_e_hits.rename(columns={'value': "HitID",'Track_ID': 'Segment_ID'})
                         temp_e_hits=temp_e_hits.drop_duplicates(keep='first')
                    if Trigger:
                            f_frames=[f_result,temp_e_hits]
                            f_result=pd.concat(f_frames)
                    else:
                            f_result=temp_e_hits
                            Trigger=True
                    _Tot_Hits=pd.merge(_Tot_Hits, temp_e_hits, how="left", left_on=['_l_HitID'], right_on=['HitID'])
                    _Tot_Hits=_Tot_Hits[_Tot_Hits['Segment_ID'].isnull()]
                    _Tot_Hits=_Tot_Hits[['_l_HitID','_r_HitID','r_z','link_strength']]

                    _Tot_Hits=pd.merge(_Tot_Hits, temp_e_hits, how="left", left_on=['_r_HitID'], right_on=['HitID'])
                    _Tot_Hits=_Tot_Hits[_Tot_Hits['Segment_ID'].isnull()]
                    _Tot_Hits=_Tot_Hits[['_l_HitID','_r_HitID','r_z','link_strength']]
                    _Tot_Hits=_Tot_Hits.drop_duplicates(keep='first')
                    if len(_Tot_Hits)==0:
                        break
            if len(f_result)>0:
                f_result_sl=f_result.groupby(by=['Segment_ID'])['HitID'].count().reset_index()
                f_result_sl=f_result_sl.rename(columns={"HitID": "Segment_Fit"})
                f_result=pd.merge(f_result, f_result_sl, how="inner", on=['Segment_ID'])
                f_result=f_result.sort_values(by=['HitID','Segment_Fit'], ascending=False)
                f_result=f_result.drop_duplicates(subset='HitID',keep='first')
                f_result=f_result[['HitID','Segment_ID']]
                _l_fHits= f_result.rename(columns={"HitID": "_l_HitID"})
                _l_Tot_fHits=pd.merge(_l_MCHits, _l_fHits, how="left", on=['_l_HitID'])
                _r_fHits= f_result.rename(columns={"HitID": "_r_HitID"})

                #Join hits + MC truth
                _r_Tot_fHits=pd.merge(_r_MCHits, _r_fHits, how="right", on=['_r_HitID'])
                _r_Tot_fHits.drop_duplicates(subset=['_r_HitID'],keep='first', inplace=True)
                _l_Tot_fHits.drop_duplicates(subset=['_l_HitID'],keep='first', inplace=True)
                _Tot_fHits=pd.merge(_l_Tot_fHits, _r_Tot_fHits, how="inner",on=["Segment_ID"])
                _Tot_fHits.l_MC_ID= _Tot_fHits.l_MC_ID.fillna(_Tot_fHits._l_HitID)
                _Tot_fHits.r_MC_ID= _Tot_fHits.r_MC_ID.fillna(_Tot_fHits._r_HitID)
                _Tot_fHits.drop(_Tot_fHits.index[_Tot_fHits['_l_HitID'] == _Tot_fHits['_r_HitID']], inplace = True)
                _Tot_fHits["Pair_ID"]= ['-'.join(sorted(tup)) for tup in zip(_Tot_fHits['_l_HitID'], _Tot_fHits['_r_HitID'])]
                _Tot_fHits.drop_duplicates(subset="Pair_ID",keep='first',inplace=True)
                StatFakeValues.append(len(_Tot_fHits.axes[0])-len(_Tot_fHits.drop(_Tot_fHits.index[_Tot_fHits['l_MC_ID'] != _Tot_fHits['r_MC_ID']]).axes[0]))
                StatTruthValues.append(len(_Tot_fHits.drop(_Tot_fHits.index[_Tot_fHits['l_MC_ID'] != _Tot_fHits['r_MC_ID']]).axes[0]))
            else:
               StatFakeValues.append(0)
               StatTruthValues.append(0)
            self.RecStats=[StatLabels,StatFakeValues,StatTruthValues]
            #print(self.RecStats)
            #f_result.to_csv('LinkStats_Crosslink_Final.csv',index=False)

            #Temp
            #f_result.to_csv('Link_Output.csv',index=False)

      def TestKalmanHits(self,FEDRAdata_list,MCdata_list):
          import pandas as pd
          _Tot_Hits_df=pd.DataFrame(self.ClusterHits, columns = ['HitID','x','y','z','tx','ty'])[['HitID','z']]
          _Tot_Hits_df["z"] = pd.to_numeric(_Tot_Hits_df["z"],downcast='float')

          _MCClusterHits=[]
          _FEDRAClusterHits=[]
          StatFakeValues=[]
          StatTruthValues=[]
          StatLabels=['Initial # of combinations','Delete self-permutations','Enforce positive directionality','Fedra Track Reconstruction']
          for s in MCdata_list:
             if s[1]>=self.ClusterID[0]*self.Step[0] and s[1]<((self.ClusterID[0]+1)*self.Step[0]):
                    if s[2]>=self.ClusterID[1]*self.Step[1] and s[2]<((self.ClusterID[1]+1)*self.Step[1]):
                        if s[3]>=self.ClusterID[2]*self.Step[2] and s[3]<((self.ClusterID[2]+1)*self.Step[2]):
                           _MCClusterHits.append([s[0],s[6]])
          for s in FEDRAdata_list:
             if s[1]>=self.ClusterID[0]*self.Step[0] and s[1]<((self.ClusterID[0]+1)*self.Step[0]):
                    if s[2]>=self.ClusterID[1]*self.Step[1] and s[2]<((self.ClusterID[1]+1)*self.Step[1]):
                        if s[3]>=self.ClusterID[2]*self.Step[2] and s[3]<((self.ClusterID[2]+1)*self.Step[2]):
                           _FEDRAClusterHits.append([s[0],s[6]])
          #Preparing Raw and MC combined data 1
          _l_MCHits=pd.DataFrame(_MCClusterHits, columns = ['l_HitID','l_MC_ID'])
          _r_MCHits=pd.DataFrame(_MCClusterHits, columns = ['r_HitID','r_MC_ID'])
          _l_FHits=pd.DataFrame(_FEDRAClusterHits, columns = ['l_HitID','l_FEDRA_ID'])
          _r_FHits=pd.DataFrame(_FEDRAClusterHits, columns = ['r_HitID','r_FEDRA_ID'])
          _l_Hits=_Tot_Hits_df.rename(columns={"z": "l_z","HitID": "l_HitID" })
          _r_Hits=_Tot_Hits_df.rename(columns={"z": "r_z","HitID": "r_HitID" })
          #Join hits + MC truth
          _l_Tot_Hits=pd.merge(_l_MCHits, _l_Hits, how="right", on=['l_HitID'])
          _r_Tot_Hits=pd.merge(_r_MCHits, _r_Hits, how="right", on=['r_HitID'])
          _l_Tot_Hits=pd.merge(_l_FHits, _l_Tot_Hits, how="right", on=['l_HitID'])
          _r_Tot_Hits=pd.merge(_r_FHits, _r_Tot_Hits, how="right", on=['r_HitID'])
          _l_Tot_Hits['join_key'] = 'join_key'
          _r_Tot_Hits['join_key'] = 'join_key'
          _Tot_Hits=pd.merge(_l_Tot_Hits, _r_Tot_Hits, how="inner", on=["join_key"])
          _Tot_Hits.l_MC_ID= _Tot_Hits.l_MC_ID.fillna(_Tot_Hits.l_HitID)
          _Tot_Hits.r_MC_ID= _Tot_Hits.r_MC_ID.fillna(_Tot_Hits.r_HitID)
          _Tot_Hits=_Tot_Hits.drop(['join_key'], axis=1)
          StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
          StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

          _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_HitID'] == _Tot_Hits['r_HitID']], inplace = True)
          StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
          StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

          _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_z'] <= _Tot_Hits['r_z']], inplace = True)
          StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
          StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

          _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['r_FEDRA_ID'] != _Tot_Hits['l_FEDRA_ID']], inplace = True)
          StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
          StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
          self.KalmanRecStats=[StatLabels,StatFakeValues,StatTruthValues]

      @staticmethod
      def GenerateLinks(_input,_ClusterID):
          _Top=[]
          _Bottom=[]
          for ip in _input:
              _Top.append(_ClusterID.index(ip[0]))
              _Bottom.append(_ClusterID.index(ip[1]))
          return [_Top,_Bottom]
      def GenerateEdgeAttributes(_input,_ClusterID):
          _Top=[]
          _Bottom=[]
          for ip in _input:
              _Top.append(_ClusterID.index(ip[0]))
              _Bottom.append(_ClusterID.index(ip[1]))
          return [_Top,_Bottom]


def CleanFolder(folder,key):
    if key=='':
      for the_file in os.listdir(folder):
                file_path=os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
    else:
      for the_file in os.listdir(folder):
                file_path=os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path) and (key in the_file):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
#This function automates csv read/write operations
def LogOperations(flocation,mode, message):
    if mode=='UpdateLog':
        csv_writer_log=open(flocation,"a")
        log_writer = csv.writer(csv_writer_log)
        if len(message)>0:
         for m in message:
          log_writer.writerow(m)
        csv_writer_log.close()
    if mode=='StartLog':
        csv_writer_log=open(flocation,"w")
        log_writer = csv.writer(csv_writer_log)
        if len(message)>0:
         for m in message:
           log_writer.writerow(m)
        csv_writer_log.close()
    if mode=='ReadLog':
        csv_reader_log=open(flocation,"r")
        log_reader = csv.reader(csv_reader_log)
        return list(log_reader)

def RecCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'EDER-GNN'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/REC_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def EvalCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'EDER-GNN'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/TEST_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def TrainCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'EDER-GNN'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/TRAIN_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      EOSsubModelDIR=EOSsubDIR+'/'+'Models'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def LoadRenderImages(Tracks,StartTrack,EndTrack):
    import tensorflow as tf
    from tensorflow import keras
    NewTracks=Tracks[StartTrack-1:min(EndTrack,len(Tracks))]
    ImagesY=np.empty([len(NewTracks),1])
    ImagesX=np.empty([len(NewTracks),NewTracks[0].H,NewTracks[0].W,NewTracks[0].L],dtype=np.bool)
    for im in range(len(NewTracks)):
        if hasattr(NewTracks[im],'MC_truth_label'):
           ImagesY[im]=int(float(NewTracks[im].MC_truth_label))
        else:
           ImagesY[im]=0
        BlankRenderedImage=[]
        for x in range(-NewTracks[im].bX,NewTracks[im].bX):
          for y in range(-NewTracks[im].bY,NewTracks[im].bY):
            for z in range(0,NewTracks[im].bZ):
             BlankRenderedImage.append(0)
        RenderedImage = np.array(BlankRenderedImage)
        RenderedImage = np.reshape(RenderedImage,(NewTracks[im].H,NewTracks[im].W,NewTracks[im].L))
        for Hits in NewTracks[im].TrackPrint:
                   RenderedImage[Hits[0]+NewTracks[im].bX][Hits[1]+NewTracks[im].bY][Hits[2]]=1
        ImagesX[im]=RenderedImage
    ImagesX= ImagesX[..., np.newaxis]
    ImagesY=tf.keras.utils.to_categorical(ImagesY,2)
    return (ImagesX,ImagesY)

def SubmitJobs2Condor(job):
    SHName = job[2]
    SUBName = job[3]
    if job[8]:
        MSGName=job[4]
    OptionLine = job[0][0]+str(job[1][0])
    for line in range(1,len(job[0])):
        OptionLine+=job[0][line]
        OptionLine+=str(job[1][line])
    f = open(SUBName, "w")
    f.write("executable = " + SHName)
    f.write("\n")
    if job[8]:
        f.write("output ="+MSGName+".out")
        f.write("\n")
        f.write("error ="+MSGName+".err")
        f.write("\n")
        f.write("log ="+MSGName+".log")
        f.write("\n")
    f.write('requirements = (CERNEnvironment =!= "qa")')
    f.write("\n")
    if job[9]:
        f.write('request_gpus = 1')
        f.write("\n")
    f.write('arguments = $(Process)')
    f.write("\n")
    f.write('+SoftUsed = '+'"'+job[7]+'"')
    f.write("\n")
    f.write('transfer_output_files = ""')
    f.write("\n")
    f.write('+JobFlavour = "workday"')
    f.write("\n")
    f.write('queue ' + str(job[6]))
    f.write("\n")
    f.close()
    TotalLine = 'python3 ' + job[5] + OptionLine
    f = open(SHName, "w")
    f.write("#!/bin/bash")
    f.write("\n")
    f.write("set -ux")
    f.write("\n")
    f.write(TotalLine)
    f.write("\n")
    f.close()
    subprocess.call(['condor_submit', SUBName])
    print(TotalLine, " has been successfully submitted")

def ErrorOperations(a,b,a_e,b_e,mode):
    if mode=='+' or mode == '-':
        c_e=math.sqrt((a_e**2) + (b_e**2))
        return(c_e)
    if mode=='*':
        c_e=a*b*math.sqrt(((a_e/a)**2) + ((b_e/b)**2))
        return(c_e)
    if mode=='/':
        c_e=(a/b)*math.sqrt(((a_e/a)**2) + ((b_e/b)**2))
        return(c_e)
