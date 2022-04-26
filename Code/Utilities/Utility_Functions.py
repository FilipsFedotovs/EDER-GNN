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
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_HitID'] == _Tot_Hits['r_HitID']], inplace = True)
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_z'] <= _Tot_Hits['r_z']], inplace = True)
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
      def GiveStats(self,MCHits,cut_dt, cut_dr): #Decorate hit information
           import pandas as pd
           _MCClusterHits=[]
           StatFakeValues=[]
           StatTruthValues=[]
           StatLabels=['Initial # of combinations','Delete self-permutations','Enforce positive directionality','Cut on delta t', 'Cut on delta x']
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

           StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
           StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_HitID'] == _Tot_Hits['r_HitID']], inplace = True)
           StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
           StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

           _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_z'] <= _Tot_Hits['r_z']], inplace = True)
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
           StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
           StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))


           self.Stats=[StatLabels,StatFakeValues,StatTruthValues]
      def GiveLinkStats(self,MCHits,cut_dt, cut_dr): #Decorate hit information
           import pandas as pd
           _MCClusterHits=[]
           StatFakeValues=[]
           StatTruthValues=[]
           StatLabels=['Initial # of combinations','Delete self-permutations','Enforce positive directionality','Cut on delta t', 'Cut on delta x']
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
           print(_l_Tot_Hits)
           exit()
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
           StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
           StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))


           self.Stats=[StatLabels,StatFakeValues,StatTruthValues]
      def LinkHits(self,hits,GiveStats,MCHits,MaxHits,cut_dt,cut_dr):
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

            StatLabels=['Initial # of combinations','Delete self-permutations','Enforce positive directionality','Cut on delta t', 'Cut on delta x']
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
            _l_Tot_Hits=pd.merge(_l_MCHits, _l_Hits, how="left", on=['_l_HitID'])
            #_l_Tot_Hits=_l_Tot_Hits.loc[_l_Tot_Hits['_l_HitID'] == '5532157']
            #Preparing Raw and MC combined data 2
            _r_MCHits=pd.DataFrame(_MCClusterHits, columns = ['_r_HitID','r_MC_ID'])
            _r_Hits=_Tot_Hits_df[['_l_HitID', 'x', 'y', 'z', 'tx', 'ty']].rename(columns={"x": "r_x", "y": "r_y", "z": "r_z", "tx": "r_tx","ty": "r_ty","_l_HitID": "_r_HitID" })
            #Join hits + MC truth
            _r_Tot_Hits=pd.merge(_r_MCHits, _r_Hits, how="right", on=['_r_HitID'])
            _r_Tot_Hits.drop_duplicates(subset=['_r_HitID'],keep='first', inplace=True)

            _Tot_Hits=pd.merge(_l_Tot_Hits, _r_Tot_Hits, how="inner", left_on=["_link_HitID"], right_on=["_r_HitID"])
            _Tot_Hits.l_MC_ID= _Tot_Hits.l_MC_ID.fillna(_Tot_Hits._l_HitID)
            _Tot_Hits.r_MC_ID= _Tot_Hits.r_MC_ID.fillna(_Tot_Hits._r_HitID)

            StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['_l_HitID'] == _Tot_Hits['_r_HitID']], inplace = True)
            StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

            _Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_z'] <= _Tot_Hits['r_z']], inplace = True)
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

            StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            _Tot_Hits=_Tot_Hits.sort_values(by=['_r_HitID','l_z','_l_HitID','link_strength'], ascending=False)
            _Tot_Hits.drop_duplicates(subset=['_r_HitID','l_z'],keep='first', inplace=True)
            StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            print(StatFakeValues)
            print(StatTruthValues)
            _Tot_Hits=_Tot_Hits[['_l_HitID','_r_HitID','r_z','l_MC_ID','r_MC_ID']]
            print(_Tot_Hits)
            z_ind=_Tot_Hits.sort_values(by=['r_z'], ascending=True)[['r_z']].drop_duplicates(subset=['r_z'],keep='first').values.tolist()
            print(z_ind)
            for z in z_ind:
                temp_s_hits=_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['r_z'] != z[0]])
                print(temp_s_hits)
                exit()
            exit()


      @staticmethod
      def GenerateLinks(_input,_ClusterID):
          _Top=[]
          _Bottom=[]
          for ip in _input:
              _Top.append(_ClusterID.index(ip[0]))
              _Bottom.append(_ClusterID.index(ip[1]))
          return [_Top,_Bottom]
class Track:
      def __init__(self,segments):
          self.SegmentHeader=sorted(segments, key=str.lower)
          self.Segmentation=len(self.SegmentHeader)
      def __eq__(self, other):
        return ('-'.join(self.SegmentHeader)) == ('-'.join(other.SegmentHeader))
      def __hash__(self):
        return hash(('-'.join(self.SegmentHeader)))
      def DecorateSegments(self,RawHits): #Decorate hit information
          self.SegmentHits=[]

          for s in range(len(self.SegmentHeader)):
              self.SegmentHits.append([])
              for t in RawHits:
                   if self.SegmentHeader[s]==t[3]:
                      self.SegmentHits[s].append(t[:3])
          for Hit in range(0, len(self.SegmentHits)):
             self.SegmentHits[Hit]=sorted(self.SegmentHits[Hit],key=lambda x: float(x[2]),reverse=False)
          import torch
          __ClusterNodes=torch.Tensor(self.ClusterHits)

      def DecorateTrackGeoInfo(self):
          if hasattr(self,'SegmentHits'):
             if self.Segmentation==2:
                __XZ1=Track.GetEquationOfTrack(self.SegmentHits[0])[0]
                __XZ2=Track.GetEquationOfTrack(self.SegmentHits[1])[0]
                __YZ1=Track.GetEquationOfTrack(self.SegmentHits[0])[1]
                __YZ2=Track.GetEquationOfTrack(self.SegmentHits[1])[1]
                __X1S=Track.GetEquationOfTrack(self.SegmentHits[0])[3]
                __X2S=Track.GetEquationOfTrack(self.SegmentHits[1])[3]
                __Y1S=Track.GetEquationOfTrack(self.SegmentHits[0])[4]
                __Y2S=Track.GetEquationOfTrack(self.SegmentHits[1])[4]
                __Z1S=Track.GetEquationOfTrack(self.SegmentHits[0])[5]
                __Z2S=Track.GetEquationOfTrack(self.SegmentHits[1])[5]
                __vector_1_st = np.array([np.polyval(__XZ1,self.SegmentHits[0][0][2]),np.polyval(__YZ1,self.SegmentHits[0][0][2]),self.SegmentHits[0][0][2]])
                __vector_1_end = np.array([np.polyval(__XZ1,self.SegmentHits[0][len(self.SegmentHits[0])-1][2]),np.polyval(__YZ1,self.SegmentHits[0][len(self.SegmentHits[0])-1][2]),self.SegmentHits[0][len(self.SegmentHits[0])-1][2]])
                __vector_2_st = np.array([np.polyval(__XZ2,self.SegmentHits[0][0][2]),np.polyval(__YZ2,self.SegmentHits[0][0][2]),self.SegmentHits[0][0][2]])
                __vector_2_end = np.array([np.polyval(__XZ2,self.SegmentHits[0][len(self.SegmentHits[0])-1][2]),np.polyval(__YZ2,self.SegmentHits[0][len(self.SegmentHits[0])-1][2]),self.SegmentHits[0][len(self.SegmentHits[0])-1][2]])
                __result=Track.closestDistanceBetweenLines(__vector_1_st,__vector_1_end,__vector_2_st,__vector_2_end,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False)
                __midpoint=(__result[0]+__result[1])/2
                __v1=np.subtract(__vector_1_end,__midpoint)
                __v2=np.subtract(__vector_2_end,__midpoint)
                if self.SegmentHits[0][len(self.SegmentHits)-1][2]>self.SegmentHits[1][len(self.SegmentHits)-1][2]: #Workout which track is leading (has highest z-coordinate)
                    __leading_seg=0
                    __subleading_seg=1
                else:
                    __leading_seg=1
                    __subleading_seg=0
                self.angle=Track.angle_between(__v1, __v2)
                self.DOCA=__result[2]
                self.Seg_Lon_Gap=float(self.SegmentHits[__leading_seg][0][2])-float(self.SegmentHits[__subleading_seg][len(self.SegmentHits[__subleading_seg])-1][2])
                __x2=float(self.SegmentHits[__leading_seg][0][0])
                __x1=self.SegmentHits[__subleading_seg][len(self.SegmentHits[__subleading_seg])-1][0]
                __y2=float(self.SegmentHits[__leading_seg][0][1])
                __y1=self.SegmentHits[__subleading_seg][len(self.SegmentHits[__subleading_seg])-1][1]
                self.Seg_Transv_Gap=math.sqrt(((__x2-__x1)**2)+((__y2-__y1)**2))
             else:
                 raise ValueError("Method 'DecorateTrackGeoInfo' currently works for track segment combinations with number of segments of 2 only")
          else:
                raise ValueError("Method 'DecorateTrackGeoInfo' works only if 'DecorateTracks' method has been acted upon the seed before")

      def TrackQualityCheck(self,MaxDoca,MaxSLG, MaxSTG,MaxAngle):
                    self.GeoFit = (self.DOCA<=MaxDoca and self.Seg_Lon_Gap<=MaxSLG and self.Seg_Transv_Gap<=(MaxSTG+(self.Seg_Lon_Gap*0.96)) and abs(self.angle)<=MaxAngle)

      def CNNFitTrack(self,Prediction):
          self.Track_CNN_Fit=Prediction
          self.TR_CNN_FIT=[Prediction,Prediction]

      def AssignCNNTrId(self,ID):
          self.TR_CNN_ID=ID

      def InjectTrack(self,OtherTrack):
          self_matx=Track.DensityMatrix(OtherTrack.SegmentHeader,self.SegmentHeader)
          if Track.Overlap(self_matx)==False:
              return Track.Overlap(self_matx)
          new_seed_header=Track.ProjectVectorElements(self_matx,self.SegmentHeader)
          new_self_hits=Track.ProjectVectorElements(self_matx,self.SegmentHits)
          new_self_fit=Track.ProjectVectorElements(self_matx,self.TR_CNN_FIT)
          remain_1_s = Track.GenerateInverseVector(self.SegmentHeader,new_seed_header)
          remain_1_o = Track.GenerateInverseVector(OtherTrack.SegmentHeader,new_seed_header)
          OtherTrack.SegmentHeader=Track.ProjectVectorElements([remain_1_o],OtherTrack.SegmentHeader)
          self.SegmentHeader=Track.ProjectVectorElements([remain_1_s],self.SegmentHeader)
          OtherTrack.SegmentHits=Track.ProjectVectorElements([remain_1_o],OtherTrack.SegmentHits)
          self.SegmentHits=Track.ProjectVectorElements([remain_1_s],self.SegmentHits)
          OtherTrack.TR_CNN_FIT=Track.ProjectVectorElements([remain_1_o],OtherTrack.TR_CNN_FIT)
          self.TR_CNN_FIT=Track.ProjectVectorElements([remain_1_s],self.TR_CNN_FIT)
          # print('1',self.SegmentHeader,OtherTrack.SegmentHeader)
          # print('1',self.SegmentHits,OtherTrack.SegmentHits)
          # print('1',self.TR_CNN_FIT,OtherTrack.TR_CNN_FIT)
          # print('1',new_seed_header,new_self_hits,new_self_fit)
          # print('1',self_matx)
          if (len(OtherTrack.SegmentHeader))==0:
              self.SegmentHeader+=new_seed_header
              self.SegmentHits+=new_self_hits
              self.TR_CNN_FIT+=new_self_fit
              self.Track_CNN_Fit=sum(self.TR_CNN_FIT)/len(self.TR_CNN_FIT)
              self.Segmentation=len(self.SegmentHeader)
              if len(self.TR_CNN_FIT)!=len(self.SegmentHeader):
                  raise Exception('Fit error')
                  exit()

              return True
          if (len(self.SegmentHeader))==0:
              self.SegmentHeader+=new_seed_header
              self.SegmentHits+=new_self_hits
              self.TR_CNN_FIT+=new_self_fit
              self.SegmentHeader+=OtherTrack.SegmentHeader
              self.SegmentHits+=OtherTrack.SegmentHits
              self.TR_CNN_FIT+=OtherTrack.TR_CNN_FIT
              self.Track_CNN_Fit=sum(self.TR_CNN_FIT)/len(self.TR_CNN_FIT)
              self.Segmentation=len(self.SegmentHeader)
              if len(self.TR_CNN_FIT)!=len(self.SegmentHeader):
                  raise Exception('Fit error')
                  exit()
              return True
          self_2_matx=Track.DensityMatrix(OtherTrack.SegmentHits,self.SegmentHits)
          other_2_matx=Track.DensityMatrix(self.SegmentHits,OtherTrack.SegmentHits)
          #print('Test',self_2_matx,self_2_matx)
          last_s_seed_header=Track.ProjectVectorElements(self_2_matx,self.SegmentHeader)
          last_o_seed_header=Track.ProjectVectorElements(other_2_matx,OtherTrack.SegmentHeader)
          remain_2_s = Track.GenerateInverseVector(self.SegmentHeader,last_s_seed_header)
          remain_2_o = Track.GenerateInverseVector(OtherTrack.SegmentHeader,last_o_seed_header)

          new_seed_header+=Track.ProjectVectorElements([remain_2_s],self.SegmentHeader)
          new_seed_header+=Track.ProjectVectorElements([remain_2_o],OtherTrack.SegmentHeader)
          new_self_fit+=Track.ProjectVectorElements([remain_2_s],self.TR_CNN_FIT)
          new_self_fit+=Track.ProjectVectorElements([remain_2_o],OtherTrack.TR_CNN_FIT)
          new_self_hits+=Track.ProjectVectorElements([remain_2_s],self.SegmentHits)
          new_self_hits+=Track.ProjectVectorElements([remain_2_o],OtherTrack.SegmentHits)


          last_remain_headers_s = Track.GenerateInverseVector(self.SegmentHeader,new_seed_header)
          last_remain_headers_o = Track.GenerateInverseVector(OtherTrack.SegmentHeader,new_seed_header)
          last_self_headers=Track.ProjectVectorElements([last_remain_headers_s],self.SegmentHeader)
          last_other_headers=Track.ProjectVectorElements([last_remain_headers_o],OtherTrack.SegmentHeader)
          # print('2',self.SegmentHeader,OtherTrack.SegmentHeader)
          # print('2',self.SegmentHits,OtherTrack.SegmentHits)
          # print('2',self.TR_CNN_FIT,OtherTrack.TR_CNN_FIT)
          # print('2',new_seed_header,new_self_hits,new_self_fit)
          # print('2',self_2_matx)
          # print('2',last_self_headers,last_other_headers)
          if (len(last_other_headers))==0:
              self.SegmentHeader=new_seed_header
              self.SegmentHits=new_self_hits
              self.TR_CNN_FIT=new_self_fit
              self.Track_CNN_Fit=sum(self.TR_CNN_FIT)/len(self.TR_CNN_FIT)
              self.Segmentation=len(self.SegmentHeader)
              if len(self.TR_CNN_FIT)!=len(self.SegmentHeader):
                  raise Exception('Fit error')
                  exit()
              return True

          last_self_hits=Track.ProjectVectorElements([last_remain_headers_s],self.SegmentHits)
          last_other_hits=Track.ProjectVectorElements([last_remain_headers_o],OtherTrack.SegmentHits)
          last_self_fits=Track.ProjectVectorElements([last_remain_headers_s],self.TR_CNN_FIT)
          last_other_fits=Track.ProjectVectorElements([last_remain_headers_o],OtherTrack.TR_CNN_FIT)
          last_remain_matr=Track.DensityMatrix(last_other_hits,last_self_hits)

          new_seed_header+=Track.ReplaceWeakerTracks(last_remain_matr,last_other_headers,last_self_headers,last_other_fits,last_self_fits)
          new_self_fit+=Track.ReplaceWeakerFits(new_seed_header,last_self_headers,last_other_headers,last_other_fits,last_self_fits)[0:len(Track.ReplaceWeakerFits(new_seed_header,last_self_headers,last_other_headers,last_other_fits,last_self_fits))]
          new_self_hits+=Track.ReplaceWeakerTracks(last_remain_matr,last_other_hits,last_self_hits,last_other_fits,last_self_fits)
          # print('3',self.SegmentHeader,OtherTrack.SegmentHeader)
          # print('3',self.SegmentHits,OtherTrack.SegmentHits)
          # print('3',self.TR_CNN_FIT,OtherTrack.TR_CNN_FIT)
          # print('3',new_seed_header,new_self_hits,new_self_fit)
          # print('3',last_remain_matr)
          # print('3',last_self_headers,last_other_headers,last_self_fits,last_other_fits)
          # print('check',Track.ReplaceWeakerTracks(last_remain_matr,last_other_fits,last_self_fits,last_other_fits,last_self_fits))
          self.SegmentHeader=new_seed_header
          self.SegmentHits=new_self_hits
          self.TR_CNN_FIT=new_self_fit
          self.Track_CNN_Fit=sum(self.TR_CNN_FIT)/len(self.TR_CNN_FIT)
          self.Segmentation=len(self.SegmentHeader)
          if len(self.TR_CNN_FIT)!=len(self.SegmentHeader):
                  raise Exception('Fit error')
                  exit()

          return True

      def MCtruthClassifyTrack(self,label):
          self.MC_truth_label=label

      def PrepareTrackPrint(self,MaxX,MaxY,MaxZ,Res,Rescale):
          __TempTrack=copy.deepcopy(self.SegmentHits)
          self.Resolution=Res
          self.bX=int(round(MaxX/self.Resolution,0))
          self.bY=int(round(MaxY/self.Resolution,0))
          self.bZ=int(round(MaxZ/self.Resolution,0))
          self.H=(self.bX)*2
          self.W=(self.bY)*2
          self.L=(self.bZ)
          __StartTrackZ=6666666666
          __EndTrackZ=-6666666666
          for __Track in __TempTrack:
            __CurrentZ=float(__Track[0][2])
            if __CurrentZ<=__StartTrackZ:
                __StartTrackZ=__CurrentZ
                __FinX=float(__Track[0][0])
                __FinY=float(__Track[0][1])
                __FinZ=float(__Track[0][2])
                self.PrecedingTrackInd=__TempTrack.index(__Track)
            if __CurrentZ>=__EndTrackZ:
                __EndTrackZ=__CurrentZ
                self.LagTrackInd=__TempTrack.index(__Track)
          for __Tracks in __TempTrack:
              for __Hits in __Tracks:
                  __Hits[0]=float(__Hits[0])-__FinX
                  __Hits[1]=float(__Hits[1])-__FinY
                  __Hits[2]=float(__Hits[2])-__FinZ

          #
          #Lon Rotate x
          __Track=__TempTrack[self.LagTrackInd]
          __Vardiff=float(__Track[len(__Track)-1][0])
          __Zdiff=float(__Track[len(__Track)-1][2])
          __vector_1 = [__Zdiff, 0]
          __vector_2 = [__Zdiff, __Vardiff]
          __Angle=Track.angle_between(__vector_1, __vector_2)
          if np.isnan(__Angle)==True:
                    __Angle=0.0
          for __Tracks in __TempTrack:
            for __hits in __Tracks:
                 __Z=float(__hits[2])
                 __Pos=float(__hits[0])
                 __hits[2]=(__Z*math.cos(-__Angle)) - (__Pos * math.sin(-__Angle))
                 __hits[0]=(__Z*math.sin(-__Angle)) + (__Pos * math.cos(-__Angle))
          #Lon Rotate y
          __Track=__TempTrack[self.LagTrackInd]
          __Vardiff=float(__Track[len(__Track)-1][1])
          __Zdiff=float(__Track[len(__Track)-1][2])
          __vector_1 = [__Zdiff, 0]
          __vector_2 = [__Zdiff, __Vardiff]
          __Angle=Track.angle_between(__vector_1, __vector_2)
          if np.isnan(__Angle)==True:
                     __Angle=0.0
          for __Tracks in __TempTrack:
            for __hits in __Tracks:
                 __Z=float(__hits[2])
                 __Pos=float(__hits[1])
                 __hits[2]=(__Z*math.cos(-__Angle)) - (__Pos * math.sin(-__Angle))
                 __hits[1]=(__Z*math.sin(-__Angle)) + (__Pos * math.cos(-__Angle))
           #Phi rotate print

          __LongestDistance=0.0
          for __Track in __TempTrack:
                 __X=float(__Track[len(__Track)-1][0])
                 __Y=float(__Track[len(__Track)-1][1])
                 __Distance=math.sqrt((__X**2)+(__Y**2))
                 if __Distance>=__LongestDistance:
                  __LongestDistance=__Distance
                  __vector_1 = [__Distance, 0]
                  __vector_2 = [__X, __Y]
                  __Angle=-Track.angle_between(__vector_1,__vector_2)
          if np.isnan(__Angle)==True:
                     __Angle=0.0
          for __Tracks in __TempTrack:
             for __hits in __Tracks:
                 __X=float(__hits[0])
                 __Y=float(__hits[1])
                 __hits[0]=(__X*math.cos(__Angle)) - (__Y * math.sin(__Angle))
                 __hits[1]=(__X*math.sin(__Angle)) + (__Y * math.cos(__Angle))

          if Rescale:
               __X=[]
               __Y=[]
               __Z=[]
               for __Tracks in __TempTrack:
                 for __hits in __Tracks:
                     __X.append(__hits[0])
                     __Y.append(__hits[1])
                     __Z.append(__hits[2])
               __dUpX=MaxX-max(__X)
               __dDownX=MaxX+min(__X)
               __dX=(__dUpX+__dDownX)/2
               __xshift=__dUpX-__dX
               __X=[]
               for __Tracks in __TempTrack:
                 for __hits in __Tracks:
                     __hits[0]=__hits[0]+__xshift
                     __X.append(__hits[0])
             ##########Y
               __dUpY=MaxY-max(__Y)
               __dDownY=MaxY+min(__Y)
               __dY=(__dUpY+__dDownY)/2
               __yshift=__dUpY-__dY
               __Y=[]
               for __Tracks in __TempTrack:
                 for __hits in __Tracks:
                     __hits[1]=__hits[1]+__yshift
                     __Y.append(__hits[1])
               __min_scale=max(max(__X)/(MaxX-(2*self.Resolution)),max(__Y)/(MaxY-(2*self.Resolution)), max(__Z)/(MaxZ-(2*self.Resolution)))
               for __Tracks in __TempTrack:
                 for __hits in __Tracks:
                     __hits[0]=int(round(__hits[0]/__min_scale,0))
                     __hits[1]=int(round(__hits[1]/__min_scale,0))
                     __hits[2]=int(round(__hits[2]/__min_scale,0))

          #
          #Enchance track
          __TempEnchTrack=[]
          for __Tracks in __TempTrack:
               for h in range(0,len(__Tracks)-1):
                   __deltaX=float(__Tracks[h+1][0])-float(__Tracks[h][0])
                   __deltaZ=float(__Tracks[h+1][2])-float(__Tracks[h][2])
                   __deltaY=float(__Tracks[h+1][1])-float(__Tracks[h][1])
                   try:
                    __vector_1 = [__deltaZ,0]
                    __vector_2 = [__deltaZ, __deltaX]
                    __ThetaAngle=Track.angle_between(__vector_1, __vector_2)
                   except:
                     __ThetaAngle=0.0
                   try:
                     __vector_1 = [__deltaZ,0]
                     __vector_2 = [__deltaZ, __deltaY]
                     __PhiAngle=Track.angle_between(__vector_1, __vector_2)
                   except:
                     __PhiAngle=0.0
                   __TotalDistance=math.sqrt((__deltaX**2)+(__deltaY**2)+(__deltaZ**2))
                   __Distance=(float(self.Resolution)/3)
                   if __Distance>=0 and __Distance<1:
                      __Distance=1.0
                   if __Distance<0 and __Distance>-1:
                      __Distance=-1.0
                   __Iterations=int(round(__TotalDistance/__Distance,0))
                   for i in range(1,__Iterations):
                       __New_Hit=[]
                       if math.isnan(float(__Tracks[h][0])+float(i)*__Distance*math.sin(__ThetaAngle)):
                          continue
                       if math.isnan(float(__Tracks[h][1])+float(i)*__Distance*math.sin(__PhiAngle)):
                          continue
                       if math.isnan(float(__Tracks[h][2])+float(i)*__Distance*math.cos(__ThetaAngle)):
                          continue
                       __New_Hit.append(float(__Tracks[h][0])+float(i)*__Distance*math.sin(__ThetaAngle))
                       __New_Hit.append(float(__Tracks[h][1])+float(i)*__Distance*math.sin(__PhiAngle))
                       __New_Hit.append(float(__Tracks[h][2])+float(i)*__Distance*math.cos(__ThetaAngle))
                       __TempEnchTrack.append(__New_Hit)
          #
          # #Pixelise print

          self.TrackPrint=[]
          for __Tracks in __TempTrack:
               for __Hits in __Tracks:
                   __Hits[0]=int(round(float(__Hits[0])/self.Resolution,0))
                   __Hits[1]=int(round(float(__Hits[1])/self.Resolution,0))
                   __Hits[2]=int(round(float(__Hits[2])/self.Resolution,0))
                   self.TrackPrint.append(str(__Hits))
          for __Hits in __TempEnchTrack:
                   __Hits[0]=int(round(float(__Hits[0])/self.Resolution,0))
                   __Hits[1]=int(round(float(__Hits[1])/self.Resolution,0))
                   __Hits[2]=int(round(float(__Hits[2])/self.Resolution,0))
                   self.TrackPrint.append(str(__Hits))
          self.TrackPrint=list(set(self.TrackPrint))
          for p in range(len(self.TrackPrint)):
               self.TrackPrint[p]=ast.literal_eval(self.TrackPrint[p])
          self.TrackPrint=[p for p in self.TrackPrint if (abs(p[0])<self.bX and abs(p[1])<self.bY and abs(p[2])<self.bZ)]
          del __TempEnchTrack
          del __TempTrack

      def UnloadTrackPrint(self):
          delattr(self,'TrackPrint')
          delattr(self,'bX')
          delattr(self,'bY')
          delattr(self,'bZ')
          delattr(self,'H')
          delattr(self,'W')
          delattr(self,'L')
          delattr(self,'LagTrackInd')
          delattr(self,'PrecedingTrackInd')


      def Plot(self,PlotType):
        if PlotType=='XZ' or PlotType=='ZX':
          __InitialData=[]
          __Index=-1
          for x in range(-self.bX,self.bX):
             for z in range(0,self.bZ):
                 __InitialData.append(0.0)
          __Matrix = np.array(__InitialData)
          __Matrix=np.reshape(__Matrix,(self.H,self.L))
          for __Hits in self.TrackPrint:
                   __Matrix[int(__Hits[0])+self.bX][int(__Hits[2])]=1
          import matplotlib as plt
          from matplotlib.colors import LogNorm
          from matplotlib import pyplot as plt
          plt.title('Track '+':'.join(self.SegmentHeader))
          plt.xlabel('Z [microns /'+str(int(self.Resolution))+']')
          plt.ylabel('X [microns /'+str(int(self.Resolution))+']')
          __image=plt.imshow(__Matrix,cmap='gray_r',extent=[0,self.bZ,self.bX,-self.bX])
          plt.gca().invert_yaxis()
          plt.show()
        elif PlotType=='YZ' or PlotType=='ZY':
          __InitialData=[]
          __Index=-1
          for y in range(-self.bY,self.bY):
             for z in range(0,self.bZ):
                 __InitialData.append(0.0)
          __Matrix = np.array(__InitialData)
          __Matrix=np.reshape(__Matrix,(self.W,self.L))
          for __Hits in self.TrackPrint:
                   __Matrix[int(__Hits[1])+self.bY][int(__Hits[2])]=1
          import matplotlib as plt
          from matplotlib.colors import LogNorm
          from matplotlib import pyplot as plt
          plt.title('Track '+':'.join(self.SegmentHeader))
          plt.xlabel('Z [microns /'+str(int(self.Resolution))+']')
          plt.ylabel('Y [microns /'+str(int(self.Resolution))+']')
          __image=plt.imshow(__Matrix,cmap='gray_r',extent=[0,self.bZ,self.bY,-self.bY])
          plt.gca().invert_yaxis()
          plt.show()
        elif PlotType=='XY' or PlotType=='YX':
          __InitialData=[]
          __Index=-1
          for x in range(-self.bX,self.bX):
             for y in range(-self.bY,self.bY):
                 __InitialData.append(0.0)
          __Matrix = np.array(__InitialData)
          __Matrix=np.reshape(__Matrix,(self.H,self.W))
          for __Hits in self.TrackPrint:
                   __Matrix[int(__Hits[0])+self.bX][int(__Hits[1]+self.bY)]=1
          import matplotlib as plt
          from matplotlib.colors import LogNorm
          from matplotlib import pyplot as plt
          plt.title('Track '+':'.join(self.SegmentHeader))
          plt.xlabel('Y [microns /'+str(int(self.Resolution))+']')
          plt.ylabel('X [microns /'+str(int(self.Resolution))+']')
          __image=plt.imshow(__Matrix,cmap='gray_r',extent=[-self.bY,self.bY,self.bX,-self.bX])
          plt.gca().invert_yaxis()
          plt.show()
        else:
          print('Invalid plot type input value! Should be XZ, YZ or XY')


      @staticmethod
      def unit_vector(vector):
          return vector / np.linalg.norm(vector)

      def angle_between(v1, v2):
            v1_u = Track.unit_vector(v1)
            v2_u = Track.unit_vector(v2)
            dot = v1_u[0]*v2_u[0] + v1_u[1]*v2_u[1]      # dot product
            det = v1_u[0]*v2_u[1] - v1_u[1]*v2_u[0]      # determinant
            return np.arctan2(det, dot)

      def GetEquationOfTrack(Track):
          Xval=[]
          Yval=[]
          Zval=[]
          for Hits in Track:
              Xval.append(Hits[0])
              Yval.append(Hits[1])
              Zval.append(Hits[2])
          XZ=np.polyfit(Zval,Xval,1)
          YZ=np.polyfit(Zval,Yval,1)
          return (XZ,YZ, 'N/A',Xval[0],Yval[0],Zval[0])

      def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):
            a0=np.array(a0)
            a1=np.array(a1)
            b0=np.array(b0)
            b1=np.array(b1)
            # If clampAll=True, set all clamps to True
            if clampAll:
                clampA0=True
                clampA1=True
                clampB0=True
                clampB1=True


            # Calculate denomitator
            A = a1 - a0
            B = b1 - b0
            magA = np.linalg.norm(A)
            magB = np.linalg.norm(B)

            _A = A / magA
            _B = B / magB

            cross = np.cross(_A, _B);
            denom = np.linalg.norm(cross)**2


            # If lines are parallel (denom=0) test if lines overlap.
            # If they don't overlap then there is a closest point solution.
            # If they do overlap, there are infinite closest positions, but there is a closest distance
            if not denom:
                d0 = np.dot(_A,(b0-a0))

                # Overlap only possible with clamping
                if clampA0 or clampA1 or clampB0 or clampB1:
                    d1 = np.dot(_A,(b1-a0))

                    # Is segment B before A?
                    if d0 <= 0 >= d1:
                        if clampA0 and clampB1:
                            if np.absolute(d0) < np.absolute(d1):
                                return a0,b0,np.linalg.norm(a0-b0)
                            return a0,b1,np.linalg.norm(a0-b1)


                    # Is segment B after A?
                    elif d0 >= magA <= d1:
                        if clampA1 and clampB0:
                            if np.absolute(d0) < np.absolute(d1):
                                return a1,b0,np.linalg.norm(a1-b0)
                            return a1,b1,np.linalg.norm(a1-b1)


                # Segments overlap, return distance between parallel segments
                return None,None,np.linalg.norm(((d0*_A)+a0)-b0)



            # Lines criss-cross: Calculate the projected closest points
            t = (b0 - a0);
            detA = np.linalg.det([t, _B, cross])
            detB = np.linalg.det([t, _A, cross])

            t0 = detA/denom;
            t1 = detB/denom;

            pA = a0 + (_A * t0) # Projected closest point on segment A
            pB = b0 + (_B * t1) # Projected closest point on segment B


            # Clamp projections
            if clampA0 or clampA1 or clampB0 or clampB1:
                if clampA0 and t0 < 0:
                    pA = a0
                elif clampA1 and t0 > magA:
                    pA = a1

                if clampB0 and t1 < 0:
                    pB = b0
                elif clampB1 and t1 > magB:
                    pB = b1

                # Clamp projection A
                if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
                    dot = np.dot(_B,(pA-b0))
                    if clampB0 and dot < 0:
                        dot = 0
                    elif clampB1 and dot > magB:
                        dot = magB
                    pB = b0 + (_B * dot)

                # Clamp projection B
                if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
                    dot = np.dot(_A,(pB-a0))
                    if clampA0 and dot < 0:
                        dot = 0
                    elif clampA1 and dot > magA:
                        dot = magA
                    pA = a0 + (_A * dot)


            return pA,pB,np.linalg.norm(pA-pB)

      def Product(a,b):
         if type(a) is str:
             if type(b) is str:
                 return(int(a==b))
             if type(b) is int:
                 return(b)
         if type(b) is str:
             if type(a) is str:
                 return(int(a==b))
             if type(a) is int:
                 return(a)
         if type(a) is list:
             if type(b) is list:
                 a_temp=[]
                 b_temp=[]
                 for el in a:
                     a_temp.append(el[2])
                 for el in b:
                     b_temp.append(el[2])
                 min_a=min(a_temp)
                 min_b=min(b_temp)
                 max_a=max(a_temp)
                 max_b=max(b_temp)
                 if (min_b>=min_a) and (max_b<=max_a):
                     return(1)
                 elif (min_a>=min_b) and (max_a<=max_b):
                     return(1)
                 elif (max_a>min_b) and (max_a<max_b):
                     return(1)
                 elif (max_b>min_a) and (max_b<max_a):
                     return(1)
                 return(0)
             elif b==1:
                 return(a)
             elif b==0:
                 return(b)
             else:
                 raise Exception('Value incompatibility error')
         if type(b) is list:
             if type(a) is list:
                 a_temp=[]
                 b_temp=[]
                 for el in a:
                     a_temp.append(el[2])
                 for el in b:
                     b_temp.append(el[2])
                 min_a=min(a_temp)
                 min_b=min(b_temp)
                 max_a=max(a_temp)
                 max_b=max(b_temp)
                 if (min_b>=min_a) and (max_b<=max_a):
                     return(1)
                 elif (min_a>=min_b) and (max_a<=max_b):
                     return(1)
                 elif (max_a>min_b) and (max_a<max_b):
                     return(1)
                 elif (max_b>min_a) and (max_b<max_a):
                     return(1)
                 return(0)
             elif a==1:
                 return(b)
             elif a==0:
                 return(a)
             else:
                 raise Exception('Value incompatibility error')
         if type(b) is int and type(a) is int:
             return(a*b)
         elif type(b) is int and ((type(a) is float) or (type(a) is np.float32)):
             return(a*b)
         elif type(a) is int and ((type(b) is float) or (type(b) is np.float32)):
             return(a*b)

      def DensityMatrix(m,f):
            matrix=[]
            for j in m:
                row=[]
                for i in f:
                    row.append(Track.Product(j,i))
                matrix.append(row)
            return matrix

      def Overlap(a):
            overlap=0
            for j in a:
                for i in j:
                    overlap+=i
            return(overlap>0)

      def DotProduct(a,b):
            if len(b)!=len(a):
                 raise Exception("Number of elements in vectors don't match")
            element=0
            for i in range(len(a)):
                element+=Track.Product(a[i],b[i])
            return(element)


      def ReplaceWeakerTracks(matx,m,f,m_fit,f_fit):
                      res_vector=[]
                      delete_vec=[]
                      for j in range(len(m)):
                          accumulative_fit_f=0
                          accumulative_fit_m=m_fit[j]
                          del_temp_vec=[]
                          counter=0
                          for i in range(len(matx[j])):
                                  if matx[j][i]==1:
                                      accumulative_fit_f+=f_fit[i]
                                      del_temp_vec.append(f[i])
                                      counter+=1
                          if (accumulative_fit_m>accumulative_fit_f/counter):
                              res_vector.append(m[j])
                              delete_vec+=del_temp_vec
                          else:
                              res_vector+=del_temp_vec
                      final_vector=[]
                      for mel in m:
                          if (mel in res_vector):
                             final_vector.append(mel)
                      for fel in f:
                          if (fel in delete_vec)==False:
                             final_vector.append(fel)
                      return(final_vector)
      def ReplaceWeakerFits(h,l_f,l_m,m_fit,f_fit):
                      new_h=l_f+l_m
                      #print('ff',new_h)
                      new_fit=f_fit+m_fit
                      #print('ff',new_fit)
                      res_fits=[]
                      for hd in range(len(new_h)):
                          if (new_h[hd] in h):
                              res_fits.append(new_fit[hd])
                      return res_fits



      def ProjectVectorElements(m,v):
                  if (len(m[0])!=len(v)):
                      raise Exception('Number of vector columns is not equal to number of acting matrix rows')
                  else:
                      res_vector=[]
                      for j in m:
                          for i in range(len(j)):
                              if (Track.Product(j[i],v[i]))==1:
                                  res_vector.append(v[i])
                              elif (Track.Product(j[i],v[i]))==v[i]:
                                  res_vector.append(v[i])
                      return(res_vector)

      def GenerateInverseVector(ov,v):
            inv_vector=[]
            for el in ov:
               if (el in v) == False:
                   inv_vector.append(1)
               elif (el in v):
                   inv_vector.append(0)
            return(inv_vector)

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
