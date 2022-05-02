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

            _Tot_Hits=_Tot_Hits[(_Tot_Hits['l_MC_ID']== '55281-1') | (_Tot_Hits['r_MC_ID']== '55281-1') |(_Tot_Hits['l_MC_ID']== '55281-0') | (_Tot_Hits['r_MC_ID']== '55281-0')]
            print(_Tot_Hits)
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
            # print(_Tot_Hits[['_r_HitID', 'r_z','_l_HitID','l_z','link_strength']])
            # _Tot_Hits=_Tot_Hits.sort_values(by=['_l_HitID','r_z','link_strength'], ascending=False)
            # print(_Tot_Hits[['_r_HitID', 'r_z','_l_HitID','l_z','link_strength']])
            # _Tot_Hits.drop_duplicates(subset=['_l_HitID','r_z'],keep='first', inplace=True)
            # print(_Tot_Hits[['_r_HitID', 'r_z','_l_HitID','l_z','link_strength']])
            # StatFakeValues.append(len(_Tot_Hits.axes[0])-len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))
            # StatTruthValues.append(len(_Tot_Hits.drop(_Tot_Hits.index[_Tot_Hits['l_MC_ID'] != _Tot_Hits['r_MC_ID']]).axes[0]))

            print(StatFakeValues)
            print(StatTruthValues)
            #_Tot_Hits=_Tot_Hits[(_Tot_Hits['l_MC_ID']== '54703-0')]
            #print(_Tot_Hits.sort_values(by=['l_MC_ID','r_MC_ID'], ascending=False))
            #93735-3975
            #exit()
            _Tot_Hits=_Tot_Hits[['_l_HitID','_r_HitID','r_z','link_strength']]
            Trigger=False
            print(_Tot_Hits)
            input("Press Enter to continue...")
            _Tot_Hits_Pool=_Tot_Hits

            z_ind=_Tot_Hits_Pool.sort_values(by=['r_z'], ascending=True)[['r_z']].drop_duplicates(subset=['r_z'],keep='first').values.tolist()
            for z in range(0,len(z_ind)):
                temp_s_hits=_Tot_Hits_Pool.drop(_Tot_Hits_Pool.index[_Tot_Hits_Pool['r_z'] != z_ind[z][0]])
                #_Tot_Hits_Pool=_Tot_Hits_Pool.drop(_Tot_Hits_Pool.index[_Tot_Hits_Pool['r_z'] == z_ind[z][0]])
                temp_s_hits['Segment_0']=temp_s_hits['_r_HitID']
                temp_s_hits['Track_ID']=temp_s_hits['_r_HitID']
                #(temp_s_hits)
                temp_s_hits=temp_s_hits.rename(columns={"link_strength": "Fit"})
                temp_s_hits['DoF']=2
                temp_s_hits=temp_s_hits.drop(["_r_HitID",'r_z'], axis=1)
                temp_s_hits=temp_s_hits.rename(columns={"_l_HitID": "_r_HitID" })
                print(temp_s_hits)

                input("Press Enter to continue...")
                #temp_s_hits=temp_s_hits.loc[temp_s_hits['Track_ID'] == '9796888']

                for zz in range(z+1,len(z_ind)):
                    temp_m_hits=_Tot_Hits_Pool.drop(_Tot_Hits_Pool.index[_Tot_Hits_Pool['r_z'] != z_ind[zz][0]])

                    temp_s_hits=pd.merge(temp_s_hits, temp_m_hits, how="right", on=['_r_HitID'])

                    temp_s_hits['Segment_'+str(zz)]=temp_s_hits['_r_HitID']
                    print(temp_s_hits)
                    print(_Tot_Hits_Pool)
                    _Tot_Hits_Pool=pd.merge(_Tot_Hits_Pool, temp_s_hits[['_r_HitID','_l_HitID','Segment_'+str(zz)]], how="left", on=['_r_HitID','_l_HitID'])
                    _Tot_Hits_Pool=_Tot_Hits_Pool[_Tot_Hits_Pool['Segment_'+str(zz)].isnull()]
                    _Tot_Hits_Pool=_Tot_Hits_Pool.drop(['Segment_'+str(zz)], axis=1)

                    temp_s_hits['Track_ID']+=('-'+temp_s_hits['_r_HitID'])
                    temp_s_hits['DoF']+=1
                    temp_s_hits['Fit']+=temp_s_hits['link_strength']
                    if zz==len(z_ind)-1:
                        print(temp_s_hits)
                        temp_s_hits['Track_ID']+=('-'+temp_s_hits['_l_HitID'])
                        temp_s_hits['Segment_'+str(zz+1)]=temp_s_hits['_l_HitID']

                    temp_s_hits=temp_s_hits.drop(["_r_HitID",'r_z','link_strength'], axis=1)
                    temp_s_hits=temp_s_hits.rename(columns={"_l_HitID": "_r_HitID" })
                    if zz==len(z_ind)-1:
                        print(temp_s_hits)
                        temp_s_hits=temp_s_hits.drop(["_r_HitID"], axis=1)
                    print(temp_s_hits)
                    input("Press Enter ed to continue...")
                print(temp_s_hits)
                print(_Tot_Hits_Pool)
                temp_s_hits['join_key']='join_key'
                _Tot_Hits_Pool['join_key']='join_key'
                temp_s_hits['left_hit']=False
                temp_s_hits['right_hit']=False
                temp_e_hits=pd.merge(temp_s_hits, _Tot_Hits_Pool, how="left", on=['join_key'])
                print(temp_e_hits)
                columns=[col for col in temp_e_hits.columns if 'Segment' in col]
                print(columns)
                def Check_lOverlap(row):
                    for c in columns:
                      if row[c]==row["_r_HitID"]:
                         return True
                    return False
                def Check_rOverlap(row):
                    for c in columns:
                      if row[c]==row["_l_HitID"]:
                         return True
                    return False
                temp_e_hits['left_hit'] = temp_e_hits.apply(Check_lOverlap,axis=1)
                temp_e_hits['right_hit'] = temp_e_hits.apply(Check_rOverlap,axis=1)

                def Check_bOverlap(row):

                         if row['left_hit']==row['right_hit']==True:
                             return row['link_strength']
                         else:
                             return 0.0

                temp_e_hits['link_strength']=temp_e_hits.apply(Check_bOverlap,axis=1)
                temp_e_hits=temp_e_hits.groupby(['Track_ID','DoF','Fit'])['link_strength'].sum().reset_index()
                temp_e_hits['Fit/DOF']=(temp_e_hits['link_strength']+temp_e_hits['Fit'])/(temp_e_hits['DoF']-1)
                temp_e_hits=temp_e_hits.sort_values(by=['Fit/DOF'], ascending=False)
                temp_e_hits=temp_e_hits[['Track_ID']].iloc[:1]
                temp_e_hits=pd.merge(temp_e_hits, temp_s_hits, how="inner", on=['Track_ID'])
                temp_e_hits=temp_e_hits.drop(["Fit",'DoF','join_key','left_hit','right_hit'], axis=1)
                temp_e_hits=pd.melt(temp_e_hits, id_vars=['Track_ID'])
                temp_e_hits=temp_e_hits.drop(["variable"], axis=1)
                temp_e_hits= temp_e_hits.rename(columns={'value': "HitID",'Track_ID': 'Segment_ID'})
                print(temp_e_hits)
                if Trigger:
                    f_frames=[f_result,temp_e_hits]
                    f_result=pd.concat(f_frames)
                else:
                    f_result=temp_e_hits
                    Trigger=True
                _Tot_Hits=pd.merge(_Tot_Hits, temp_e_hits, how="left", left_on=['_l_HitID'], right_on=['Segment_ID'])
                print(_Tot_Hits)
                exit()
                #
                #
                # f_result_sl=f_result.groupby(by=['Segment_ID'])['HitID'].count().reset_index()
                # f_result_sl=f_result_sl.rename(columns={"HitID": "Segment_Fit"})
                # f_result=pd.merge(f_result, f_result_sl, how="inner", on=['Segment_ID'])
                # f_result=f_result.sort_values(by=['HitID','Segment_Fit'], ascending=False)
                # f_result=f_result.drop_duplicates(subset='HitID',keep='first')
                # print(f_result)
                # f_result=f_result[['HitID','Segment_ID']]
                # _l_fHits= f_result.rename(columns={"HitID": "_l_HitID"})
                # _l_Tot_fHits=pd.merge(_l_MCHits, _l_fHits, how="left", on=['_l_HitID'])
                # _r_fHits= f_result.rename(columns={"HitID": "_r_HitID"})
                #
                # #Join hits + MC truth
                # _r_Tot_fHits=pd.merge(_r_MCHits, _r_fHits, how="right", on=['_r_HitID'])
                # _r_Tot_fHits.drop_duplicates(subset=['_r_HitID'],keep='first', inplace=True)
                # _l_Tot_fHits.drop_duplicates(subset=['_l_HitID'],keep='first', inplace=True)
                # _Tot_fHits=pd.merge(_l_Tot_fHits, _r_Tot_fHits, how="inner",on=["Segment_ID"])
                # _Tot_fHits.l_MC_ID= _Tot_fHits.l_MC_ID.fillna(_Tot_fHits._l_HitID)
                # _Tot_fHits.r_MC_ID= _Tot_fHits.r_MC_ID.fillna(_Tot_fHits._r_HitID)
                # _Tot_fHits.drop(_Tot_fHits.index[_Tot_fHits['_l_HitID'] == _Tot_fHits['_r_HitID']], inplace = True)
                # _Tot_fHits["Pair_ID"]= ['-'.join(sorted(tup)) for tup in zip(_Tot_fHits['_l_HitID'], _Tot_fHits['_r_HitID'])]
                # _Tot_fHits.drop_duplicates(subset="Pair_ID",keep='first',inplace=True)
                # print(_Tot_fHits[(_Tot_fHits['l_MC_ID']!= _Tot_fHits['r_MC_ID'])])
                # exit()
                # StatFakeValues.append(len(_Tot_fHits.axes[0])-len(_Tot_fHits.drop(_Tot_fHits.index[_Tot_fHits['l_MC_ID'] != _Tot_fHits['r_MC_ID']]).axes[0]))
                # StatTruthValues.append(len(_Tot_fHits.drop(_Tot_fHits.index[_Tot_fHits['l_MC_ID'] != _Tot_fHits['r_MC_ID']]).axes[0]))
                # print(StatFakeValues)
                # print(StatTruthValues)
                # exit()
                # # exit()


      @staticmethod
      def GenerateLinks(_input,_ClusterID):
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
