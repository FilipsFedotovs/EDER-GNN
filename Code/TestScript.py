import Utility_Functions
from Utility_Functions import HitCluster
hc=HitCluster([0,0,0],[1000,1000,1000])
raw_data=[[1,1,2,1,1,1],[2,500,500,500,1,1],[3,999,999,800,1,1]]
hc.LoadClusterHits(raw_data)
mc_data=[[1,1,2,1,1,1,'1'],[2,500,500,500,1,1,'2'],[3,999,999,800,1,1,'2']]
hc.LabelClusterHits(mc_data)
