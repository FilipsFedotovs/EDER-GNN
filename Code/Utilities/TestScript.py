import Utility_Functions
from Utility_Functions import HitCluster
import torch
import torch_geometric
from torch_geometric.utils import train_test_split_edges
import pickle
#
#
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"


#print(hc.GiveStats(mc_data))
base_data_file=open('/eos/user/f/ffedship/EDER-GNN/Data/TRAIN_SET/M1_M2_SelectedTrainClusters_1.pkl','rb')
base_data=pickle.load(base_data_file)
hc=base_data[0]
hc.ClusterGraph.train_mask = hc.ClusterGraph.val_mask = hc.ClusterGraph.test_mask = hc.ClusterGraph.y = None

print(hc.ClusterGraph.train_pos_edge_index)
data=hc.ClusterGraph


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 128)
        self.conv2 = GCNConv(128, 64)

    def encode(self):
        x = self.conv1(data.x, data.train_pos_edge_index) # convolution 1
        x = x.relu()
        return self.conv2(x, data.train_pos_edge_index) # convolution 2

    def decode(self, z, pos_edge_index, neg_edge_index): # only pos and neg edges
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) # concatenate pos and neg edges
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # dot product
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t() # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t() # get predicted edge_list



model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
#
#
#
def get_link_labels(pos_edge_index, neg_edge_index):
    # returns a tensor:
    # [1,1,1,1,...,0,0,0,0,0,..] with the number of ones is equel to the lenght of pos_edge_index
    # and the number of zeros is equal to the length of neg_edge_index
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

#
#
def train():
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, #positive edges
        num_nodes=data.num_nodes, # number of nodes
        num_neg_samples=data.train_pos_edge_index.size(1)) # number of neg_sample equal to number of pos_edges

    optimizer.zero_grad()

    z = model.encode() #encode
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index) # decode

    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss
#
#
@torch.no_grad()
def test():
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']

        z = model.encode() # encode train
        link_logits = model.decode(z, pos_edge_index, neg_edge_index) # decode test or val
        link_probs = link_logits.sigmoid() # apply sigmoid

        link_labels = get_link_labels(pos_edge_index, neg_edge_index) # get link

        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu())) #compute roc_auc score
    return perfs

best_val_perf = test_perf = 0
for epoch in range(1, 2000):
     train_loss = train()
     val_perf, tmp_test_perf = test()
     if val_perf > best_val_perf:
         best_val_perf = val_perf
         test_perf = tmp_test_perf
         print(epoch, train_loss, best_val_perf, test_perf)



z = model.encode()
final_edge_index = model.decode_all(z)
print(final_edge_index)
torch.save(model.state_dict(), 'Test')

model1 = Net().to(device)
model1.load_state_dict(torch.load('Test'))
model1.eval()
z = model1.encode()
final_edge_index1 = model1.decode_all(z)
print(final_edge_index1)
GCNConv
