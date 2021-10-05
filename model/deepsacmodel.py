import torch
from torch import nn #, cuda, backends, FloatTensor, LongTensor, optim
#from torch.autograd import Variable
import torch.nn.functional as F

from sacapp import app

out_sz = 1

class SACModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, szs, drops, emb_drop, y_range):
        super().__init__()

        n_emb = sum([s for c,s in emb_szs])
        self.n_emb, self.n_cont = n_emb, n_cont
        szs = [n_emb+n_cont] + szs

        self.embs = nn.ModuleList([nn.Embedding(c, s) for c,s in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)

        self.bn = nn.BatchNorm1d(n_cont)

        self.lins = nn.ModuleList([nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(sz) for sz in szs[1:]])

        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])

        self.outp = nn.Linear(szs[-1], out_sz)

        self.y_range = y_range

    def forward(self, x_cat, x_cont):
        ### embedding layer
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embs)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)

        ### concat embed(x_cat) and x_cont
        x2 = self.bn(x_cont)
        #x = torch.cat([x, x_cont], 1)
        x = torch.cat([x, x2], 1)

        ### linear layer block
        for l,d,b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            x = b(x)
            x = d(x)

        ### output block
        x = self.outp(x)
        x = F.sigmoid(x)
        x = x*(self.y_range[1] - self.y_range[0])
        x = x+self.y_range[0]
        return x

def get_pert_model():
    emb_szs = [(482, 50), (946, 50)]
    n_cont = 1
    szs = [50,10]
    drops = [0.01,0.01]
    emb_drop = 0.01
    y_range = [0.0, 18.001080322319996]

    m = SACModel(emb_szs, n_cont, out_sz, szs, drops, emb_drop, y_range)
    return m

def get_dr_model():
    emb_szs = [(97, 50)]
    n_cont = 946
    szs = [500,100,10]
    drops = [0.1,0.05,0.01]
    emb_drop = 0.01
    y_range = [0.000352, 9.5711436]

    m = SACModel(emb_szs, n_cont, out_sz, szs, drops, emb_drop, y_range)
    return m
