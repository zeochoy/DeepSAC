import json, pdb, os, numpy as np, threading, math, random
import pandas as pd
#from urllib.request import urlopen

import torch
from torch import nn, cuda, backends, FloatTensor, LongTensor
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torch.utils.model_zoo import load_url

from enum import IntEnum

from plotly.utils import PlotlyJSONEncoder
import plotly.plotly as py
import plotly.graph_objs as go

from sacapp import app
from model.deepsacmodel import *

### -------- from fastai
def to_np(v):
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    if isinstance(v, torch.cuda.HalfTensor): v=v.float()
    return v.cpu().numpy()

### -------- deepsac dataset and DataLoader
class DeepSACDataset(Dataset):
    def __init__(self, cats, conts):
        self.cats  = cats
        self.conts = conts

    def __len__(self): return len(self.cats)

    def __getitem__(self, idx):
        return self.cats[idx], self.conts[idx]

    @classmethod
    def from_data_frames(cls, df_cat, df_cont):
        cat_cols = np.array(df_cat).astype(np.int64)
        cont_cols = np.array(df_cont).astype(np.float32)
        return cls(cat_cols, cont_cols)

    @classmethod
    def from_data_frame(cls, df, cat_flds):
        return cls.from_data_frames(df[cat_flds], df.drop(cat_flds, axis=1))

### -------- deepsac utils funcs
def read_list(path, as_int=True):
    with open(path, 'r') as f:
        ls = f.read().splitlines()
    l = [l for l in ls]
    if as_int: l = [int(x) for x in l]
    return l

def read_dict(path, as_int=True):
    with open(path, 'r') as f:
        ls = f.read().splitlines()
    d = {l.split('\t')[0]:l.split('\t')[1].strip('\n') for l in ls}
    if as_int: d = {int(k):int(v) for k,v in d.items()}
    return d

def parse_cell_line_exp(cell_line):
    ### parse CCLE cell line RPKM from file
    df = pd.read_csv(app.config['CCLE_RPKM_FILE'], index_col=False)
    tdf = df[df.cell_line==cell_line]
    tdf.drop('cell_line', axis=1, inplace=True)
    return tdf

def tfm_exp2pert(exp_df, pert_drugs, pert_drug2idx, gene2idx):
    ### transform expression df to pert model input
    tdf = pd.concat([pd.DataFrame(exp_df)]*len(pert_drugs), ignore_index=True)
    gidxs = [gene2idx[g] for g in tdf.columns]
    tdf.columns = gidxs
    didxs = [pert_drug2idx[d] for d in pert_drugs]
    tdf['didx'] = didxs
    tdf = pd.melt(tdf, id_vars=['didx'], var_name='gidx', value_name='exp_ctl')
    return tdf

def tfm_exp2dr(exp_df, ic50_drugs, ic50_drug2idx, ic50_genes):
    ### transform expression df with one cell line to dr model input
    tdf = pd.DataFrame(exp_df[ic50_genes])
    tdf = pd.concat([tdf]*len(ic50_drugs), ignore_index=True)
    didxs = [ic50_drug2idx[d] for d in ic50_drugs]
    tdf['didx'] = didxs
    tdf['didx'] = tdf['didx'].astype('category')
    return tdf

def tfm_pert2dr(in_df, outs, pert_drugs, pert_drug2idx, ic50_drugs, ic50_drug2idx, gene2idx, ic50_genes):
    df = in_df
    df = df.drop('exp_ctl', axis=1)
    df['exp_pert'] = outs
    pert_idx2drug = {v:k for k,v in pert_drug2idx.items()}
    in_drugids = [pert_idx2drug[i] for i in df.didx]
    df['pert_drugid'] = in_drugids
    idx2gene = {v:k for k,v in gene2idx.items()}
    genes = [idx2gene[i] for i in df.gidx]
    df['gene'] = genes
    df = df.drop('gidx', axis=1)
    df = df.pivot(index='pert_drugid', columns='gene')['exp_pert']
    in_drugids = df.index
    df = df[ic50_genes]
    df = pd.concat([df]*len(ic50_drugs), ignore_index=True)
    didxs = [ic50_drug2idx[d] for d in ic50_drugs]
    didxs = np.repeat(didxs, len(pert_drugs))
    df['didx'] = didxs
    return df, in_drugids

def get_prediction(dl, model):
    ### generate prediction from dataset and loaded model
    preds = []
    for _,x in enumerate(dl):
        cat, cont = x[0], x[1]
        tmp = to_np(model(Variable(cat), Variable(cont)))
        preds.append(tmp)
    return preds

def gen_ic50_diff(org_ic50, pert_ic50, drug2id, in_drugids, ic50_drugs, pert_drugs):
    ### generate df of ic50 difference
    id2drug = {v:k for k,v in drug2id.items()}

    pert_drugid = list(in_drugids) * len(ic50_drugs)
    dr_drugid = np.repeat(ic50_drugs, len(pert_drugs))
    pert_dn = [id2drug[i] for i in pert_drugid]
    dr_dn = [id2drug[i] for i in dr_drugid]

    org_ic50_re = np.repeat(org_ic50, len(pert_drugs))
    d_ic50 = pert_ic50 - org_ic50_re

    res_df = pd.DataFrame.from_dict({'sensitizer':pert_dn, 'drug':dr_dn, 'delta_ic50':d_ic50, 'IC50_ori':org_ic50_re, 'IC50_pert':pert_ic50})
    res_df['delta_ic50'] = np.where(res_df.sensitizer==res_df.drug, np.nan, res_df.delta_ic50)
    res_df = res_df[['sensitizer', 'drug', 'delta_ic50', 'IC50_ori', 'IC50_pert']]
    return res_df

def plot_heatmap(diff_df):
    # plot delta ic50 heatmap
    tdf = diff_df.drop(columns=['IC50_ori', 'IC50_pert'])
    tdf = tdf.pivot(index='drug', columns='sensitizer')['delta_ic50']
    data = [go.Heatmap(z=tdf.values.tolist(), x=tdf.index.values.tolist(), y=tdf.columns.values.tolist())]
    #fig = go.Figure(data=data)
    graphJSON = json.dumps(data, cls=PlotlyJSONEncoder)
    return graphJSON

### -------- sacapp funcs
def get_deepsac_prediction(cell_line):
    # load drug2idx, gene2idx
    ic50_drugs = read_list(app.config['IC50_DRUGS_FILE'])
    pert_drugs = read_list(app.config['PERT_DRUGS_FILE'])
    ic50_drug2idx = read_dict(app.config['IC50_DRUG2ID_FILE'])
    pert_drug2idx = read_dict(app.config['PERT_DRUG2ID_FILE'])
    gene2idx = read_dict(app.config['GENE2ID_FILE'], as_int=False)
    gene2idx = {k:int(v) for k,v in gene2idx.items()}
    ic50_genes = read_list(app.config['IC50_GENES_FILE'], as_int=False)
    drug2id = read_dict(app.config['DRUG2ID_FILE'], as_int=False)
    drug2id = {k:int(v) for k,v in drug2id.items()}

    # load model
    #pert_models = load_pert()
    #dr_models = load_dr()

    # load data
    exp_df = parse_cell_line_exp(cell_line)

    ### prepare pert input
    pert_in = tfm_exp2pert(exp_df, pert_drugs, pert_drug2idx, gene2idx)
    pert_bs = 1892
    pert_ds = DeepSACDataset.from_data_frame(pert_in, ['didx', 'gidx'])
    pert_dl = DataLoader(pert_ds, batch_size=pert_bs, shuffle=False)

    ### perturbation
    pert_wgts_path = get_pert_path()
    pert_outs = []
    for p in pert_wgts_path:
        tmp_pert_model = get_pert_model()
        tmp_pert_model.load_state_dict(torch.load(p))
        tmp_pert_model.eval()
        tmp_pert_out = get_prediction(pert_dl, tmp_pert_model)
        pert_out = np.reshape(tmp_pert_out, len(tmp_pert_out)*pert_bs)
        pert_outs.append(pert_out)
    pert_outs = np.mean(np.array(pert_outs).T, axis=1)

    ### prepare dr input
    dr_bs = 97
    # without perturbation
    org_dr_in = tfm_exp2dr(exp_df, ic50_drugs, ic50_drug2idx, ic50_genes)
    org_dr_ds = DeepSACDataset.from_data_frame(org_dr_in, ['didx'])
    org_dr_dl = DataLoader(org_dr_ds, batch_size=dr_bs, shuffle=False)
    # after perturbation
    pert_dr_in, in_drugids = tfm_pert2dr(pert_in, pert_outs, pert_drugs, pert_drug2idx, ic50_drugs, ic50_drug2idx, gene2idx, ic50_genes)
    pert_dr_ds = DeepSACDataset.from_data_frame(pert_dr_in, ['didx'])
    pert_dr_dl = DataLoader(pert_dr_ds, batch_size=dr_bs, shuffle=False)

    ### dr
    dr_wgts_path = get_dr_path()
    org_dr_outs = []
    pert_dr_outs = []
    for p in dr_wgts_path:
        tmp_dr_model = get_dr_model()
        tmp_dr_model.load_state_dict(torch.load(p))
        tmp_dr_model.eval()
        tmp_org_dr_out = get_prediction(org_dr_dl, tmp_dr_model)
        tmp_pert_dr_out = get_prediction(pert_dr_dl, tmp_dr_model)
        org_dr_out = np.reshape(tmp_org_dr_out, len(tmp_org_dr_out)*dr_bs)
        pert_dr_out = np.reshape(tmp_pert_dr_out, len(tmp_pert_dr_out)*dr_bs)
        org_dr_outs.append(org_dr_out)
        pert_dr_outs.append(pert_dr_out)
    org_dr_outs = np.mean(np.array(org_dr_outs).T, axis=1)
    pert_dr_outs = np.mean(np.array(pert_dr_outs).T, axis=1)

    #compute diff
    diff_df = gen_ic50_diff(org_dr_outs, pert_dr_outs, drug2id, in_drugids, ic50_drugs, pert_drugs)
    # generate output for plotly
    graphJSON = plot_heatmap(diff_df)
    #graphJSON = json.dumps(hm, cls=PlotlyJSONEncoder)

    # generate output for table view
    diff_df = diff_df.dropna()
    diff_df.columns = ['Sensitizer', 'Drug', '&#9651;ln(IC50)', 'ln(IC50_original)', 'ln(IC50_pert)']
    df_html = diff_df.to_html(header=True, index=False, escape=False, table_id='datatable').replace(' border="1"','').replace('class="dataframe"','class="display"').replace('<tr style="text-align: right;">','<tr>').replace('\n','')

    return {'dfHTML':df_html, 'graphJSON':graphJSON}

def get_pert_path():
    dst_folder = app.config['MODEL_FOLDER']
    wgts_dsts = [dst_folder+'/deepsac_pert_'+str(i+1)+'.pt' for i in range(5)]
    return wgts_dsts

def get_dr_path():
    dst_folder = app.config['MODEL_FOLDER']
    wgts_dsts = [dst_folder+'/deepsac_dr_'+str(i+1)+'.pt' for i in range(5)]
    return wgts_dsts

#def load_pert():
#    dst_folder = app.config['MODEL_FOLDER']
#    wgts_dsts = [dst_folder+'/deepsac_pert_'+str(i+1)+'.pt' for i in range(5)]
#    model = get_pert_model()
#    models = [model.load_state_dict(torch.load(d)) for d in wgts_dsts]
#    return models

#def load_dr():
#    dst_folder = app.config['MODEL_FOLDER']
#    wgts_dsts = [dst_folder+'/deepsac_dr_'+str(i+1)+'.pt' for i in range(5)]
#    model = get_dr_model()
#    models = [model.load_state_dict(torch.load(d)) for d in wgts_dsts]
#    return models
