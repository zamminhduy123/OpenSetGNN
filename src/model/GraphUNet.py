import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F

'''
Reference: Graph U-Nets (ICML'19)
https://github.com/HongyangGao/Graph-U-Nets
'''

import torch
import torch.nn as nn
import numpy as np


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h


class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, dim=-1)
    degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)
    g = g / degrees
    return g
    
class Encoder(nn.Module):
    def __init__(self, ks, dim, act, drop_p):
        super(Encoder, self).__init__()
        self.ks = ks
        self.down_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.LNs = nn.ModuleList()
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.LNs.append(nn.LayerNorm(dim))
    
    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []

        for i in range(self.l_n):
            g = norm_g(g)
            h1 = self.down_gcns[i](g, h)
            h = self.LNs[i](h + h1)
            down_outs.append(h)
            adj_ms.append(g)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)
        return g, h, adj_ms, down_outs, indices_list

class Decoder(nn.Module):
    '''
    gcn
    '''
    def __init__(self, ks, dim, act, drop_p) -> None:
        super(Decoder, self).__init__()
        self.inp_LNs = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.LNs = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.inp_LNs.append(nn.LayerNorm(dim))
            self.unpools.append(Unpool())
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.LNs.append(nn.LayerNorm(dim))

        self.out_ln = nn.LayerNorm(dim)

    def forward(self, h, ori_h, down_outs, adj_ms, indices_list):
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, idx)
            h1 = self.inp_LNs[i](down_outs[up_idx] + h)
            g = norm_g(g)
            h = self.up_gcns[i](g, h1)
            h = self.LNs[i](h + h1)
        h = self.out_ln(h + ori_h)
        return h


class Unet(nn.Module):
    '''
    two-way network
    '''
    def __init__(self, in_dim=None, args=None, s_gcn_state=None, encoder_state=None, s_ln_state=None) -> None:
        super(Unet, self).__init__()
        self.act = getattr(nn, args.act)()
        self.mask_ratio = args.mask_ratio

        self.s_gcn = GCN(in_dim, args.dim, self.act, args.drop_p)
        self.s_ln = nn.LayerNorm(args.dim)
        if s_gcn_state:
            self.s_gcn.load_state_dict(s_gcn_state)
            for param in self.s_gcn.parameters(): # freeze the grad of source gcn
                param.requires_grad = False
        if s_ln_state:
            self.s_ln.load_state_dict(s_ln_state)
            for param in self.s_ln.parameters(): # freeze the grad of source gcn
                param.requires_grad = False

        self.g_enc = Encoder(args.ks, args.dim, self.act, args.drop_p)
        if encoder_state:
            self.g_enc.load_state_dict(encoder_state)
            for param in self.g_enc.parameters(): # freeze the grad of encoder
                param.requires_grad = False

        self.bot_gcn = GCN(args.dim, args.dim, self.act, args.drop_p)
        self.bot_ln = nn.LayerNorm(args.dim)
        self.g_dec1 = Decoder(args.ks, args.dim, self.act, args.drop_p)
        self.g_dec2 = Decoder(args.ks, args.dim, self.act, args.drop_p)

        self.reduce1 = nn.Linear(args.dim, args.dim)
        self.reduce2 = nn.Linear(args.dim, args.dim)
    
    def forward(self, gs, hs):
        o_gs = self.embed(gs, hs)
        return self.customBCE(o_gs, gs), o_gs
    
    def embed(self, gs, hs):
        o_gs = []
        for g, h in zip(gs, hs):
            og = self.embed_one(g, h)
            o_gs.append(og)
        return o_gs

    def embed_one(self, g, h):
        g = norm_g(g)
        h = self.s_gcn(g, h)
        h = self.s_ln(h)
        ori_h = h
        g, h, adj_ms, down_outs, indices_list = self.g_enc(g, h)

        g = norm_g(g)
        h = self.bot_gcn(g, h)
        h = self.bot_ln(h)
        h1 = self.g_dec1(h, ori_h, down_outs, adj_ms, indices_list)
        h2 = self.g_dec2(h, ori_h, down_outs, adj_ms, indices_list)

        h1 = self.reduce1(h1)
        h2 = self.reduce2(h2)
        h = (h1 @ h2.T)

        return torch.sigmoid(h)
        # return torch.sigmoid((h+h.T)/2)

    def customBCE(self, o_gs, gs):
        loss = []
        cnts = 0
        for og, g in zip(o_gs, gs):
            tn = g.numel()
            zeros = tn - g.sum()
            ones = g.sum()
            one_weight = tn / 2 / ones
            zero_weight = tn / 2 / zeros
            weights = torch.where(g == 0, zero_weight, one_weight)
            loss.append(F.binary_cross_entropy(og, g, weight=weights))
        return torch.tensor(loss, requires_grad=True)
    
