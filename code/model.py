"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import torch
from torch import nn
import numpy as np
import random
import itertools

class LightGCN(nn.Module):
    def __init__(self, args, device, dataset):
        super(LightGCN, self).__init__()
        self.args = args
        self.device = device
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.args.dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.args.dim)

        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.args.split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self, print_norm=False):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        
        if self.args.dropout:
            if self.training:
                g_droped = self.__dropout(self.args.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.args.layer):
            if self.args.split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        if self.args.comb_method == 'mean':
            light_out = torch.mean(embs, dim=1)
        elif self.args.comb_method == 'sum':
            light_out = torch.sum(embs, dim=1)
        elif self.args.comb_method == 'final':
            light_out = embs[:, -1, :]

        users, items = torch.split(light_out, [self.num_users, self.num_items])

        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()].unsqueeze(1)
        items_emb = all_items.unsqueeze(0)
        rating = -torch.sum((users_emb - items_emb) ** 2, 2)
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def loss(self, S, num_items_per_user):
        users = torch.Tensor(S[:, 0]).long()
        pos_items = torch.Tensor(S[:, 1]).long()
        neg_items = torch.Tensor(S[:, 2:]).long()

        users = users.to(self.device)
        pos_items = pos_items.to(self.device)
        neg_items = neg_items.to(self.device)

        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users, pos_items, neg_items)

        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2)) / float(len(users)) + \
                   (1 / 2) * negEmb0.norm(2).pow(2) / float(len(users)*self.args.num_neg)

        # positive item to user distance (N)
        pos_distances = torch.sum((users_emb - pos_emb) ** 2, 1)

        # distance to negative items (N x W)
        distance_to_neg_items = torch.sum((users_emb.unsqueeze(-1) - neg_emb.transpose(-2, -1)) ** 2, 1)
        min_neg_per_item = distance_to_neg_items.min(1)[0]

        # mining
        start_idx = 0
        pos_lengths = []
        neg_length = []
        for i in num_items_per_user:

            max_pos_length = pos_distances[start_idx: start_idx+i].max()
            pos_lengths.append(max_pos_length)
            
            min_neg_length = min_neg_per_item[start_idx: start_idx+i].min()
            neg_length.append(min_neg_length)
            
            start_idx += i

        num_items_per_user = torch.LongTensor(num_items_per_user)

        pos_lengths = torch.repeat_interleave(torch.tensor(pos_lengths), num_items_per_user)
        pos_lengths = pos_lengths.to(self.device)
        neg_length = torch.repeat_interleave(torch.tensor(neg_length), num_items_per_user)
        neg_length = neg_length.to(self.device)

        # negative mining using max pos length
        neg_idx = (distance_to_neg_items - (self.args.margin + pos_lengths.unsqueeze(-1))) >= 0
        distance_to_neg_items = distance_to_neg_items + torch.where(neg_idx, float('inf'), 0.)

        # positive mining using min neg length
        pos_idx = (pos_distances - (neg_length - self.args.margin)) <= 0
        pos_distances = pos_distances + torch.where(pos_idx, -float('inf'), 0.)

        # compute loss
        neg_loss = 1.0 / self.args.beta * torch.log(1 + torch.sum(torch.exp(-self.args.beta * (distance_to_neg_items + self.args.lamb_n)))/self.args.batch_size)
        pos_loss = 1.0 / self.args.alpha * torch.log(1 + torch.sum(torch.exp(self.args.alpha * (pos_distances + self.args.lamb_p)))/self.args.batch_size)
        
        return (neg_loss+pos_loss), reg_loss
