"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import torch
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time

def binarize_dataset(threshold, training_users, training_items, training_ratings):
    for i in range(len(training_ratings)):
        if training_ratings[i] > threshold:
            training_ratings[i] = 1
        else:
            training_ratings[i] = 0
    training_users = [training_users[i] for i in range(len(training_ratings)) if training_ratings[i] != 0]
    training_items = [training_items[i] for i in range(len(training_ratings)) if training_ratings[i] != 0]
    training_ratings = [rating for rating in training_ratings if rating != 0]
    return training_users, training_items, training_ratings
    
class Loader(Dataset):
    def __init__(self, args, device, path):

        self.args = args
        self.device = device
        self.split = args.split
        self.folds = args.folds
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.pkl'
        test_file = path + '/test.pkl'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        train = pickle.load(open(train_file, "rb"))
        if self.args.dataset in ['amazon-grocery', 'amazon-digital-music']:
            train_users, train_items, train_ratings = train
            train_users, train_items, train_ratings = binarize_dataset(3, train_users, train_items,
                                                                           train_ratings)
            self.train_new = []
            for uid, iid in zip(train_users, train_items):
                self.train_new.append([uid, iid])
        else:
            self.train_new = []
            for user, items in train.items():
                for item in items:
                    self.train_new.append([user, item])

        train = np.array(self.train_new).astype(int)
        self.trainUser = train[:, 0]
        self.trainItem = train[:, 1]
        self.trainUniqueUsers = np.unique(self.trainUser)

        self.n_user = np.max(self.trainUser)
        self.m_item = np.max(self.trainItem)
        self.traindataSize = train.shape[0]

        test = pickle.load(open(test_file, "rb"))
        if self.args.dataset in ['amazon-grocery', 'amazon-digital-music']:
            test_users, test_items, test_ratings = test
            test_users, test_items, test_ratings = binarize_dataset(3, test_users, test_items,
                                                                     test_ratings)
            self.test_new = []
            for uid, iid in zip(test_users, test_items):
                self.test_new.append([uid, iid])
        else:
            self.test_new = []
            for user, items in test.items():
                for item in items:
                    self.test_new.append([user, item])

        self.test_dict = {}
        for u, i in self.test_new:
            if u in self.test_dict:
                self.test_dict[u].append(i)
            else:
                self.test_dict[u] = [i]

        test = np.array(self.test_new).astype(int)

        self.testUser = test[:, 0]
        self.testItem = test[:, 1]
        self.testUniqueUsers = np.unique(self.testUser)
        self.testDataSize = test.shape[0]

        self.n_user = max(self.n_user, np.max(self.testUser))
        self.m_item = max(self.m_item, np.max(self.testItem))

        self.m_item += 1
        self.n_user += 1

        self.Graph = None
        print(f"{self.n_user} users")
        print(f"{self.m_item} items")
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.args.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{self.args.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

# class TAFALoader(BasicDataset):

#     def __init__(self, config=world.config, path="../data/TAFA-digital-music"):
#         import pickle
#         # path = "../data/" + config['dataset']
#         # train or test
#         cprint(f'loading [{path}]')
#         self.split = config['A_split']
#         self.folds = config['A_n_fold']
#         self.mode_dict = {'train': 0, "test": 1}
#         self.mode = self.mode_dict['train']
#         self.n_user = 0
#         self.m_item = 0
#         train_file = path + '/train.pkl'
#         test_file = path + '/val.pkl'
#         self.path = path
#         trainUniqueUsers, trainItem, trainUser = [], [], []
#         testUniqueUsers, testItem, testUser = [], [], []
#         self.traindataSize = 0
#         self.testDataSize = 0

#         train = pickle.load(open(train_file, "rb"))
#         train_users, train_items, train_ratings = train
#         train_users, train_items, train_ratings = binarize_dataset(3, train_users, train_items,
#                                                                        train_ratings)
#         self.train_new = []
#         for uid, iid in zip(train_users, train_items):
#             self.train_new.append([uid, iid])

#         train = np.array(self.train_new).astype(int)
#         self.trainUser = train[:, 0]
#         self.trainItem = train[:, 1]
#         self.trainUniqueUsers = np.unique(self.trainUser)

#         self.n_user = np.max(self.trainUser)
#         self.m_item = np.max(self.trainItem)
#         self.traindataSize = train.shape[0]

#         test = pickle.load(open(test_file, "rb"))
#         test_users, test_items, test_ratings = test
#         test_users, test_items, test_ratings = binarize_dataset(3, test_users, test_items,
#                                                                  test_ratings)

#         self.test_new = []
#         for uid, iid in zip(test_users, test_items):
#             self.test_new.append([uid, iid])

#         self.test_dict = {}
#         for u, i in self.test_new:
#             if u in self.test_dict:
#                 self.test_dict[u].append(i)
#             else:
#                 self.test_dict[u] = [i]

#         test = np.array(self.test_new).astype(int)

#         self.testUser = test[:, 0]
#         self.testItem = test[:, 1]
#         self.testUniqueUsers = np.unique(self.testUser)
#         self.testDataSize = test.shape[0]

#         self.n_user = max(self.n_user, np.max(self.testUser))
#         self.m_item = max(self.m_item, np.max(self.testItem))

#         self.m_item += 1
#         self.n_user += 1

#         self.Graph = None
#         print(f"{self.n_user} users")
#         print(f"{self.m_item} items")
#         print(f"{self.trainDataSize} interactions for training")
#         print(f"{self.testDataSize} interactions for testing")
#         print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

#         # (users,items), bipartite graph
#         self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
#                                       shape=(self.n_user, self.m_item))
#         self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
#         self.users_D[self.users_D == 0.] = 1
#         self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
#         self.items_D[self.items_D == 0.] = 1.
#         # pre-calculate
#         self._allPos = self.getUserPosItems(list(range(self.n_user)))
#         self.__testDict = self.__build_test()
#         print(f"{world.dataset} is ready to go")

#     @property
#     def n_users(self):
#         return self.n_user

#     @property
#     def m_items(self):
#         return self.m_item

#     @property
#     def trainDataSize(self):
#         return self.traindataSize

#     @property
#     def testDict(self):
#         return self.__testDict

#     @property
#     def allPos(self):
#         return self._allPos

#     def _split_A_hat(self, A):
#         A_fold = []
#         fold_len = (self.n_users + self.m_items) // self.folds
#         for i_fold in range(self.folds):
#             start = i_fold * fold_len
#             if i_fold == self.folds - 1:
#                 end = self.n_users + self.m_items
#             else:
#                 end = (i_fold + 1) * fold_len
#             A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
#         return A_fold

#     def _convert_sp_mat_to_sp_tensor(self, X):
#         coo = X.tocoo().astype(np.float32)
#         row = torch.Tensor(coo.row).long()
#         col = torch.Tensor(coo.col).long()
#         index = torch.stack([row, col])
#         data = torch.FloatTensor(coo.data)
#         return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

#     def getSparseGraph(self):
#         print("loading adjacency matrix")
#         if self.Graph is None:
#             try:
#                 pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
#                 print("successfully loaded...")
#                 norm_adj = pre_adj_mat
#             except:
#                 print("generating adjacency matrix")
#                 s = time()
#                 adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
#                 adj_mat = adj_mat.tolil()
#                 R = self.UserItemNet.tolil()
#                 adj_mat[:self.n_users, self.n_users:] = R
#                 adj_mat[self.n_users:, :self.n_users] = R.T
#                 adj_mat = adj_mat.todok()
#                 # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

#                 rowsum = np.array(adj_mat.sum(axis=1))
#                 d_inv = np.power(rowsum, -0.5).flatten()
#                 d_inv[np.isinf(d_inv)] = 0.
#                 d_mat = sp.diags(d_inv)

#                 norm_adj = d_mat.dot(adj_mat)
#                 norm_adj = norm_adj.dot(d_mat)
#                 norm_adj = norm_adj.tocsr()

#                 # rowsum = np.array(adj_mat.sum(1))
#                 # d_inv = np.power(rowsum, -1).flatten()
#                 # d_inv[np.isinf(d_inv)] = 0.
#                 # d_mat_inv = sp.diags(d_inv)
#                 # norm_adj = d_mat_inv.dot(adj_mat).tocsr()

#                 end = time()
#                 print(f"costing {end - s}s, saved norm_mat...")
#                 sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

#             if self.split == True:
#                 self.Graph = self._split_A_hat(norm_adj)
#                 print("done split matrix")
#             else:
#                 self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
#                 self.Graph = self.Graph.coalesce().to(world.device)
#                 print("don't split the matrix")
#         return self.Graph

#     def __build_test(self):
#         """
#         return:
#             dict: {user: [items]}
#         """
#         test_data = {}
#         for i, item in enumerate(self.testItem):
#             user = self.testUser[i]
#             if test_data.get(user):
#                 test_data[user].append(item)
#             else:
#                 test_data[user] = [item]
#         return test_data

#     def getUserItemFeedback(self, users, items):
#         """
#         users:
#             shape [-1]
#         items:
#             shape [-1]
#         return:
#             feedback [-1]
#         """
#         return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

#     def getUserPosItems(self, users):
#         posItems = []
#         for user in users:
#             posItems.append(self.UserItemNet[user].nonzero()[1])
#         return posItems
