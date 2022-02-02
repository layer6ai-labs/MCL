'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

from multiprocessing import Process, Queue
import numpy as np
import random
import os
import torch
from torch import nn, optim

class MetricLoss:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.opt = optim.Adam((model.embedding_user.weight, model.embedding_item.weight), lr=self.args.lr)

    def stageOne(self, S, num_items_per_user):

        metric_loss, reg_loss = self.model.loss(S, num_items_per_user)
        reg_loss = reg_loss * self.args.decay
        loss = metric_loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return metric_loss.cpu().item(), reg_loss.cpu().item()

class WarpSampler(object):
    """
    A generator that, in parallel, generates tuples: user-positive-item pairs, negative-items
    of the shapes (Batch Size, 2) and (Batch Size, N_Negative)
    """

    def __init__(self, dataset, batch_size=1000, neg_k=5, n_workers=5):
        self.result_queue = Queue(maxsize=n_workers * 2)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=UniformSample_original, args=(dataset.allPos,
                                                            dataset.n_user,
                                                            dataset.m_item,
                                                            batch_size,
                                                            neg_k,
                                                            self.result_queue)))
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:  # type: Process
            p.terminate()
            p.join()

def UniformSample_original(allPos, num_users, num_items, batch_size, neg_k, result_queue):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """

    user_idx = np.arange(num_users)

    while True:
        np.random.shuffle(user_idx)
        for k in range(int(num_users / batch_size)):

            user_positive_items_pairs = []
            num_items_per_user = []

            # get positive edges
            for user in user_idx[k * batch_size: (k + 1) * batch_size]:
                posForUser = allPos[user]
                if len(posForUser) == 0:
                    continue

                for i in posForUser:
                    user_positive_items_pairs.append([user, i])
                num_items_per_user.append(len(posForUser))

            # get negative edges
            num_edges = len(user_positive_items_pairs)
            user_negative_samples = np.random.randint(0, num_items, size=(num_edges, neg_k))
            for user_positive, negatives, i in zip(user_positive_items_pairs,
                                           user_negative_samples,
                                           range(num_edges)):
                user = user_positive[0]
                for j, neg in enumerate(negatives):
                    while neg in allPos[user]:
                        user_negative_samples[i, j] = neg = np.random.randint(0, num_items)

            user_triples = np.hstack((user_positive_items_pairs, user_negative_samples))
            result_queue.put((user_triples, num_items_per_user))

def set_seed(seed):
    np.random.seed(seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName(args):
    path = "./checkpoints"
    file = f"{args.dataset}-{args.layer}-{args.dim}-{args.alpha}-{args.beta}-{args.lamb_p}-{args.lamb_n}.pth.tar"
    return os.path.join(path,file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size')

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

# def shuffle(*arrays, **kwargs):

#     require_indices = kwargs.get('indices', False)

#     if len(set(len(x) for x in arrays)) != 1:
#         raise ValueError('All inputs to shuffle must have '
#                          'the same length.')

#     shuffle_indices = np.arange(len(arrays[0]))
#     np.random.shuffle(shuffle_indices)

#     if len(arrays) == 1:
#         result = arrays[0][shuffle_indices]
#     else:
#         result = tuple(x[shuffle_indices] for x in arrays)

#     if require_indices:
#         return result, shuffle_indices
#     else:
#         return result

def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')
