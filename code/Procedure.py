'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
from itertools import repeat
import numpy as np
import torch
import utils
import multiprocessing

CORES = 1

def Metric_train_original(args, dataset, model, loss_class, epoch, sampler, w=None):
    Recmodel = model
    Recmodel.train()
    metric = loss_class
    batch_size = args.batch_size

    total_batch = dataset.n_users // batch_size
    aver_metric_loss = 0.
    aver_reg_loss = 0.
    for k in range(total_batch):
        samples = sampler.next_batch()
        S = samples[0]
        num_items_per_user = samples[1]

        metric_loss, reg_loss = metric.stageOne(S, num_items_per_user)

        aver_metric_loss += metric_loss
        aver_reg_loss += reg_loss
        if args.tensorboard:
            w.add_scalar(f'MetricLoss/BPR', metric_loss, epoch * total_batch + k)
            w.add_scalar(f'RegLoss/BPR', reg_loss, epoch * total_batch + k)
    aver_metric_loss = aver_metric_loss / total_batch
    aver_reg_loss = aver_reg_loss / total_batch

    return f"aver metric loss{aver_metric_loss:.3e}, aver reg loss{aver_reg_loss:.3e}"
    
def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    assert num_users == true_users
    return sum_recall / true_users
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    topks = X[2]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
            
def Test(args, dataset, Recmodel, epoch, device, w=None, multicore=0):

    u_batch_size = args.testbatch
    testDict = dataset.testDict
    topks = eval(args.topks)

    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        auc_record = []
        total_batch = len(users) // u_batch_size + 1

        all_users, all_items = Recmodel.computer(True)

        items_emb = all_items.unsqueeze(0)

        avg_dist = 0
        avg_pos_dist = 0

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(device)
            users_emb = all_users[batch_users_gpu.long()].unsqueeze(1)
            rating = -torch.sum((users_emb - items_emb) ** 2, 2)
            avg_dist -= rating.mean().item() / float(total_batch)
            
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            avg_pos_dist -= rating[exclude_index, exclude_items].mean().item() / float(total_batch)

            rating[exclude_index, exclude_items] = np.NINF
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list, repeat(topks))
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        if args.tensorboard:
            w.add_scalars(f'Test/Recall@{topks}',
                          {str(topks[i]): results['recall'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/Precision@{topks}',
                          {str(topks[i]): results['precision'][i] for i in range(len(topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{topks}',
                          {str(topks[i]): results['ndcg'][i] for i in range(len(topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
