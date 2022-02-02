'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")

    parser.add_argument('--split', type=int, default=0, help="whether to split the graph")
    parser.add_argument('--folds', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")

    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--topks', nargs='?',default="[5, 10, 20]",
                        help="@k test list")

    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--multicore', type=int, default=1, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')

    # edge drop arguments
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keep_prob', type=float,default=0.8,
                        help="the batch size for bpr loss training procedure")

    # model parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--batch_size', type=int, default=1000,
                        help="the batch size for metric loss training procedure")
    parser.add_argument('--dim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--dataset', type=str, default='amazon-digital-music',
                        help="[amazon-digital-music, amazon-grocery, amazon-book, yelp]")
    parser.add_argument('--comb_method', type=str, default='sum',
                        help="combination method for combine the convolution layers [sum, mean, final]")
    parser.add_argument('--num_neg', type=int, default=10, help="number of negative edges")
    parser.add_argument('--decay', type=float, default=1e-4, help="the weight decay for l2 normalizaton")
    
    # Loss arguments
    parser.add_argument('--margin', type=float, default=1.0, help="margin for the metric loss")
    parser.add_argument('--alpha', type=float, default=1.25, help="pos, alpha for mul loss")
    parser.add_argument('--beta', type=float, default=5.0, help="neg, beta for mul loss")
    parser.add_argument('--lamb_p', type=float, default=6.5, help="negative threshold")
    parser.add_argument('--lamb_n', type=float, default=-0.5, help="positive threshold")

    return parser.parse_args()