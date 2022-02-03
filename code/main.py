from dataloader import Loader
from model import LightGCN
import numpy as np
from os.path import join
from parse import parse_args
import Procedure
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import utils

if __name__ == '__main__':

    # set seed
    args = parse_args()
    utils.set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # create model and load dataset
    dataset = Loader(args, device, path="../data/" + args.dataset)
    model = LightGCN(args, device, dataset)
    model = model.to(device)
    metric = utils.MetricLoss(model, args)

    # save/load file
    weight_file = utils.getFileName(args)
    print(f"load and save to {weight_file}")
    if args.load:
        try:
            model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            print(f"loaded model weights from {weight_file}") 
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")

    # init tensorboard
    if args.tensorboard:
        w : SummaryWriter = SummaryWriter(
                                join("./runs", time.strftime("%m-%d-%Hh%Mm%Ss-"))
                            )
    else:
        w = None

    # init sampler
    sampler = utils.WarpSampler(dataset, args.batch_size, args.num_neg)

    # training
    try:
        topks = eval(args.topks)
        best_result = np.zeros(2*len(topks))
        for epoch in range(1, args.epochs+1):
            print(f'Epoch {epoch}/{args.epochs}')
            start = time.time()
            if epoch % 10 == 0:
                result = Procedure.Test(args, dataset, model, epoch, device, w, args.multicore)
                if np.sum(np.append(result['recall'], result['ndcg'])) > np.sum(best_result):
                    best_result = np.append(result['recall'], result['ndcg'])
                    torch.save(model.state_dict(), weight_file)
                print("Best so far:", best_result)

            output_information = Procedure.Metric_train_original(args, dataset, model, metric, epoch, sampler, w)

            print(f'{output_information}')
            print(f"Total time {time.time() - start}")
        
        result = Procedure.Test(args, dataset, model, epoch, device, w, args.multicore)
        if np.sum(np.append(result['recall'], result['ndcg'])) > np.sum(best_result):
            best_result = np.append(result['recall'], result['ndcg'])
            torch.save(model.state_dict(), weight_file)
        print("Best overall:", best_result)

    finally:
        sampler.close()
        if args.tensorboard:
            w.close()