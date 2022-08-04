import argparse
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import yaml
from sklearn.metrics import roc_auc_score
import numpy as np 


import constants as const
import dataset
import fastflow
import utils

import hostlist
import torch.distributed as dist

def build_train_data_loader(args, config):
    train_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )


def build_train_loader_parallel(args, config, idr_torch_size, idr_torch_rank) :
    train_dataset = dataset.MVTecDataset(
                root=args.data,
                category=args.category,
                input_size=config["input_size"],
                is_train=True,
            )
    batch_size = const.BATCH_SIZE
    batch_size_per_gpu = batch_size // idr_torch_size
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=idr_torch_size,
                                                                    rank=idr_torch_rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size_per_gpu,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler)
    return train_loader


def build_test_loader_parallel(args, config, idr_torch_size, idr_torch_rank) :
    test_dataset = dataset.MVTecDataset(
                root=args.data,
                category=args.category,
                input_size=config["input_size"],
                is_train=False,
            )
    batch_size = const.BATCH_SIZE
    batch_size_per_gpu = batch_size // idr_torch_size
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                    num_replicas=idr_torch_size,
                                                                    rank=idr_torch_rank)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size_per_gpu,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=test_sampler,
                                                 drop_last=False,)
    print('Test loader created')
    return test_loader

def build_test_data_loader(args, config):
    test_dataset = dataset.MVTecDataset(
        root=args.data,
        category=args.category,
        input_size=config["input_size"],
        is_train=False,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def build_model(config):
    model = fastflow.FastFlow(
        backbone_name=config["backbone_name"],
        flow_steps=config["flow_step"],
        input_size=config["input_size"],
        conv3x3_only=config["conv3x3_only"],
        hidden_ratio=config["hidden_ratio"],
    )
    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model


def build_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=const.LR, weight_decay=const.WEIGHT_DECAY
    )


def train_one_epoch(dataloader, model, optimizer, epoch):
    model.train()
    loss_meter = utils.AverageMeter()
    for step, data in enumerate(dataloader):
        # forward
        data = data.cuda()
        ret = model(data)
        loss = ret["loss"]
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log
        loss_meter.update(loss.item())
        if (step + 1) % const.LOG_INTERVAL == 0 or (step + 1) == len(dataloader):
            print(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )


def eval_once(dataloader, model):
    model.eval()
    Predict = []
    Target = []
    for data, targets in dataloader:
        data, targets = data.cuda(), targets.cuda()
        with torch.no_grad():
            ret = model(data)
        outputs = ret["anomaly_map"].cpu().detach()
        outputs = outputs.flatten().cpu().numpy()
        targets = targets.flatten().int().cpu().numpy()
        Predict.append(outputs.tolist())
        Target.append(targets.tolist())
    Predict = [item for sublist in Predict for item in sublist] # np.ravel(np.array(Predict, dtype=object))
    Target = [item for sublist in Target for item in sublist] #np.ravel(np.array(Target, dtype=object))

    auroc =  roc_auc_score(Target,Predict)
    print("AUROC: {}".format(auroc))
    return auroc



def eval_once_outfile(dataloader, model, epoch, args):
    current_anomaly_map_outfile_path = args.anomaly_map_outfile_path + '_epoch_' + str(epoch)
    print('current_anomaly_map_outfile_path ', current_anomaly_map_outfile_path)
    current_anomaly_scores_files =  current_anomaly_map_outfile_path + '/'+ args.anomaly_scores_summary_file
    print(current_anomaly_scores_files)
    os.makedirs(current_anomaly_map_outfile_path, exist_ok=True) 
    if args.mae_encoded_vectors_path:
        current_mae_encoded_vectors_path =  current_anomaly_map_outfile_path + '/' + args.mae_encoded_vectors_path 
        os.makedirs(current_mae_encoded_vectors_path, exist_ok=True)
    
    model.eval()
    

    if args.dataset_name != 'lnen':
        with open(current_anomaly_scores_files, 'w') as file:
            file.write('Path2Image,Anomaly,Loss,MeanAnomalyScore,MaxAnomalyScore\n')
        file.close()
        for data, targets, imgfilename in dataloader:
            data, targets = data.cuda(), targets.cuda()
            with torch.no_grad():
                ret = model(data)
            outputs = ret["anomaly_map"].cpu().detach()
            log_jac_dets = ret['log_jac_dets'].cpu().detach()
            log_jac_dets_np = log_jac_dets.numpy()
            # 0.5 * torch.sum(outputs**2, dim=(1, 2, 3)) - log_jac_dets
            outputs_np = outputs.numpy()
            log_jac_dets_np = log_jac_dets.numpy()

            for i in range(len(imgfilename)):
                anom =  imgfilename[i].split('/')[-2]
                imname = imgfilename[i].split('/')[-1]
                os.makedirs(os.path.join(current_anomaly_map_outfile_path, anom), exist_ok=True)
                if args.mae_encoded_vectors_path:
                    os.makedirs(os.path.join(current_mae_encoded_vectors_path, anom), exist_ok=True)
                loss = np.mean(0.5 * np.sum(outputs_np[i]**2, axis=(0, 1, 2)) - log_jac_dets_np[i]) 
                c_anom_map = outputs_np[i].squeeze()
                mean_anom_map = c_anom_map.mean(axis=0).mean(axis=0)
                max_anom_map = c_anom_map.max(axis=0).max(axis=0)
                with open(current_anomaly_scores_files, 'a') as file:
                    file.write(f'{imgfilename[i]},{anom},{loss},{mean_anom_map},{max_anom_map}\n')
                file.close()
                
                np.save(os.path.join(current_anomaly_map_outfile_path, anom, imname[:-4]+'.npy'), c_anom_map)

            outputs = outputs.flatten()
            targets = targets.flatten()

    else:
        with open(current_anomaly_scores_files, 'a') as file:
            file.write('Path2Image,Anomaly,Loss,MeanAnomalyScore,MaxAnomalyScore\n')
        file.close()
        for data,  imgfilename in dataloader:
            data = data.cuda()
            with torch.no_grad():
                ret = model(data)
            outputs = ret["anomaly_map"].cpu().detach()
            log_jac_dets = ret['log_jac_dets'].cpu().detach()
            features = ret["features"][0].cpu().detach()
            # 0.5 * torch.sum(outputs**2, dim=(1, 2, 3)) - log_jac_dets
            log_jac_dets_np = log_jac_dets.numpy()
            features_np = features.numpy()
            outputs_np = outputs.numpy()
            for i in range(len(imgfilename)):
                sample =  imgfilename[i].split('/')[-3] # -2 if Tumor Normal expected
                imname = imgfilename[i].split('/')[-1]
                os.makedirs(os.path.join(current_anomaly_map_outfile_path, sample), exist_ok=True)
                if args.mae_encoded_vectors_path:
                    os.makedirs(os.path.join(current_mae_encoded_vectors_path, sample), exist_ok=True)
                loss =  0.5* np.sum(outputs_np[i]**2)  - log_jac_dets_np[i]  #(0.5 * torch.sum(outputs[i]**2, dim=(1, 2, 3)) - log_jac_dets
                c_features = features_np[i].squeeze().flatten()
                c_anom_map = outputs_np[i].squeeze()
                mean_anom_map = c_anom_map.mean(axis=0).mean(axis=0)
                max_anom_map = c_anom_map.max(axis=0).max(axis=0)
                with open(current_anomaly_scores_files, 'a') as file:
                    file.write(f'{imgfilename[i]},{sample},{loss},{mean_anom_map},{max_anom_map}\n')
                file.close()
                np.save(os.path.join(current_anomaly_map_outfile_path, sample, imname[:-4]+'.npy'), c_anom_map)
                if args.mae_encoded_vectors_path:
                    np.save(os.path.join(current_mae_encoded_vectors_path, sample, imname[:-4]+'.npy'), c_features)


def train(args):
   
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_dir = os.path.join(
        args.checkpoint_dir, "exp%d" % len(os.listdir(args.checkpoint_dir))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    optimizer = build_optimizer(model)

    train_dataloader = build_train_data_loader(args, config)
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, model, optimizer, epoch)
        if (epoch + 1) % const.EVAL_INTERVAL == 0:
            eval_once(test_dataloader, model)
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )


            
def train_parallel(args):
    Best_AUC_ROC = 0
    # get SLURM variables
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    size = int(os.environ['SLURM_NTASKS'])
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])

    # get node list from slurm
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
    # define MASTER_ADD & MASTER_PORT
    os.environ['MASTER_ADDR'] = hostnames[0]
    os.environ['MASTER_PORT'] = str(12456 + int(min(gpu_ids))); #Avoid port conflits in the node #str(12345 + gpu_ids)

    dist.init_process_group(backend='nccl', 
                        init_method='env://', 
                        world_size=size, 
                        rank=rank)
    
    torch.cuda.set_device(local_rank)
    # According to the tutorial 
    gpu = torch.device("cuda")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_dir = os.path.join(
        args.checkpoint_dir, "exp%d" % len(os.listdir(args.checkpoint_dir))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    model = model.to(gpu)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = build_optimizer(ddp_model)
    if args.dataset_name == 'lnen':
        train_dataloader = build_train_loader_LNEN_parallel(args, config,size, rank) 
    else:
        train_dataloader = build_train_loader_parallel(args, config,size, rank) 
        test_dataloader = build_test_loader_parallel(args, config,size, rank) 
    model.cuda()

    for epoch in range(const.NUM_EPOCHS):
        train_one_epoch(train_dataloader, ddp_model, optimizer, epoch)
        if args.dataset_name != 'lnen':
            if (epoch + 1) % const.EVAL_INTERVAL == 0:
                auroc_per_pixel = eval_once(test_dataloader, model)
                if auroc_per_pixel > Best_AUC_ROC:
                    Best_AUC_ROC = auroc_per_pixel
                    torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": ddp_model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(checkpoint_dir, "_best_model.pt" ),
                )
        if (epoch + 1) % const.CHECKPOINT_INTERVAL == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ddp_model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, "%d.pt" % epoch),
            )


def evaluate(args, epoch ):
    config = yaml.safe_load(open(args.config, "r"))
    current_checkpoint = args.checkpoint + '/' + str(epoch) + '.pt'
    model = build_model(config)
    checkpoint = torch.load(current_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataloader = build_test_data_loader(args, config)
    model.cuda()
    eval_once_outfile(test_dataloader, model, epoch, args)


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="path whre the checkoint will be saved")
    
    ## Dataset 
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")

    parser.add_argument("--dataset_name", type=str, required=False, help="Data set name")

    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    

    parser.add_argument("--parallel", action="store_true", help="Train parallel")
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    
    # Evaluation output path
    parser.add_argument(
        "-anom_map", "--anomaly_map_outfile_path", type=str, help="path to scores.txt"
    )
    
    
    parser.add_argument(
        "-mae_vector", "--mae_encoded_vectors_path", type=str, help="MAE encoded vector folder"
    )
    parser.add_argument(
       "--anomaly_scores_summary_file", type=str, help="Path + Generic filename to save the anomaly scores"
    )
    parser.add_argument( "--epochs_evaluated", required=False,  nargs='+', 
                help="Threshold optimisation maximum number of iteration ")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.eval:
        if args.parallel :
            print('\n\n\n EVAL IN PARALLEL')
#             for epoch in args.epochs_evaluated:
#             evaluate(args)
        else:
            for epoch in args.epochs_evaluated:
                evaluate(args, epoch)
        
    else:
        if args.parallel :
            train_parallel(args)
        else:
            print('FastFlow is training  ...')
            train(args)

