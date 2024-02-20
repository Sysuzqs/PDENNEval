# coding=utf-8
import argparse
import os
import sys
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from fno import FNO1d, FNO2d, FNO3d
from utils import PINODatasetSingle, PINODatasetMult, setup_seed, generate_input, count_params, to_device, timer
from loss import pde_loss

sys.path.append("..")
import metrics


def get_dataset(args):
    dataset_args = args["dataset"]
    if dataset_args["single_file"]:
        print("PINODatasetSingle")
        train_data = PINODatasetSingle(dataset_args["file_name"],
                                       dataset_args["saved_folder"],
                                       initial_step=args["initial_step"],
                                       reduced_resolution=dataset_args["reduced_resolution"],
                                       reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                       reduced_batch=dataset_args["reduced_batch"],
                                       test_ratio=dataset_args["test_ratio"],
                                       if_test=False,
                                       if_grid_norm=dataset_args["if_grid_norm"])
        train_pde = PINODatasetSingle(dataset_args["file_name"],
                                       dataset_args["saved_folder"],
                                       initial_step=args["initial_step"],
                                       reduced_resolution=dataset_args["reduced_resolution_pde"],
                                       reduced_resolution_t=dataset_args["reduced_resolution_pde_t"],
                                       reduced_batch=dataset_args["reduced_batch"],
                                       test_ratio=dataset_args["test_ratio"],
                                       if_test=False,
                                       if_grid_norm=dataset_args["if_grid_norm"])
        val_data = PINODatasetSingle(dataset_args["file_name"],
                                     dataset_args["saved_folder"],
                                     initial_step=args["initial_step"],
                                     reduced_resolution=dataset_args["reduced_resolution"],
                                     reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                     reduced_batch=dataset_args["reduced_batch"],
                                     test_ratio=dataset_args["test_ratio"],
                                     if_test=True,
                                     if_grid_norm=dataset_args["if_grid_norm"])
    else:
        print("PINODatasetMult")
        train_data = PINODatasetMult(dataset_args["file_name"],
                                     dataset_args["saved_folder"],
                                     initial_step=args["initial_step"],
                                     reduced_resolution=dataset_args["reduced_resolution"],
                                     reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                     reduced_batch=dataset_args["reduced_batch"],
                                     test_ratio=dataset_args["test_ratio"],
                                     if_test=False,
                                     if_grid_norm=dataset_args["if_grid_norm"])
        train_pde = PINODatasetMult(dataset_args["file_name"],
                                     dataset_args["saved_folder"],
                                     initial_step=args["initial_step"],
                                     reduced_resolution=dataset_args["reduced_resolution_pde"],
                                     reduced_resolution_t=dataset_args["reduced_resolution_pde_t"],
                                     reduced_batch=dataset_args["reduced_batch"],
                                     test_ratio=dataset_args["test_ratio"],
                                     if_test=False,
                                     if_grid_norm=dataset_args["if_grid_norm"])
        val_data = PINODatasetMult(dataset_args["file_name"],
                                   dataset_args["saved_folder"],
                                   initial_step=args["initial_step"],
                                   reduced_resolution=dataset_args["reduced_resolution"],
                                   reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                   reduced_batch=dataset_args["reduced_batch"],
                                   test_ratio=dataset_args["test_ratio"],
                                   if_test=True,
                                   if_grid_norm=dataset_args["if_grid_norm"])
    return train_data, train_pde, val_data


def get_dataloader(train_data,train_pde, val_data, args):
    dataloader_args = args["dataloader"]
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=dataloader_args["batch_size"],
                              num_workers=dataloader_args["num_workers"],
                              pin_memory=dataloader_args["pin_memory"])
    pde_loader = DataLoader(train_pde, shuffle=True,
                              batch_size=dataloader_args["batch_size"],
                              num_workers=dataloader_args["num_workers"],
                              pin_memory=dataloader_args["pin_memory"])
    val_loader = DataLoader(val_data, shuffle=False,
                            batch_size=dataloader_args["batch_size"],
                            num_workers=dataloader_args["num_workers"],
                            pin_memory=dataloader_args["pin_memory"],
                            drop_last=True)
    return train_loader, pde_loader, val_loader


def get_model(spatial_dim, if_temporal, args):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."

    model_args = args["model"]
    initial_step = args["initial_step"]
    dim=spatial_dim+1 if if_temporal else spatial_dim
    if dim == 1:
        model = FNO1d(in_dim=model_args["in_channels"]*initial_step+1,
                      out_dim= model_args["out_channels"],
                       modes=model_args["modes1"],
                       fc_dim=model_args["fc_dim"],
                       width=model_args["width"],
                       act=model_args["act"])
    elif dim == 2:
        model = FNO2d(in_dim=model_args["in_channels"]*initial_step+2,
                      out_dim=model_args["out_channels"],
                      modes1=model_args["modes1"],
                      modes2=model_args["modes2"],
                      fc_dim=model_args["fc_dim"],
                      width=model_args["width"],
                      act=model_args["act"])
    elif dim == 3:
        model = FNO3d(in_dim=model_args["in_channels"]*initial_step+3,
                      out_dim=model_args["out_channels"],
                      modes1=model_args["modes1"],
                      modes2=model_args["modes2"],
                      modes3=model_args["modes3"],
                      fc_dim=model_args["fc_dim"],
                      width=model_args["width"],
                      act=model_args["act"])
    else:
        raise NotImplementedError
    print("Parameters num: "+str(count_params(model)))
    return model

def train_loop(dataloader, pdeloader, model, if_temporal, optimizer, device, train_args):
    model.train()
    train_loss = 0.0
    train_data=0.0
    train_ic = 0.0
    train_f = 0.0
    data_weight = train_args.get('xy_loss',1.0)
    f_weight = train_args.get('f_loss',0.0)
    ic_weight = train_args.get('ic_loss',0.0)
    loss_fn = nn.MSELoss(reduction="mean")
    # pbar = tqdm(range(len(dataloader)), dynamic_ncols=True, smoothing=0.05)
    dataloader=iter(dataloader)
    pdeloader=iter(pdeloader)
    # train loop
    for i in range(len(dataloader)):

        loss = 0
        # a: (bs, x1, ..., xd, init_t, c), u: (bs, x1, ..., xd, t_train, v)
        a, u, grid = next(dataloader)
        if torch.any(u.isnan()): # ill data
            continue
        bs= a.shape[0]
        a, u, grid = to_device([a,u,grid],device)
        if not if_temporal:  # DarcyFlow
            a=u[...,0:1]
            u=u[...,1:2]
            input = generate_input(a,grid[...,:-1]).squeeze(-2)
        else:
            input = generate_input(a,grid)

        pred= model(input)
        data_loss = loss_fn(pred[...,:u.shape[-1]].reshape([bs,-1]),u.reshape([bs,-1]))


        if f_weight>0:
            a, u, grid=next(pdeloader)
            a, u, grid = to_device([a,u,grid],device)
            if not if_temporal:  # DarcyFlow
                a=u[...,0:1]
                u=u[...,1:2]
                input = generate_input(a,grid[...,:-1]).squeeze(-2)
            else:
                input = generate_input(a,grid)
            pred = model(input)
            ic_loss, f_loss = pde_loss(pred, a, u, train_args, grid)
        else:
            ic_loss=torch.tensor(0.0)
            f_loss=torch.tensor(0.0)
        loss = data_weight*data_loss + ic_weight*ic_loss +f_weight*f_loss
        # if loss.isnan():
        #     continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        train_data+=data_loss.item()
        train_ic+=ic_loss.item()
        train_f+=f_loss.item()

    
    train_loss/=len(dataloader)
    train_data/=len(dataloader)
    train_ic/=len(dataloader)
    train_f/=len(dataloader)

    return train_loss, train_data, train_ic, train_f

@torch.no_grad()
def val_loop(dataloader, model, if_temporal, device, train_args):
    model.eval()
    val_l2 = 0
    # start_time = time.time()
    loss_fn = nn.MSELoss(reduction="mean")
    max_inf=0
    for a, u, grid in dataloader:
        bs = a.size(0)
        a, u, grid = to_device([a,u,grid],device)
        if not if_temporal:  # DarcyFlow
            a=u[...,0:1]
            u=u[...,1:2]
            input = generate_input(a,grid[...,:-1]).squeeze(-2)
        else:
            input = generate_input(a,grid)
        
        pred= model(input)
        data_loss = loss_fn(pred[...,:u.shape[-1]].reshape([bs,-1]),u.reshape([bs,-1]))
        L_inf= torch.norm(pred[...,:u.shape[-1]].reshape([bs,-1])-u.reshape([bs,-1]),p=float('inf')).item()
        max_inf= L_inf if max_inf<L_inf else max_inf
        val_l2+=data_loss.item()
    val_l2/=len(dataloader)  # MSE
    return data_loss, max_inf

@timer
@torch.no_grad()
def test_loop(dataloader, model,if_temporal, device, metric_names=['MSE', 'L2RE', 'MaxError']):
    model.eval()
    # initial result dict
    res_dict = {}
    for name in metric_names:
        res_dict[name] = []
    # test
    for a,u,grid in dataloader:
        a, u, grid = to_device([a,u,grid],device)
        if not if_temporal:  # DarcyFlow
            a=u[...,0:1]
            u=u[...,1:2]
            input = generate_input(a,grid[...,:-1]).squeeze(-2)
        else:
            input = generate_input(a,grid)
        pred= model(input)[...,:u.shape[-1]].reshape(u.shape)

        for name in metric_names:
            metric_fn = getattr(metrics, name)
            res_dict[name].append(metric_fn(pred, u))
    # post process
    for name in metric_names:
        res_list = res_dict[name]
        if name == "MaxError":
            res = torch.stack(res_list, dim=0)
            res, _ = torch.max(res, dim=0)
        else:
            res = torch.cat(res_list, dim=0)
            res = torch.mean(res, dim=0)
        res_dict[name] = res
    return res_dict


def main(args):
    # init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args["model_path"]) if not args["if_training"] or args["continue_training"] else None
    saved_model_name = args['train'].get('save_name',args['model_name'])
    saved_model_name = saved_model_name+ f"_lr{args['optimizer']['lr']}" + f"_bs{args['dataloader']['batch_size']}"
    saved_dir = os.path.join(args["output_dir"], os.path.splitext(args["dataset"]["file_name"])[0])
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    # data get dataloader
    train_data, train_pde,val_data = get_dataset(args)
    train_loader, pde_loader, val_loader = get_dataloader(train_data, train_pde, val_data, args)
    # set some train args
    _, sample, _ = next(iter(val_loader))
    spatial_dim = len(sample.shape) - 3
    if_temporal= True if sample.shape[-2]!=1 else False
    args["train"].update({"dx": train_pde.dx, "dt": train_pde.dt})

    # model
    model = get_model(spatial_dim, if_temporal, args)
    ## if test, load model from checkpoint
    if not args["if_training"]:
        model.load_state_dict(checkpoint["model_state_dict"])
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        model.eval()
        print("start testing...")
        print(f"testset batch num. is {len(val_loader)} , batchsize={args['dataloader']['batch_size']}")
        # time_list=[]
        # for i in range(10):
        res, time = test_loop(val_loader, model, if_temporal, device)
        #     time_list.append(time)
        # time_list.pop(time_list.index(max(time_list)))
        # time_list.pop(time_list.index(min(time_list)))
        
        # print("average time per loop:",sum(time_list)/8)
        # print(f"average time per batch:{sum(time_list)/8/len(val_loader): .4f}" )
        for name,val in res.items():
            if val.ndim==1:
                for i, v in enumerate(val):
                    print(f"{name}[{i}]: {v:.6f}")
            else:
                print(f"{name}: {val:.6f}")
        print("Done")
        return
    ## if continue training, resume model from checkpoint
    if args["continue_training"]:
        print(f"continue training, load checkpoint from {args['model_path']}")
        model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # optimizer
    optim_args = args["optimizer"]
    optim_name = optim_args.pop("name")
    ## if continue training, resume optimizer and scheduler from checkpoint
    if args["continue_training"]:
        optimizer = getattr(torch.optim, optim_name)([{'params': model.parameters(), 'initial_lr': optim_args["lr"]}], **optim_args)
    else:
        optimizer = getattr(torch.optim, optim_name)(model.parameters(), **optim_args)

    # scheduler
    start_epoch = 0
    min_val_loss = torch.inf
    if args["continue_training"]:
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['loss']
    sched_args = args["scheduler"]
    sched_name = sched_args.pop("name")
    scheduler = getattr(torch.optim.lr_scheduler, sched_name)(optimizer, last_epoch=start_epoch-1, **sched_args)

    # train
    print(f"start training from epoch {start_epoch}")
    pbar=tqdm(range(start_epoch,args["epochs"]), dynamic_ncols=True, smoothing=0.05)
    loss_curve=[]
    for epoch in pbar:
        ## train loop
        train_loss, train_data, train_ic, train_f = train_loop(train_loader,pde_loader, model, if_temporal, optimizer, device, args["train"])
        scheduler.step()
        pbar.set_description(f"[Epoch {epoch}] train_loss: {train_loss:.5e}, data_loss: {train_data:.5e}, train_f: {train_f:.5e},train_ic:{train_ic:.5e}")
        # print(f"[Epoch {epoch}] train_loss: {train_loss}, data_loss: {train_data}, train_f: {train_f},train_ic:{train_ic}")
        loss_curve.append(train_loss)
        ## save latest
        saved_path = os.path.join(saved_dir, saved_model_name)
        model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        torch.save({"epoch": epoch+1, "loss": min_val_loss,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict()
            }, saved_path + "-latest.pt")
        ## validate
        if (epoch+1) % args["save_period"] == 0:
            val_loss, L_inf = val_loop(val_loader, model, if_temporal, device, args["train"])
            print(f"[Epoch {epoch}] val_loss: {val_loss:.5e}, L_inf: {L_inf:.5e}")
            print("================================================",flush=True)
            if val_loss < min_val_loss:
                ### save best
                torch.save({"epoch": epoch+1, "loss": min_val_loss,
                    "model_state_dict": model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    }, saved_path + "-best.pt")
    torch.save({"loss_curve":loss_curve},saved_path+"_loss_curve.pt")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    print(args)
    setup_seed(args["seed"])
    main(args)
