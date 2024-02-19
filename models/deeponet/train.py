# coding=utf-8
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from deeponet import DeepONetCartesianProd2D, DeepONetCartesianProd1D
from utils import DeepOnetDatasetSingle, DeepOnetDatasetMult, setup_seed, count_params, to_device, timer
from tqdm import tqdm

sys.path.append("..")
import metrics

def get_dataset(args):
    dataset_args = args["dataset"]
    if dataset_args["single_file"]:
        print("DeepOnetDatasetSingle")
        train_data = DeepOnetDatasetSingle(dataset_args["file_name"],
                                       dataset_args["saved_folder"],
                                       initial_step=args["initial_step"],
                                       reduced_resolution=dataset_args["reduced_resolution"],
                                       reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                       reduced_batch=dataset_args["reduced_batch"],
                                       test_ratio=dataset_args["test_ratio"],
                                       if_test=False,
                                       )
        val_data = DeepOnetDatasetSingle(dataset_args["file_name"],
                                     dataset_args["saved_folder"],
                                     initial_step=args["initial_step"],
                                     reduced_resolution=dataset_args["reduced_resolution"],
                                     reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                     reduced_batch=dataset_args["reduced_batch"],
                                     test_ratio=dataset_args["test_ratio"],
                                     if_test=True,
                                     )
    else:
        print("DeepOnetDatasetMult")
        train_data = DeepOnetDatasetMult(dataset_args["file_name"],
                                     dataset_args["saved_folder"],
                                     initial_step=args["initial_step"],
                                     reduced_resolution=dataset_args["reduced_resolution"],
                                     reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                     reduced_batch=dataset_args["reduced_batch"],
                                     test_ratio=dataset_args["test_ratio"],
                                     if_test=False,
                                     )
        val_data = DeepOnetDatasetMult(dataset_args["file_name"],
                                   dataset_args["saved_folder"],
                                   initial_step=args["initial_step"],
                                   reduced_resolution=dataset_args["reduced_resolution"],
                                   reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                   reduced_batch=dataset_args["reduced_batch"],
                                   test_ratio=dataset_args["test_ratio"],
                                   if_test=True,
                                   )
    return train_data, val_data


def get_dataloader(train_data, val_data, args):
    dataloader_args = args["dataloader"]
    train_loader = DataLoader(train_data, shuffle=True,
                              batch_size=dataloader_args["batch_size"],
                              num_workers=dataloader_args["num_workers"],
                              pin_memory=dataloader_args["pin_memory"])
    val_loader = DataLoader(val_data, shuffle=False,
                            batch_size=dataloader_args["batch_size"],
                            num_workers=dataloader_args["num_workers"],
                            pin_memory=dataloader_args["pin_memory"],
                            drop_last=True)
    return train_loader, val_loader


def get_model(spatial_dim, if_temporal, args):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."

    model_args = args["model"]
    initial_step = args["initial_step"]
    # dim=spatial_dim+1 if if_temporal else spatial_dim
    if spatial_dim == 1:
        model = DeepONetCartesianProd1D(size=model_args["input_size"],
                    in_channel_branch=model_args["in_channels"]*initial_step,
                      query_dim= model_args["query_dim"],
                      out_channel=model_args["out_channels"],
                      activation=model_args["act"],
                      base_model=model_args["base_model"]
                      )
    elif spatial_dim == 2:
        model = DeepONetCartesianProd2D(size=model_args["input_size"],
                    in_channel_branch=model_args["in_channels"]*initial_step,
                      query_dim= model_args["query_dim"],
                      out_channel=model_args["out_channels"],
                      activation=model_args["act"],
                      base_model=model_args["base_model"]
                      )
    elif spatial_dim == 3:
        pass
    else:
        raise NotImplementedError
    print("Parameters num: "+str(count_params(model)))
    return model


def train_loop(train_loader, model, initial_step, if_temporal, optimizer, device, train_args):
    model.train()
    train_loss = 0
    train_l_inf = 0
    loss_fn = nn.MSELoss(reduction="mean")
    # pbar = tqdm(range(len(dataloader)), dynamic_ncols=True, smoothing=0.05)
    # train loop
    for x, y, grid in train_loader:

        loss = 0
        # a: (bs, x1, ..., xd, init_t, c), u: (bs, x1, ..., xd, t_train, v)
        if torch.any(y.isnan()): # ill data
            continue
        bs= y.shape[0]
        t_train= y.shape[-2]
        grid=grid[0]  # reduce batch
        x, y, grid = to_device([x, y, grid],device)
        # reshape input
        input_shape = list(x.shape)[:-2] # (bs, x1, ..., xd)
        input_shape.append(-1) # (bs, x1, ..., xd, -1)
        if if_temporal:
            if train_args["training_type"]=='single':
                pred = model((x.reshape(input_shape),grid))
                loss+= loss_fn(pred.reshape(bs,-1),y.reshape(bs,-1))
                train_loss += loss.item()
            else:  # autoregressive
                grid=grid[...,0,:-1]
                pred = y[..., :initial_step, :] # (bs, x1, ..., xd, init_t, v)

                for t in range(initial_step, t_train):
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    model_input = x.reshape(input_shape)
                    # Extract target at current time step
                    target = y[..., t:t+1, :]
                    # Model run
                    model_output = model((model_input, grid)).unsqueeze(-2)
                    # Loss calculation
                    _loss=loss_fn(model_output.reshape(bs, -1), target.reshape(bs, -1))
                    loss += _loss
                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, model_output), -2)
                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    x = torch.cat((x[..., 1:, :], model_output), dim=-2)

                train_loss += loss_fn(pred.reshape(bs, -1), y.reshape(bs, -1)).item()

            train_l_inf = torch.max((torch.abs(pred.reshape(bs, -1) - y.reshape(bs, -1))))
        else:  # DarcyFlow
            a=y[...,0,0:1]
            u=y[...,0,1:2]
            grid=grid[...,0,:-1]
            pred= model((a,grid))
            loss += loss_fn(pred.reshape([bs,-1]),u.reshape([bs,-1]))
            train_loss += loss.item()
            train_l_inf = max(train_l_inf, torch.max((torch.abs(pred.reshape(bs, -1) - u.reshape(bs, -1)))).item())

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()
    
    train_loss/=len(train_loader)
    return train_loss, train_l_inf


@torch.no_grad()
def val_loop(val_loader, model, initial_step, if_temporal, device, train_args):
    model.eval()
    # start_time = time.time()
    val_loss = 0
    val_l_inf = 0
    loss_fn = nn.MSELoss(reduction="mean")
    # pbar = tqdm(range(len(dataloader)), dynamic_ncols=True, smoothing=0.05)
    # val loop
    for x, y, grid in val_loader:
        # x: (bs, x1, ..., xd, init_t, c), y: (bs, x1, ..., xd, t_train, v)
        if torch.any(y.isnan()): # ill data
            continue
        bs= y.shape[0]
        t_train= y.shape[-2]
        grid=grid[0]  # reduce batch
        x, y, grid = to_device([x, y, grid],device)
        pred = y[..., :initial_step, :] # (bs, x1, ..., xd, init_t, v)
        # reshape input
        input_shape = list(x.shape)[:-2] # (bs, x1, ..., xd)
        input_shape.append(-1) # (bs, x1, ..., xd, -1)

        if if_temporal:
            if train_args["training_type"]=='single':
                pred = model((x.reshape(input_shape),grid))
                val_loss += loss_fn(pred.reshape(bs,-1),y.reshape(bs,-1)).item()
            else:  # autoregressive loop
                grid=grid[...,0,:-1]
                for t in range(initial_step, t_train):
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    model_input = x.reshape(input_shape)
                    # Model run
                    model_output = model((model_input, grid)).unsqueeze(-2)
                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, model_output), -2)
                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    x = torch.cat((x[..., 1:, :], model_output), dim=-2)
                val_loss += loss_fn(pred.reshape(bs, -1), y.reshape(bs, -1)).item()
            val_l_inf = max(val_l_inf, torch.max((torch.abs(pred.reshape(bs, -1) - y.reshape(bs, -1)))))

        else:  # DarcyFlow
            a=y[...,0,0:1]
            u=y[...,0,1:2]
            grid=grid[...,0,:-1]
            pred= model((a,grid))
            val_loss += loss_fn(pred.reshape([bs,-1]),u.reshape([bs,-1])).item()
            val_l_inf = max(val_l_inf, torch.max((torch.abs(pred.reshape(bs, -1) - u.reshape(bs, -1)))))

    val_loss/=len(val_loader) # MSE
    return val_loss, val_l_inf

@timer
@torch.no_grad()
def test_loop(dataloader, model, initial_step, if_temporal, device, train_args, metric_names=['MSE', 'L2RE', 'MaxError']):
    model.eval()
    # initial result dict
    res_dict = {}
    for name in metric_names:
        res_dict[name] = []
    # test
    for x, y, grid in dataloader:
        if torch.any(y.isnan()): # ill data
            continue
        t_train= y.shape[-2]
        grid=grid[0]  # reduce batch
        x, y, grid = to_device([x, y, grid],device)
        pred = y[..., :initial_step, :] # (bs, x1, ..., xd, init_t, v)
        # reshape input
        input_shape = list(x.shape)[:-2] # (bs, x1, ..., xd)
        input_shape.append(-1) # (bs, x1, ..., xd, -1)
        if if_temporal:
            if train_args["training_type"]=='single':
                pred = model((x.reshape(input_shape),grid))
            else:  # autoregressive loop
                grid=grid[...,0,:-1]
                for t in range(initial_step, t_train):
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    model_input = x.reshape(input_shape)
                    # Model run
                    model_output = model((model_input, grid)).unsqueeze(-2)
                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, model_output), -2)
                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    x = torch.cat((x[..., 1:, :], model_output), dim=-2)
        else: # DarcyFlow
            a=y[...,0, 0:1]
            y=y[...,0, 1:2]
            grid=grid[...,0,:-1]
            pred= model((a,grid)).squeeze(-2)

        for name in metric_names:
            metric_fn = getattr(metrics, name)
            res_dict[name].append(metric_fn(pred, y))
    # post process
    for name in metric_names:
        res_list = res_dict[name]
        if name == "MaxError":
            res = torch.stack(res_list, dim=0)
            res, _ = torch.max(res, dim=0)
        else:
            res = torch.cat(res_list, dim=0)
            res = torch.mean(res,dim=0)
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
    train_data, val_data = get_dataset(args)
    train_loader, val_loader = get_dataloader(train_data, val_data, args)
    # set some train args
    _, sample, _ = next(iter(val_loader))
    spatial_dim = len(sample.shape) - 3
    if_temporal= True if sample.shape[-2]!=1 else False
    args["train"].update({"dx": train_data.dx, "dt": train_data.dt})

    # model
    model = get_model(spatial_dim, if_temporal, args)
    ## if test, load model from checkpoint
    if not args["if_training"]:
        model.load_state_dict(checkpoint["model_state_dict"])
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        model.eval()
        # TODO: test code
        print("start testing...")
        print(f"testset size is {len(val_loader)} , batchsize={args['dataloader']['batch_size']}")
        # time_list=[]
        # for i in range(10):
        res, time = test_loop(val_loader, model, args["initial_step"], if_temporal, device, args["train"])
            # time_list.append(time)
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
        train_loss, train_l_inf = train_loop(train_loader, model, args["initial_step"],if_temporal, optimizer, device, args["train"])
        scheduler.step()
        pbar.set_description(f"[Epoch {epoch}] train_loss: {train_loss:.5e}, l_inf: {train_l_inf:.5e}")
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
            val_loss, L_inf = val_loop(val_loader, model, args["initial_step"], if_temporal, device, args["train"])
            print(f"[Epoch {epoch}] val_loss: {val_loss:.5e}, L_inf: {L_inf:.5e}")
            print("================================================",flush=True)
            if val_loss < min_val_loss:
                ### save best
                min_val_loss=val_loss
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
