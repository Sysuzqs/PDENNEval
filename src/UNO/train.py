# coding=utf-8
import argparse
import numpy as np
import os
import sys
from timeit import default_timer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from utils import *
from uno import UNO1d, UNO2d, UNO3d, UNO_maxwell

sys.path.append("..")
import metrics

def get_dataset(args):
    dataset_args = args["dataset"]
    if(dataset_args["single_file"]):
        print("UNODatasetSingle")
        train_data = UNODatasetSingle(dataset_args["file_name"],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                reduced_batch=dataset_args["reduced_batch"],
                                initial_step=args["initial_step"],
                                if_test=False,
                                saved_folder = dataset_args["saved_folder"]
                                )
        val_data = UNODatasetSingle(dataset_args["file_name"],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                reduced_batch=dataset_args["reduced_batch"],
                                initial_step=args["initial_step"],
                                if_test=True,
                                saved_folder = dataset_args["saved_folder"]
                                )
    else:
        print("UNODatasetMult")
        train_data = UNODatasetMult(dataset_args["file_name"],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                reduced_batch=dataset_args["reduced_batch"],
                                initial_step=args["initial_step"],
                                if_test=False,
                                saved_folder = dataset_args["saved_folder"]
                                )
        val_data = UNODatasetMult(dataset_args["file_name"],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                reduced_batch=dataset_args["reduced_batch"],
                                initial_step=args["initial_step"],
                                if_test=True,
                                saved_folder = dataset_args["saved_folder"]
                                )
    return train_data, val_data

def get_dataloader(train_data, val_data, args):
    dataloader_args = args["dataloader"]
    train_loader = DataLoader(train_data, shuffle=True, multiprocessing_context = 'spawn', generator=torch.Generator(device = 'cpu'), **dataloader_args)
    if args["if_training"]:
        val_loader = DataLoader(val_data, shuffle=False, multiprocessing_context = 'spawn', generator=torch.Generator(device = 'cpu'), **dataloader_args)
    else:
        val_loader = DataLoader(val_data, shuffle=False, drop_last=True, **dataloader_args)
    return train_loader, val_loader

def get_model(spatial_dim, args):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."

    model_args = args["model"]
    initial_step = args["initial_step"]
    if args['pde_name'] == "3D_Maxwell": #maxwell, input without grid
        model = UNO_maxwell(num_channels=model_args["num_channels"],
                      width=model_args["width"],
                      initial_step=initial_step)
    elif spatial_dim == 1:
        model = UNO1d(num_channels=model_args["num_channels"],
                      width=model_args["width"],
                      initial_step=initial_step)
    elif spatial_dim == 2:
        model = UNO2d(num_channels=model_args['num_channels'],
                      width = model_args['width'],
                      initial_step=initial_step)
    elif spatial_dim == 3:
        model = UNO3d(num_channels=model_args["num_channels"],
                      width = model_args['width'],
                      initial_step=initial_step)
    return model

def train_loop(model, train_loader, initial_step, t_train, optimizer, loss_fn, scheduler, device, args):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_l_inf = 0
    for x, y, grid in train_loader:
        # batch_size = x.size(0)
        loss = 0
        if args['pde_name'] == "3D_Maxwell": #maxwell, we get x, y, global maximum
            grid = grid[0] #maximum
            x = x.detach().clone() / grid #scale
            y = y.detach().clone() / grid
        x = x.to(device) # x: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
        y = y.to(device) # y: target tensor [b, x1, ..., xd, t, v]
        grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims]
        # initialize the prediction tensor
        pred = y[..., :initial_step, :] # (bs, x1, ..., xd, init_t, v)
        # reshape input
        input_shape = list(x.shape)[:-2] # (bs, x1, ..., xd)
        input_shape.append(-1) # (bs, x1, ..., xd, -1)

        if args["training_type"] in ['autoregressive']:
            # Autoregressive loop
            for t in range(initial_step, t_train):
                # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                model_input = x.reshape(input_shape)
                # Extract target at current time step
                target = y[..., t:t+1, :]
                # Model run
                model_output = model(model_input, grid)
                
                # Loss calculation
                _batch = model_output.size(0)
                loss += loss_fn(model_output.reshape(_batch, -1), target.reshape(_batch, -1))
                # Concatenate the prediction at current time step into the
                # prediction tensor
                pred = torch.cat((pred, model_output), -2)
                # Concatenate the prediction at the current time step to be used
                # as input for the next time step
                x = torch.cat((x[..., 1:, :], model_output), dim=-2)
            _batch = y.size(0)
            train_l2 += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1)).item()
            train_l_inf = max(train_l_inf, torch.max((torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1)))))

        if args["training_type"] in ['single']:
            x = x[..., 0 , :]
            y = y[..., t_train-1:t_train, :]
            pred = model(x, grid)
            _batch = y.size(0)
            loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
            train_l2 += loss.item()
            train_l_inf = max(train_l_inf, torch.max((torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1)))))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t2 = default_timer()
    return train_l2, train_l_inf, t2 - t1

def val_loop(val_loader, model, loss_fn, device, training_type, t_train, initial_step):
    model.eval()
    val_l2_full = 0
    val_l_inf_full = 0
    with torch.no_grad():
        for x, y, grid in val_loader:
            loss = 0
            x = x.to(device)
            y = y.to(device)
            grid = grid.to(device)
            
            if training_type == 'autoregressive':
                pred = y[..., :initial_step, :]
                input_shape = list(x.shape)[:-2] # (bs, x1, ..., xd)
                input_shape.append(-1) # (bs, x1, ..., xd, -1)
                for t in range(initial_step, y.shape[-2]):
                    model_input = x.reshape(input_shape)
                    target = y[..., t:t+1, :]
                    model_output = model(model_input, grid)
                    _batch = model_output.size(0)
                    loss += loss_fn(model_output.reshape(_batch, -1), target.reshape(_batch, -1))
                    pred = torch.cat((pred, model_output), -2)
                    x = torch.cat((x[..., 1:, :], model_output), dim=-2)
    
                _batch = y.size(0)
                _pred = pred[..., initial_step:t_train, :]
                _y = y[..., initial_step:t_train, :]
                val_l2_full += loss_fn(_pred.reshape(_batch, -1), _y.reshape(_batch, -1)).item()
                val_l_inf_full = max(torch.max(torch.abs(_pred.reshape(_batch, -1) - _y.reshape(_batch, -1))).item(), val_l_inf_full)

            if training_type == 'single':
                x = x[..., 0 , :]
                y = y[..., t_train-1:t_train, :]
                pred = model(x, grid)
                _batch = y.size(0)
                loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
    
                val_l2_full += loss.item()
                val_l_inf_full = max(torch.max(torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1))).item(), val_l_inf_full)
    return val_l2_full, val_l_inf_full

def test_loop(dataloader, model, device, initial_step, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError']):
    model.eval()
    # initial result dict
    res_dict = {}
    for name in metric_names:
        res_dict[name] = []
    # test
    for x, y, grid in dataloader:
        if args['pde_name'] == "3D_Maxwell": #maxwell, we get x, y, global maximum
            grid = grid[0] #maximum
            x = x.detach().clone() / grid #scale
            y = y.detach().clone() / grid
        x = x.to(device)
        y = y.to(device)
        grid = grid.to(device)
        pred = y[..., :initial_step, :]
        input_shape = list(x.shape)[:-2] # (bs, x1, ..., xd)
        input_shape.append(-1) # (bs, x1, ..., xd, -1)
        for t in range(initial_step, y.shape[-2]):
            model_input = x.reshape(input_shape)
            with torch.no_grad():
                model_output = model(model_input, grid)
            pred = torch.cat((pred, model_output), dim=-2)
            x = torch.cat((x[..., 1:, :], model_output), dim=-2)
        for name in metric_names:
            metric_fn = getattr(metrics, name)
            res_dict[name].append(metric_fn(pred[..., initial_step:, :], y[..., initial_step:, :]))

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
    #init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args["model_path"]) if not args["if_training"] or args["continue_training"] else None
    saved_model_name = args["model_name"] + f"_lr{args['optimizer']['lr']}" + f"_bs{args['dataloader']['batch_size']}"
    saved_dir = os.path.join(args["output_dir"], os.path.splitext(args["dataset"]["file_name"])[0])
    print(saved_dir)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    dataset_args = args["dataset"]
    if(dataset_args["single_file"]):
        saved_model_name = dataset_args["file_name"][:-5] + '_UNO'
    else:
        saved_model_name = dataset_args["file_name"] + '_UNO'
    
    # data get dataloader
    train_data, val_data = get_dataset(args)
    train_loader, val_loader = get_dataloader(train_data, val_data, args)

    # set some train args
    _, sample, _ = next(iter(val_loader))
    spatial_dim = len(sample.shape) - 3
    initial_step = args["initial_step"]
    t_train = min(args["t_train"], sample.shape[-2])

    #model
    model = get_model(spatial_dim, args)
    ##
    if not args["if_training"]:
        print(f"Test mode, load checkpoint from {args['model_path']}")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        # TODO: test code
        print("start testing...")
        res = test_loop(val_loader, model, device, initial_step)
        for u, v in res.items():
            dim = len(v)
            if dim == 1:
                print(u, "{0:.6f}".format(v.item()))
            else:
                for i in range(dim):
                    if i == 0:
                        print(u, "\t{0:.6f}".format(v[i].item()), end='\t')
                    else:
                        print("{0:.6f}".format(v[i].item()), end='\t')
                print("")
        print("Done")
        return
    ## if continue training, resume model from checkpoint
    if args["continue_training"]:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device) 
    model.train()

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

    # loss function
    loss_fn = nn.MSELoss(reduction="mean")

    # save loss history
    loss_history = []
    if args["continue_training"]:
        loss_history = np.load('./log/loss/' + args['pde_name'] + '_loss_history.npy')
        loss_history = loss_history.tolist()

     # train loop
    print("start training...")
    total_time = 0
    for epoch in range(start_epoch, args["epochs"]):
        train_l2, train_l_inf, time = train_loop(model,train_loader, initial_step, t_train, optimizer, loss_fn, scheduler, device, args)
        scheduler.step()
        total_time += time
        loss_history.append(train_l2)
        print(f"[Epoch {epoch}] train_l2: {train_l2}, train_l_inf: {train_l_inf}, time_spend: {time:.3f}")
        ## save latest
        saved_path = os.path.join(saved_dir, saved_model_name)
        model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        torch.save({"epoch": epoch+1, "loss": min_val_loss,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict()
            }, saved_path + "-latest.pt")
        if (epoch+1) % args["save_period"] == 0:
            print("====================validate====================")
            val_l2_full, val_l_inf = val_loop(val_loader, model, loss_fn, device, args["training_type"], t_train, initial_step)
            print(f"[Epoch {epoch}] val_l2_full: {val_l2_full} val_l_inf: {val_l_inf}")
            print("================================================")
            if val_l2_full < min_val_loss:
                min_val_loss = val_l2_full
                ## save best
                torch.save({"epoch": epoch + 1, "loss": min_val_loss,
                    "model_state_dict": model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    }, saved_path + "-best.pt")
    print("Done.")
    loss_history = np.array(loss_history)
    np.save('./log/loss/' + args['pde_name'] + '_loss_history.npy', loss_history)
    print("avg_time : {0:.5f}".format(total_time / (args["epochs"] - start_epoch)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    setup_seed(args["seed"])
    print(args)
    main(args)
# print(torch.cuda.device_count())