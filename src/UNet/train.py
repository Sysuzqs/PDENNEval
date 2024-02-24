# coding=utf-8
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from unet import UNet1d, UNet2d, UNet3d
from utils import *

sys.path.append("..")
import metrics
 

def get_dataset(args):
    dataset_args = args["dataset"]
    if dataset_args["single_file"]:
        print("UNetDatasetSingle")
        train_data = UNetDatasetSingle(dataset_args["file_name"],
                                       dataset_args["saved_folder"],
                                       initial_step=args["initial_step"],
                                       reduced_resolution=dataset_args["reduced_resolution"],
                                       reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                       reduced_batch=dataset_args["reduced_batch"],
                                       test_ratio=dataset_args["test_ratio"],
                                       if_test=False)
        val_data = UNetDatasetSingle(dataset_args["file_name"],
                                     dataset_args["saved_folder"],
                                     initial_step=args["initial_step"],
                                     reduced_resolution=dataset_args["reduced_resolution"],
                                     reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                     reduced_batch=dataset_args["reduced_batch"],
                                     test_ratio=dataset_args["test_ratio"],
                                     if_test=True)
    else:
        print("UNetDatasetMult")
        train_data = UNetDatasetMult(dataset_args["file_name"],
                                     dataset_args["saved_folder"],
                                     initial_step=args["initial_step"],
                                     reduced_resolution=dataset_args["reduced_resolution"],
                                     reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                     reduced_batch=dataset_args["reduced_batch"],
                                     test_ratio=dataset_args["test_ratio"],
                                     if_test=False)
        val_data = UNetDatasetMult(dataset_args["file_name"],
                                   dataset_args["saved_folder"],
                                   initial_step=args["initial_step"],
                                   reduced_resolution=dataset_args["reduced_resolution"],
                                   reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                   reduced_batch=dataset_args["reduced_batch"],
                                   test_ratio=dataset_args["test_ratio"],
                                   if_test=True)
    return train_data, val_data


def get_dataloader(train_data, val_data, args):
    dataloader_args = args["dataloader"]
    train_loader = DataLoader(train_data, shuffle=True, **dataloader_args)
    if args["if_training"]:
        val_loader = DataLoader(val_data, shuffle=False, **dataloader_args)
    else:
        val_loader = DataLoader(val_data, shuffle=False, drop_last=True, **dataloader_args)
    return train_loader, val_loader


def get_model(spatial_dim, args):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."
    model_args = args["model"]
    initial_step = args["initial_step"]
    if spatial_dim == 1:
        model = UNet1d(model_args["in_channels"]*initial_step, model_args["out_channels"], model_args["init_features"])
    elif spatial_dim == 2:
        model = UNet2d(model_args["in_channels"]*initial_step, model_args["out_channels"], model_args["init_features"])
    else: # spatial_dim == 3
        model = UNet3d(model_args["in_channels"]*initial_step, model_args["out_channels"], model_args["init_features"])
    return model


def train_loop(dataloader, model, loss_fn, optimizer, device, training_type, t_train, initial_step, unroll_step):
    model.train()
    step_losses = []
    full_losses = []
    # train loop
    start_time = time.time() # tic
    for x, y in dataloader:
        batch_size = x.size(0)
        loss = 0
        x = x.to(device) # (bs, x1, ..., xd, init_t, v)
        y = y.to(device) # (bs, x1, ..., xd, t_train, v)
        # initialize the prediction tensor
        pred = y[..., :initial_step, :] # (bs, x1, ..., xd, init_t, v)
        # reshape input
        input_shape = list(x.shape)[:-2] # (bs, x1, ..., xd)
        input_shape.append(-1) # (bs, x1, ..., xd, -1)
        # forward
        for t in range(initial_step, t_train):
            if training_type != "autoregressive":
                x = y[..., t-initial_step:t, :] # one step training
            model_input = x.reshape(input_shape) # (bs, x1, ..., xd, init_t*v)
            # permute input
            input_permute = [0, -1]
            input_permute.extend([i for i in range(1, len(model_input.shape)-1)]) # input permute: [0, -1, 1, 2, ..., d]
            # model input: (bs, x1, ..., xd, init_t*v) -> (bs, init_t*v, x1, ..., xd)
            model_input = model_input.permute(input_permute)
            # Define output permute
            output_permute = [0]
            output_permute.extend([i for i in range(2, len(model_input.shape))])
            output_permute.append(1) # output permute: [0, 2, 3, ..., d+1, 1]
            # Extract target of current time step
            target = y[..., t:t+1, :] # (bs, x1, ..., xd, 1, v)
            if t < t_train - unroll_step:
                with torch.no_grad():
                    # model_output: (bs, init_t*v, x1, ..., xd) -> (bs, v, x1, ..., xd) -> (bs, x1, ..., xd, 1, v)
                    model_output = model(model_input).permute(output_permute).unsqueeze(-2)
            else:
                model_output = model(model_input).permute(output_permute).unsqueeze(-2)
                loss += loss_fn(model_output.reshape(batch_size, -1), target.reshape(batch_size, -1))
            pred = torch.cat((pred, model_output), dim=-2)
            x = torch.cat((x[..., 1:, :], model_output), dim=-2) # (bs, init_t*v, x1, ..., xd)
        step_losses.append(loss.item())
        full_losses.append(loss_fn(pred.reshape(batch_size, -1), y[..., :t_train, :].reshape(batch_size, -1)).item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time() # toc
    time_spend = end_time - start_time
    
    return np.mean(step_losses), np.mean(full_losses), time_spend


def val_loop(dataloader, model, loss_fn, device, t_train, initial_step):
    model.eval()
    full_losses = []
    for x, y in dataloader:
        batch_size = x.size(0)
        loss = 0
        x = x.to(device) # (bs, x1, ..., xd, init_t, v)
        y = y.to(device) # (bs, x1, ..., xd, t_train, v)
        # initialize the prediction tensor
        pred = y[..., :initial_step, :] # (bs, x1, ..., xd, init_t, v)
        # reshape input
        input_shape = list(x.shape)[:-2] # (bs, x1, ..., xd)
        input_shape.append(-1) # (bs, x1, ..., xd, -1)
        for t in range(initial_step, t_train):
            model_input = x.reshape(input_shape) # (bs, x1, ..., xd, init_t*v)
            # permute input
            input_permute = [0, -1]
            input_permute.extend([i for i in range(1, len(model_input.shape)-1)]) # input permute: [0, -1, 1, 2, ..., d]
            # model input: (bs, x1, ..., xd, init_t*v) -> (bs, init_t*v, x1, ..., xd)
            model_input = model_input.permute(input_permute)
            # Define output permute
            output_permute = [0]
            output_permute.extend([i for i in range(2, len(model_input.shape))]) 
            output_permute.append(1) # output permute: [0, 2, 3, ..., d+1, 1]
            # Extract target of current time step
            target = y[..., t:t+1, :] # (bs, x1, ..., xd, 1, v)
            with torch.no_grad():
                # model_output: (bs, init_t*v, x1, ..., xd) -> (bs, v, x1, ..., xd) -> (bs, x1, ..., xd, 1, v)
                model_output = model(model_input).permute(output_permute).unsqueeze(-2)
            loss += loss_fn(model_output.reshape(batch_size, -1), target.reshape(batch_size, -1))
            pred = torch.cat((pred, model_output), dim=-2)
            x = torch.cat((x[..., 1:, :], model_output), dim=-2) # (bs, init_t*v, x1, ..., xd)
        full_losses.append(loss_fn(pred.reshape(batch_size, -1), y[..., :t_train, :].reshape(batch_size, -1)).item())

    return np.mean(full_losses)


def test_loop(dataloader, model, device, initial_step, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError']):
    model.eval()
    res_dict = {}
    # create dict to record results
    for name in metric_names:
        res_dict[name] = []
    # test loop
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        pred = y[..., :initial_step, :]
        input_shape = list(x.shape)[:-2]
        input_shape.append(-1) # (bs, x1, ..., xd, -1)
        for _ in range(initial_step, y.shape[-2]):
            model_input = x.reshape(input_shape)
            input_permute = [0, -1]
            input_permute.extend([i for i in range(1, len(model_input.shape)-1)])
            model_input = model_input.permute(input_permute)
            output_permute = [0]
            output_permute.extend([i for i in range(2, len(model_input.shape))]) 
            output_permute.append(1)
            with torch.no_grad():
                model_output = model(model_input).permute(output_permute).unsqueeze(-2)
            pred = torch.cat((pred, model_output), dim=-2)
            x = torch.cat((x[..., 1:, :], model_output), dim=-2)
        # compute metric
        for name in metric_names:
            metric_fn = getattr(metrics, name)
            res_dict[name].append(metric_fn(pred[..., initial_step:, :], y[..., initial_step:, :]))
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
    setup_seed(args["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args["model_path"]) if not args["if_training"] or args["continue_training"] else None
    saved_model_name = (args["model_name"] + 
                        f"_lr{args['optimizer']['lr']}" + 
                        f"_bs{args['dataloader']['batch_size']}" + 
                        f"_wd{args['optimizer']['weight_decay']}" +
                        f"_{args['training_type']}")
    saved_dir = os.path.join(args["output_dir"], os.path.splitext(args["dataset"]["file_name"])[0])
    # prepare directory 
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    # data and dataloader
    train_data, val_data = get_dataset(args)
    train_loader, val_loader = get_dataloader(train_data, val_data, args)
    
    # set training args
    x, y = next(iter(val_loader))
    spatial_dim = len(x.shape) - 3
    initial_step = args["initial_step"]
    t_train = y.shape[-2]
    unroll_step = t_train - initial_step if t_train - initial_step < args["unroll_step"] else args["unroll_step"]
    if args["training_type"] == "autoregressive":
        if args["pushforward"]:
            saved_model_name += '_PF' + str(unroll_step)
            print("Training type: autoregressive + pushforward trick.")
        else:
            unroll_step = t_train - initial_step
            saved_model_name += '_AR'
            print("Training type: autoregressive")
    else:
        unroll_step = t_train - initial_step
        saved_model_name += "_1step"
        print("Training type: one step")
    print(f"Unroll step: {unroll_step}")

    # visualize
    if args["if_training"] and args["tensorboard"]:
        log_path = os.path.join(args['log_dir'], os.path.splitext(args["dataset"]["file_name"])[0], saved_model_name)
        writer = SummaryWriter(log_path)

    # if spatial_dim == 3:
    #     torch.backends.cudnn.enabled = False

    # model
    model = get_model(spatial_dim, args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of model parameters to train:", total_params)
    # if test, load model from checkpoint
    if not args["if_training"]:
        print(f"Test mode, load checkpoint from {args['model_path']}")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Best epoch: {checkpoint['epoch']}")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        print("Start testing")
        res = test_loop(val_loader, model, device, initial_step)
        print(res)
        print("Done")
        return
    # if continue training, resume model from checkpoint
    if args["continue_training"]:
        print(f"Continue training, load checkpoint from {args['model_path']}")
        model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # optimizer
    optim_args = args["optimizer"]
    optim_name = optim_args.pop("name")
    # if continue training, resume optimizer and scheduler from checkpoint
    if args["continue_training"]:
        optimizer = getattr(torch.optim, optim_name)(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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

    # train
    print(f"Start training from epoch {start_epoch}")
    total_time = 0
    for epoch in range(start_epoch, args["epochs"]):
        # train loop
        train_step_loss, train_full_loss, time_spend = train_loop(train_loader, model, loss_fn, optimizer, device, args["training_type"], t_train, initial_step, unroll_step)
        total_time += time_spend
        scheduler.step()
        if args["tensorboard"]:
            writer.add_scalar('Train Loss/train', train_step_loss, epoch)
            writer.add_scalar('Full Loss/train', train_full_loss, epoch)
        print(f"[Epoch {epoch}] train_l2_step: {train_step_loss}, train_l2_full: {train_full_loss}, time_spend: {time_spend:.4f}s")
        # always save the latest epoch as checkpoint
        saved_path = os.path.join(saved_dir, saved_model_name)
        model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        torch.save({"epoch": epoch + 1, "loss": min_val_loss,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict()
                }, saved_path + "-latest.pt")
        # validate every args["save_period"] epochs
        if (epoch + 1) % args["save_period"] == 0:
            print("====================validate====================")
            val_full_loss = val_loop(val_loader, model, loss_fn, device, t_train, initial_step)
            if args["tensorboard"]:
                writer.add_scalar('Full Loss/val', val_full_loss, epoch)
            print(f"[Epoch {epoch}] val_l2_full: {val_full_loss}")
            print("================================================")
            if val_full_loss < min_val_loss:
                min_val_loss = val_full_loss
                # save checkpoint if best
                torch.save({"epoch": epoch + 1, "loss": min_val_loss,
                        "model_state_dict": model_state_dict,
                        "optimizer_state_dict": optimizer.state_dict()
                    }, saved_path + "-best.pt")
    print(f"Total train time: {total_time:.4f}s. Train one epoch every {total_time / (args['epochs'] - start_epoch):.4f}s on average.")
    print("Done.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    cmd_args = parser.parse_args()
    # read config file
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    print(args)
    main(args)
