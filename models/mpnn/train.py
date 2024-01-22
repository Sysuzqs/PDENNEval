# coding=utf-8
import argparse
import numpy as np
import os
import random
import torch
import yaml
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *
from mpnn import MPNN

import metrics

def get_dataset(args):
    dataset_args = args["dataset"]
    if dataset_args["single_file"]:
        print("SingleDataset")
        train_data = MPNNDatasetSingle(dataset_args["file_name"],
                                      dataset_args["saved_folder"],
                                      reduced_resolution=dataset_args["reduced_resolution"],
                                      reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                      reduced_batch=dataset_args["reduced_batch"],
                                      test_ratio=dataset_args["test_ratio"],
                                      if_test=False,
                                      variables=args["variables"])
        val_data = MPNNDatasetSingle(dataset_args["file_name"],
                                    dataset_args["saved_folder"],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    test_ratio=dataset_args["test_ratio"],
                                    if_test=True,
                                    variables=args["variables"])
    else:
        print("MultDataset")
        train_data = MPNNDatasetMult(dataset_args["file_name"],
                                     dataset_args["saved_folder"],
                                     reduced_resolution=dataset_args["reduced_resolution"],
                                     reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                     reduced_batch=dataset_args["reduced_batch"],
                                     test_ratio=dataset_args["test_ratio"],
                                     if_test=False,
                                     variables=args["variables"])
        val_data = MPNNDatasetMult(dataset_args["file_name"],
                                  dataset_args["saved_folder"],
                                  reduced_resolution=dataset_args["reduced_resolution"],
                                  reduced_resolution_t=dataset_args["reduced_resolution_t"],
                                  reduced_batch=dataset_args["reduced_batch"],
                                  test_ratio=dataset_args["test_ratio"],
                                  if_test=True,
                                  variables=args["variables"])
    return train_data, val_data


def get_dataloader(train_data, val_data, args):
    dataloader_args = args["dataloader"]
    train_loader = DataLoader(train_data, shuffle=True, **dataloader_args)
    if args["if_training"]:
        val_loader = DataLoader(val_data, shuffle=False, **dataloader_args)
    else:
        val_loader = DataLoader(val_data, shuffle=False, drop_last=True, **dataloader_args)
    return train_loader, val_loader


def get_model(args, pde: PDE):
    model = nn.ModuleList([MPNN(pde, time_window=args["time_window"], hidden_features=args["model"]["hidden_features"],
                hidden_layers=args["model"]["hidden_layer"], eq_variables=args["variables"]) for _ in range(args["num_outputs"])])
    return model


def train_loop(dataloader, model, loss_fn, optimizer, device, graph_creator, unrolling):
    model.train()
    losses = []
    # train one epoch
    start_time = time.time()
    for u, x, variables in dataloader:
        batch_size = u.shape[0]
        num_outputs = u.shape[-1]
        
        # Randomly choose number of unrollings
        unrolled_graphs = random.choice(unrolling)
        assert (graph_creator.nt - graph_creator.tw - (graph_creator.tw * unrolled_graphs) > graph_creator.tw)
        steps = [t for t in range(graph_creator.tw, graph_creator.nt - graph_creator.tw - (graph_creator.tw * unrolled_graphs) + 1)]
        
        # Randomly choose starting (time) point at the PDE solution manifold
        random_steps = random.choices(steps, k=batch_size)
        data, labels = graph_creator.create_data(u, random_steps)
        graph = graph_creator.create_graph(data, labels, x, variables, random_steps).to(device)

        # Unrolling of the equation which serves as input at the current step
        # This is the pushforward trick!!!
        with torch.no_grad():
            for _ in range(unrolled_graphs):
                pred = torch.empty_like(graph.y)
                for i in range(num_outputs):
                    pred[..., i] = model[i](graph, i)
                # next iter
                random_steps = [rs + graph_creator.tw for rs in random_steps]
                _, labels = graph_creator.create_data(u, random_steps)
                graph = graph_creator.create_next_graph(graph, pred, labels, random_steps).to(device)
        
        # compute loss
        pred = torch.empty_like(graph.y)
        for i in range(num_outputs):
            pred[..., i] = model[i](graph, i)
        loss = loss_fn(pred, graph.y)
        # loss = torch.sqrt(loss)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # losses.append(loss.detach() / batch_size)
        losses.append(loss.item())

    end_time = time.time()
    time_spend = end_time - start_time

    return np.mean(losses), time_spend


def val_loop(dataloader, model, loss_fn, device, graph_creator):
    model.eval()
    losses = []
    for u, x, variables in dataloader:
        batch_size = u.shape[0]
        num_outputs = u.shape[-1]
        target, pred= torch.Tensor().to(device), torch.Tensor().to(device)
        steps = [graph_creator.tw] * batch_size # same steps
        data, labels = graph_creator.create_data(u, steps)
        graph = graph_creator.create_graph(data, labels, x, variables, steps).to(device)
        target = torch.cat((target, graph.y), dim=-2)
        with torch.no_grad():
            output = torch.empty_like(graph.y)
            for i in range(num_outputs):
                output[..., i] = model[i](graph, i)
        pred = torch.cat((pred, output), dim=-2)

        for step in range(2*graph_creator.tw, graph_creator.nt - graph_creator.tw + 1, graph_creator.tw):
            steps = [step] * batch_size
            _, labels = graph_creator.create_data(u, steps)
            graph = graph_creator.create_next_graph(graph, output, labels, steps).to(device)
            target = torch.cat((target, graph.y), dim=-2)
            with torch.no_grad():
                output = torch.empty_like(graph.y)
                for i in range(num_outputs):
                    output[..., i] = model[i](graph, i)
            pred = torch.cat((pred, output), dim=-2)

        losses.append(loss_fn(pred, target).item())
        
    return np.mean(losses)


# @timeit(10)
def test_loop(dataloader, model, device, graph_creator, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError']):
    model.eval()
    res_dict = {}
    for name in metric_names:
        res_dict[name] = []
    # test
    for u, x, variables in dataloader:
        batch_size = u.shape[0]
        num_outputs = u.shape[-1]
        target, pred= torch.Tensor().to(device), torch.Tensor().to(device)
        steps = [graph_creator.tw] * batch_size # same steps
        data, labels = graph_creator.create_data(u, steps)
        graph = graph_creator.create_graph(data, labels, x, variables, steps).to(device)
        target = torch.cat((target, graph.y), dim=-2)
        with torch.no_grad():
            output = torch.empty_like(graph.y)
            for i in range(num_outputs):
                output[..., i] = model[i](graph, i)
        pred = torch.cat((pred, output), dim=-2)

        for step in range(2*graph_creator.tw, graph_creator.nt - graph_creator.tw + 1, graph_creator.tw):
            steps = [step] * batch_size
            _, labels = graph_creator.create_data(u, steps)
            graph = graph_creator.create_next_graph(graph, output, labels, steps).to(device)
            target = torch.cat((target, graph.y), dim=-2)
            with torch.no_grad():
                output = torch.empty_like(graph.y)
                for i in range(num_outputs):
                    output[..., i] = model[i](graph, i)
            pred = torch.cat((pred, output), dim=-2)

        pred = to_PDEBench_format(pred, batch_size, graph_creator.pde)
        target = to_PDEBench_format(target, batch_size, graph_creator.pde)
   
        for name in metric_names:
            metric_fn = getattr(metrics, name)
            res_dict[name].append(metric_fn(pred, target))
            
    # post process
    for name in metric_names:
        res_list = res_dict[name]
        if name == "MaxError":
            res = torch.stack(res_list, dim=0)
            res, _ = torch.max(res, dim=0)
        else:
            res = torch.cat(res_list, dim=0) # (num_samples, v)
            res = torch.mean(res, dim=0) # (v)
        res_dict[name] = res

    return res_dict



def main(args):
    # init
    setup_seed(args["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args["model_path"]) if not args["if_training"] or args["continue_training"] else None
    saved_model_name = (args["model_name"] + 
                        f"_{args['optimizer']['name']}" + 
                        f"_lr{args['optimizer']['lr']}" + 
                        f"_bs{args['dataloader']['batch_size']}" +
                        f"_wd{args['optimizer']['weight_decay']}")
    saved_dir = os.path.join(args["output_dir"], os.path.splitext(args["dataset"]["file_name"])[0])

    if args["if_training"]:
        if not os.path.exists(saved_dir): # prepare directory
            os.makedirs(saved_dir)
        if args["tensorboard"]: # visualize
            log_path = os.path.join(args["log_dir"], os.path.splitext(args["dataset"]["file_name"])[0], saved_model_name)
            writer = SummaryWriter(log_path)

    # PDE
    pde = PDE(args["pde_name"],
              variables=args["variables"],
              temporal_domain=eval(args["temporal_domain"]), 
              resolution_t=args["resolution_t"],
              spatial_domain=eval(args["spatial_domain"]),
              resolution=args["resolution"],
              reduced_resolution_t=args["dataset"]["reduced_resolution_t"], 
              reduced_resolution=args["dataset"]["reduced_resolution"]
              )

    # dataset and dataloader
    train_data, val_data = get_dataset(args)
    train_loader, val_loader = get_dataloader(train_data, val_data, args)

    # graph creator
    graph_creator = GraphCreator(pde, neighbors=args["neighbors"], time_window=args["time_window"]).to(device)
    
    # model
    model = get_model(args, pde)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of model parameter to train:", total_params)
    # if test, load model from checkpoint
    if not args["if_training"]:
        print(f"Test mode, load checkpoint from {args['model_path']}")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Best epoch: {checkpoint['epoch']}")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        print("start testing...")
        # coding...
        res = test_loop(val_loader, model, device, graph_creator)
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
    # loss_fn = nn.MSELoss(reduction="sum")
    loss_fn = nn.MSELoss(reduction="mean") # modified

    # train
    print(f"Start training from epoch {start_epoch}")
    total_time = 0
    for epoch in range(start_epoch, args["epochs"]):
        epoch_time = 0
        losses = []
        # max_unrolling = epoch if epoch <= args["unrolling"] else args["unrolling"]
        max_unrolling = args["unrolling"]
        unrolling = [r for r in range(max_unrolling + 1)]
        # for _ in range(graph_creator.nt):
        for _ in range(args["unroll_step"] // args["time_window"]):
            loss, time_spend = train_loop(train_loader, model, loss_fn, optimizer, device, graph_creator, unrolling)
            losses.append(loss)
            epoch_time += time_spend
        print(f"[Epoch {epoch}] MSELoss: {np.mean(losses)}, time: {epoch_time:.3f}s")
        if args["tensorboard"]:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar('Loss/train', np.mean(losses), epoch)
        total_time += epoch_time
        scheduler.step()
        # save checkpoint
        saved_path = os.path.join(saved_dir, saved_model_name)
        model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        torch.save({"epoch": epoch + 1, "loss": min_val_loss,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict()
            }, saved_path + "-latest.pt")
        # validate and save model
        if (epoch + 1) % args["save_period"] == 0:
            print("====================validate====================")
            val_loss = val_loop(val_loader, model, loss_fn, device, graph_creator)
            print(f"[Epoch {epoch}] val_loss: {val_loss}")
            if args["tensorboard"]:
                writer.add_scalar("Loss/val", val_loss, epoch)
            print("================================================")
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                # save checkpoint if best
                torch.save({"epoch": epoch + 1, "loss": min_val_loss,
                        "model_state_dict": model_state_dict,
                        "optimizer_state_dict": optimizer.state_dict()
                    }, saved_path + "-best.pt")
    print("Done.")
    print(f"Total train time: {total_time}s. Train one epoch every {total_time / (args['epochs'] - start_epoch)}s on average.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    print(args)
    
    main(args)