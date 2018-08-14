import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from context import *

from torch.nn.functional import pairwise_distance
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from iso_nn_data_util import generate_batch
from iso_nn_data_util import generate_example
from models import SiameseNetwork, ContrastiveLoss
from embedding_models import GCN
from dataset_util import GEDDataset, ToTensor


def initialise_log_dir(aggregate_log_dir='./log'):
    """Initialises a new directory for the run and returns its path"""
    # Create the logging directory if it doesn't exist
    if not os.path.exists(aggregate_log_dir):
        os.makedirs(aggregate_log_dir)

    dirs = os.listdir(aggregate_log_dir)
    previous_runs = list(filter(lambda d: d.startswith('run_'), dirs))
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

    log_dir = f"run_{run_number}"
    return os.path.join(aggregate_log_dir, log_dir)


def accuracy_metrics_classification(distance, label, batch_size):
    num_positive = torch.sum(label)
    num_negative = batch_size - num_positive
    # Average distance between points from the same class
    intraclass_distance = torch.sum(distance * label.squeeze()) / num_positive
    # Average distance between points from different classes
    interclass_distance = torch.sum(distance - distance * label.squeeze()) / num_negative
    return intraclass_distance, interclass_distance


# def accuracy_metrics_regression(distance, label, batch_size):
#     num_positive = torch.sum(label)
#     num_negative = batch_size - num_positive
#     # Average distance between points from the same class
#     intraclass_distance = torch.sum(distance * label.squeeze()) / num_positive
#     # Average distance between points from different classes
#     interclass_distance = torch.sum(distance - distance * label.squeeze()) / num_negative
#     return intraclass_distance, interclass_distance


def train(path_to_dataset, batch_size=8, embed_size=10, num_epochs=10, learning_rate=0.1, aggregate_log_dir='./log'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initiate the TensorBoard logging pipeline
    log_dir = initialise_log_dir(aggregate_log_dir)
    writer = SummaryWriter(log_dir)

    net = SiameseNetwork(1, embed_size)
    net.to(device)
    net.train()

    criterion = nn.MSELoss()
    optimiser = optim.Adam(net.parameters(), lr=learning_rate)

    counter = []
    loss_history = []
    running_loss = 0.0
    log_every = 100

    dataset = GEDDataset(path_to_dataset, which_set='train', adj_dtype=np.float32, transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    i = 0
    for epoch in range(0, num_epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            graph1_batch, graph2_batch, label_batch = sample_batched['graph1'],\
                                                      sample_batched['graph2'],\
                                                      sample_batched['label']
            # graph1_batch, graph2_batch, label_batch = torch.from_numpy(graph1_batch),\
            #                                           torch.from_numpy(graph2_batch),\
            #                                           torch.from_numpy(label_batch)
            graph1_batch, graph2_batch, label_batch = map(lambda x: x.to(device),
                                                          [graph1_batch, graph2_batch, label_batch])

            # Compute the embeddings
            output1, output2 = net(graph1_batch, graph2_batch)
            # Compute the distance between the embeddings
            distance = pairwise_distance(output1, output2)

            optimiser.zero_grad()
            loss = criterion(distance, label_batch)
            loss.backward()
            optimiser.step()

            # Evaluation metrics
            # intraclass_distance, interclass_distance = accuracy_metrics(distance, label, batch_size)
            # Log metrics to TensorBoard
            writer.add_scalar('train/loss', loss, i)
            # writer.add_scalar('train/intraclass_distance', intraclass_distance, i)
            # writer.add_scalar('train/interclass_distance', interclass_distance, i)

            running_loss += loss.item()
            if i % log_every == log_every - 1:
                print(f"Current epoch: {epoch} / {num_epochs} | Batch: {i_batch} | "
                      f"\n Loss {running_loss / log_every}\n")
                counter.append(i)
                loss_history.append(running_loss / log_every)
                running_loss = 0.0
            i += 1

    writer.close()
    # show_plot(counter, loss_history)

    return net, counter, loss_history


def eval(net, batch_size=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
    net.eval()

    criterion = nn.MSELoss()

    dataset = GEDDataset(path_to_dataset, which_set='val', adj_dtype=np.float32, transform=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    total_loss = 0.0

    num_batches = 0
    for i_batch, sample_batched in enumerate(dataloader):
        graph1_batch, graph2_batch, label_batch = sample_batched['graph1'], \
                                                  sample_batched['graph2'], \
                                                  sample_batched['label']
        graph1_batch, graph2_batch, label_batch = map(lambda x: x.to(device),
                                                      [graph1_batch, graph2_batch, label_batch])

        # Compute the embeddings
        output1, output2 = net(graph1_batch, graph2_batch)
        # Compute the distance between the embeddings
        distance = pairwise_distance(output1, output2)

        loss = criterion(distance, label_batch)
        total_loss += loss
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


if __name__ == '__main__':
    path_to_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/ged_dataset.h5'))

    # Hyperparameters
    batch_size = 16
    embed_size = 10
    num_epochs = 10
    learning_rate = 0.0001

    net, _ = train(path_to_dataset, batch_size, embed_size, num_epochs, learning_rate)

    avg_loss = eval(net, batch_size=1)
