import math
import os
import sys
import shutil
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import time
import matplotlib.pyplot as plt
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


train_example_command='python3 graph_conv_main.py -m train --epochs 10 ' \
                      '--batch_size 8 --save_dir saves/graph_conv_std ../datasets/ged_dataset.h5'
eval_example_command='python3 graph_conv_main.py -m eval --which_set val --epochs 10 --batch_size 8 --checkpoint saves/graph_conv_std --plot_predictions True ../datasets/ged_dataset.h5'

parser = argparse.ArgumentParser(description='Graph Convolution Network for GED prediction Training.\n\n'
                                             'Example Command Training:\n' + train_example_command +
                                             '\nExample Command Evaluation:\n' + eval_example_command,
                                 formatter_class=RawTextHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='Path to the dataset')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='Number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=8, type=int,
                    metavar='N', help='Mini_batch size')
parser.add_argument('--lr', '--learning_rate', dest='learning_rate', default=0.0005, type=float,
                    metavar='LR', help='Initial learning rate')
parser.add_argument('--embedding_size', default=10, type=int,
                    metavar='N', help='Size of the embedding to use.')
parser.add_argument('--checkpoint', '--cp', default='', type=str, metavar='PATH',
                    help='Path to the checkpoint to use(default: none)')
parser.add_argument('--save_dir', metavar='DIR', type=str, default='',
                    help='Path to directory in which to store the model. Defaults to not saving the model')
parser.add_argument('-m', '--mode', dest='mode', default='train', type=str, choices=['train', 'eval'],
                    help='Whether to: train the network ("train") or evaluate it ("eval")')
parser.add_argument('-s', '--which_set', dest='which_set', default='train', type=str, choices=['train', 'val', 'test'],
                    help='Whether to: train ("train"), evaluate on validation set ("val") or test set ("test")')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--post_training_eval', default=True, type=bool, metavar='BOOL',
                    help='Whether to run an evaluation on the validation set after training is complete.')
parser.add_argument('--plot_predictions', dest='plot_predictions', default=False, action='store_true',
                    help='Whether to make the prediction plot. The data must have been evaluated for that purpose.')


def main():
    """Interface for training and evaluating using the command line"""
    global args
    args = parser.parse_args()

    model = SiameseNetwork(1, args.embedding_size)

    # If a checkpoint provided, load it's values
    if args.checkpoint:
        state = torch.load(args.checkpoint)
        model.load_state_dict(state['state_dict'])
    else:
        state = None

    # Run the model on a GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Train the network
    if args.mode == 'train':
        dataset = GEDDataset(args.data, which_set='train', adj_dtype=np.float32, transform=None)
        train(model, dataset, batch_size=args.batch_size, embed_size=args.embedding_size, num_epochs=args.epochs,
              learning_rate=args.learning_rate, save_to=args.save_dir, resume_state=args.checkpoint)

    # Whether to store the predictions from eval for plotting
    store_res = args.plot_predicitons

    if args.mode == 'eval':
        dataset = GEDDataset(path_to_dataset, which_set=args.which_set, adj_dtype=np.float32, transform=None)
        *_, results = eval(model, dataset, batch_size=args.batch_size, store_results=store_res)
    if args.post_training_eval:
        dataset = GEDDataset(path_to_dataset, which_set='val', adj_dtype=np.float32, transform=None)
        *_, results = eval(model, dataset, batch_size=args.batch_size, store_results=store_res)

    # Finally, if plotting the results:
    if args.plot_predicitons:
        # Assert that the data has been evaluated
        if not (args.mode == 'eval' or args.post_training_eval):
            raise AttributeError('The flags provided did not specify to evaluate the dataset, which is required for'
                                 'plotting')




def train(model, dataset, batch_size=8, embed_size=10, num_epochs=10, learning_rate=0.1, save_to=None,
          aggregate_log_dir='./log', resume_state=None):
    """Function for training a model"""
    print(f"\t>  Running the evaluation on the {dataset.which_set} dataset.")
    model.train()

    # Initiate the TensorBoard logging pipeline
    log_dir = initialise_log_dir(aggregate_log_dir)
    writer = SummaryWriter(log_dir)

    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    if resume_state:
        optimiser.load_state_dict(resume_state['optimiser'])
        init_epoch = resume_state['epoch']
    else:
        init_epoch = 0

    running_loss = 0.0
    running_l1 = 0.0
    log_every = 100

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    i = 0
    for epoch in range(0, num_epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            graph1_batch, graph2_batch, label_batch = sample_batched['graph1'],\
                                                      sample_batched['graph2'],\
                                                      sample_batched['label']
            # graph1_batch, graph2_batch, label_batch = map(lambda x: x.to(device),
            #                                               [graph1_batch, graph2_batch, label_batch])

            # Compute the embeddings
            output1, output2 = model(graph1_batch, graph2_batch)
            # Compute the distance between the embeddings
            distance = pairwise_distance(output1, output2)

            optimiser.zero_grad()
            loss = criterion(distance, label_batch)
            loss.backward()
            optimiser.step()

            # Accuracy metrics
            mean_abs_error = torch.mean(torch.abs(distance - label_batch))

            # Log metrics to TensorBoard
            writer.add_scalar('train/loss', loss, i)
            writer.add_scalar('train/MAE', mean_abs_error, i)

            running_loss += loss.item()
            running_l1 += mean_abs_error
            if i % log_every == log_every - 1:
                print(f"Current epoch: {epoch} / {num_epochs} | Batch iteration: {i_batch + 1} | "
                      f"\n Loss: {running_loss / log_every:8f} | Mean Absolute Error: {running_l1 / log_every:8f}\n")
                running_loss = 0.0
                running_l1 = 0.0
            i += 1

    writer.close()

    # Save the model checkpoint
    if save_to:
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimiser': optimiser.state_dict(),
        }
        save_checkpoint()

    return model, optimiser, epoch


def eval(model, dataset, batch_size=8, store_results=False):
    print(f"\t>  Running the evaluation on the {dataset.which_set} dataset.")
    model.eval()

    # Accuracy criteria:
    l1_loss_op = nn.L1Loss()
    l2_loss_op = nn.MSELoss()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    total_l1 = 0.
    total_l2 = 0.

    num_batches = 0

    if store_results:
        # Store the predictions and labels in an array
        results = np.empty([len(dataset), 2], dtype=np.float32)

    start_time = time.time()
    for i_batch, sample_batched in enumerate(dataloader):
        graph1_batch, graph2_batch, label_batch = sample_batched['graph1'], \
                                                  sample_batched['graph2'], \
                                                  sample_batched['label']
        graph1_batch, graph2_batch, label_batch = map(lambda x: x.to(device),
                                                      [graph1_batch, graph2_batch, label_batch])

        # Compute the embeddings
        output1, output2 = model(graph1_batch, graph2_batch)
        # Compute the distance between the embeddings
        distance = pairwise_distance(output1, output2)

        l1 = l1_loss_op(distance, label_batch)
        l2 = l2_loss_op(distance, label_batch)
        total_l1 += l1
        total_l2 += l2
        num_batches += 1

        if store_results:
            idx = i_batch * batch_size
            results[idx:idx + batch_size, 0] = distance
            results[idx:idx + batch_size, 1] = label_batch

    avg_l1 = total_l1 / num_batches
    avg_l2 = total_l2 / num_batches

    print(f"Evaluation results:\n\tMean Square Error: {avg_l1:8f} | Mean Absolute Error: {avg_l2:8f}"
          f"\n\tTime taken: {time.time() - start_time:4f}s for {len(dataset)} examples")
    if store_results:
        return avg_l1, avg_l2, results
    else:
        return avg_l1, avg_l2, None


def initialise_log_dir(aggregate_log_dir='./log', model_name=''):
    """Initialises a new logging directory for the run and returns its path"""
    # Create the logging directory if it doesn't exist
    if not os.path.exists(aggregate_log_dir):
        os.makedirs(aggregate_log_dir)

    log_dir = get_next_available_name(aggregate_log_dir, startswith='run_')
    return os.path.join(aggregate_log_dir, log_dir)


def get_next_available_name(dir_path, startswith):
    """
    Get the next available path that begins with the phrase of choice and ends with an incrementing number
    :param dir_path: path to directory where the files are located
    :param startswith: the phrase that the filenames start with
    :return: the next available filename
    """
    dirs = os.listdir(dir_path)
    previous_runs = list(filter(lambda d: d.startswith(startswith), dirs))
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split(startswith)[1]) for s in previous_runs]) + 1
    path = startswith + str(run_number)
    return path


def save_checkpoint(state, save_dir, is_best=False, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pth.tar'))


def accuracy_metrics_classification(distance, label, batch_size):
    num_positive = torch.sum(label)
    num_negative = batch_size - num_positive
    # Average distance between points from the same class
    intraclass_distance = torch.sum(distance * label.squeeze()) / num_positive
    # Average distance between points from different classes
    interclass_distance = torch.sum(distance - distance * label.squeeze()) / num_negative
    return intraclass_distance, interclass_distance


def plot_prediction(results, plots_dir='../plots'):
    predictions = results[0, :]
    labels = results[1, :]

    x_max = np.max(labels) * 1.05
    plt.xlim(xmin=0., xmax=x_max)
    plt.ylim(ymin=0.)

    plt.scatter(predictions, labels, color='#40ddbe', alpha=0.3)
    # Plot the desired results (a straight y = x line)
    x =  np.linspace(0., x_max, 1000)
    plt.plot(x, x, color='#3572b7')

    plt.xlabel("Actual Distance Value")
    plt.ylabel("Predicted Distance Value")

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(get_next_available_name(plots_dir, 'pred_vs_label_'), bbox_inches='tight', dpi=200)
    return


if __name__ == '__main__':
    main()

