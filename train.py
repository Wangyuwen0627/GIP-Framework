import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import argparse
import os
import random
import shutil
import time
import cross_val
import encoders
import gen.feat as featgen
import load_data
import util
from graph_sampler import GraphSampler


def evaluate(dataset, model, args, max_num_examples=None):
    model.eval()
    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = data['adj'].float().cuda()
        h0 = data['feats'].float().cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = data['assign_feats'].float().cuda()
        with torch.no_grad():
            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())
        if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
                break
    labels = np.hstack(labels)
    preds = np.hstack(preds)
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print("test result: ", result['acc'])
    return result

def validate(dataset, model, args, max_num_examples=None):
    model.eval()
    labels = []
    preds = []
    avg_loss = 0.0
    for batch_idx, data in enumerate(dataset):
        adj = data['adj'].float().cuda()
        h0 = data['feats'].float().cuda()
        label = data['label'].long().cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = data['assign_feats'].float().cuda()
        with torch.no_grad():
            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            loss = model.loss(ypred, label, adj, batch_num_nodes)
            avg_loss += loss
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())
        if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
                break
    avg_loss /= batch_idx + 1
    labels = np.hstack(labels)
    preds = np.hstack(preds)
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro"),
              'loss': avg_loss}
    return result

def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio*100))
        if args.linkpred:
            name += '_lp'
    else:
        name += '_l' + str(args.num_gc_layers)
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name

def train(dataset, model, args, fold, same_feat=True, val_dataset=None, test_dataset=None, writer=None, mask_nodes = True):
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr)
    iter = 0
    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    best_test_result = {
            'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    val_accs = []
    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        model.train()
        for batch_idx, data in enumerate(dataset):
            begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()
            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', total_time)
        result = validate(dataset, model, args, max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = validate(val_dataset, model, args)
            val_accs.append(val_result['acc'])
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args)
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = best_val_result['loss']
            torch.save(model, 'best_model_{}.pkl'.format(fold))
        if test_result['acc'] > best_test_result['acc'] - 1e-7:
            best_test_result['acc'] = test_result['acc']
        if writer is not None:
            writer.add_scalar('acc/train_acc', result['acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
        print('Best val result: ', best_val_result)
        print('Best test result: ', best_test_result)
        best_val_epochs.append(epoch)
        best_val_accs.append(val_result['acc'])
    return model, val_accs, best_test_result['acc']

def split_test(graphs, args):
    random.shuffle(graphs)
    # get graphs for training
    train_idx = int(len(graphs) * (1-args.test_ratio))
    train_graphs = graphs[:train_idx]
    # get graphs for testing
    test_graphs = graphs[train_idx:]
    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=args.max_nodes, features=args.feature_type)
    test_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers)
    return train_graphs, test_dataset_loader

def benchmark_task(args, writer=None, feat='node-feat'):
    all_vals = []
    all_test = []
    # load graph data with labels
    graphs, labels = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    # split test data
    graphs, test_graphs = split_test(graphs, args)
    example_node = util.node_dict(graphs[0])[0]
    # define type of node features
    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:
        for G in graphs:
            for u in G.nodes():
                util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    else:
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
    # K-fold cross-validation
    print("batch_size:", args.batch_size)
    for i in range(1, 10):
        # split train dataset and valid dataset
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        # load model
        model = encoders.GIPMatching(
                max_num_nodes,
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, args.alpha_1, args.alpha_2, args.alpha_3, args.alpha_4, args.beta_1, args.beta_2, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                assign_input_dim=assign_input_dim, max_step=args.ker_max_step, hidden_graphs=args.ker_hidden_graphs, size_hidden_graphs=args.ker_size_hidden_graphs, ker_hidden_dim=args.ker_hidden_dim, ker_normalize=args.ker_normalize).cuda()
        # train
        _, val_accs, best_test = train(train_dataset, model, args, i, val_dataset=val_dataset, test_dataset=test_graphs, writer=writer)
        all_vals.append(np.max(val_accs))
        all_test.append(best_test)
    print("best_val: ", np.max(all_vals))
    print("best_test: ", np.max(all_test))
    # # get best model on the valid dataset
    model = torch.load('best_model_{}.pkl'.format(np.argmax(all_vals)))
    # evaluate model performance on the test dataset (reported in the paper)
    evaluate(test_graphs, model, args)
    
def arg_parse():
    # parameter settings
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', help='Input dataset.')
    io_parser.add_argument('--pkl', dest='pkl_fname', help='Name of the pkl data file')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname', help='Name of the benchmark dataset')
    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')

    # parameters about prototype matching
    ker_parser = parser.add_argument_group()
    ker_parser.add_argument('--use-node-labels', action='store_true', default=False, help='Whether to use node labels')
    ker_parser.add_argument('--use-node-feats', action='store_true', default=True, help='Whether to use node attributes')
    ker_parser.add_argument('--ker-dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    ker_parser.add_argument('--ker-hidden-graphs', type=int, default=18, metavar='N', help='Number of hidden graphs')
    ker_parser.add_argument('--ker-size-hidden-graphs', type=int, default=10, metavar='N', help='Number of nodes of each hidden graph')
    ker_parser.add_argument('--ker-hidden-dim', type=int, default=3, metavar='N', help='Size of hidden layer of NN')
    ker_parser.add_argument('--ker-penultimate-dim', type=int, default=24, metavar='N', help='Size of penultimate layer of NN')
    ker_parser.add_argument('--ker-max-step', type=int, default=3, metavar='N', help='Max length of walks')
    ker_parser.add_argument('--ker-normalize', action='store_true', default=False, help='Whether to normalize the kernel values')

    # parameters about graph compression
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
            const=True, default=False,
            help='Whether link prediction side objective is used')
    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda', help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
            const=False, default=False,
            help='Whether disable log graph')

    parser.add_argument('--method', dest='method', help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix', help='suffix added to the output filename')
    parser.add_argument('--alpha-1', dest='alpha_1', help='the ratio of clustering loss')
    parser.add_argument('--alpha-2', dest='alpha_2', help='the ratio of balanced loss')
    parser.add_argument('--alpha-3', dest='alpha_3', help='the ratio of multi-similarity loss')
    parser.add_argument('--alpha-4', dest='alpha_4', help='the ratio of diversity loss')
    parser.add_argument('--beta-1', dest='beta_1', help='the ratio of clustering assignment module')
    parser.add_argument('--beta-2', dest='beta_2', help='the ratio of common pattern matching module')


    # default settings
    parser.set_defaults(datadir='data',
                        logdir='log',
                        dataset='syn1v2',
                        max_nodes=1000,
                        cuda='2',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=12,
                        num_epochs=500,
                        train_ratio=0.8,
                        test_ratio=0.15,
                        num_workers=1,
                        input_dim=48,
                        hidden_dim=24,
                        ker_hidden_dim=24,
                        output_dim=8,
                        num_classes=6,
                        num_gc_layers=3,
                        dropout=0.05,
                        alpha_1=0.6,
                        alpha_2=0.5,
                        alpha_3=0.4,
                        alpha_4=0.5,
                        beta_1=0.2,
                        beta_2=0.1,
                        method='soft-assign',
                        name_suffix='',
                        assign_ratio=0.05,
                        num_pool=1
                       )
    return parser.parse_args()

def main():
    prog_args = arg_parse()
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    if os.path.isdir(path):
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    torch.cuda.set_device(2)
    benchmark_task(prog_args, writer=writer, feat='node-feat')
    writer.close()

if __name__ == "__main__":
    main()

