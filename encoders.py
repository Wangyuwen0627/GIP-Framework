import copy
import torch
import torch.nn as nn
from torch.nn import init, Parameter
import torch.nn.functional as F
import numpy as np
from mlp import MLP
from set2set import Set2Set
from grakel import Graph
from grakel.kernels import RandomWalk
import math
EPS = 1e-15

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers, pred_hidden_dims=[], concat=True,
                 bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1
        self.bias = True
        if args is not None:
            self.bias = args.bias
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(input_dim, hidden_dim, embedding_dim,
                                                                                  num_layers, add_self, normalize=True,
                                                                                  dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim
        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, label_dim,
                                                 num_aggs=self.num_aggs)
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self, normalize=False,
                          dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''
        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)

        return ypred

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        # return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())

def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out

class GcnSet2SetEncoder(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnSet2SetEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                num_layers, pred_hidden_dims, concat, bn, dropout, args=args)
        self.s2s = Set2Set(self.pred_input_dim, self.pred_input_dim * 2)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out = self.s2s(embedding_tensor)
        # out, _ = torch.max(embedding_tensor, dim=1)
        ypred = self.pred_model(out)
        return ypred

class GIPMatching(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            assign_hidden_dim, alpha_1, alpha_2, alpha_3, alpha_4, beta_1, beta_2, assign_ratio=0.25, assign_num_layers=-1, num_pooling=2,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
            assign_input_dim=-1, args=None, max_step=2, hidden_graphs=6, size_hidden_graphs=6, ker_hidden_dim=3, ker_normalize=False):
        '''
        Args:
            qq1q: number of gc layers before each pooling,
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''
        # embedding_dim=output_dim, label_dim=num_classes
        super(GIPMatching, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(self.pred_input_dim, hidden_dim, embedding_dim, num_layers, add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(input_dim * 2, hidden_dim))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)
            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_prototypes_per_class = hidden_graphs // label_dim
        self.max_step = max_step
        self.label_dim = label_dim
        self.hidden_graphs = hidden_graphs
        self.hidden_dim = hidden_dim
        self.size_hidden_graphs = size_hidden_graphs
        self.normalize = ker_normalize
        self.adj_hidden = Parameter(torch.FloatTensor(hidden_graphs, (size_hidden_graphs * (size_hidden_graphs - 1)) // 2))
        self.features_hidden = Parameter(torch.FloatTensor(hidden_graphs, size_hidden_graphs, ker_hidden_dim))
        self.fc = nn.Linear(hidden_dim * (num_layers - 1) + embedding_dim, ker_hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_graphs)
        self.fc2 = nn.Linear(hidden_graphs, label_dim)
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_3 = alpha_3
        self.alpha_4 = alpha_4
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.num_of_classes = label_dim
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.prototype_class_identity = torch.zeros(hidden_graphs, label_dim)
        for j in range(hidden_graphs):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1
        # initialize the last layer
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.15)
        self.init_weights()

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations
        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.fc2.weight.data.copy_(correct_class_connection * positive_one_weights_locations+ incorrect_class_connection * negative_one_weights_locations)

    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None
        out_all = []
        embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)
        self.previous_x = []
        self.previous_adj = []
        self.previous_label = []
        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None
            for idx, num_nodes in enumerate(batch_num_nodes):
                self.previous_x.append(x[idx, :num_nodes])
                self.previous_adj.append(adj[idx, :num_nodes, :num_nodes])

            self.assign_tensor = self.gcn_forward(x_a, adj, self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i], embedding_mask)
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            self.max_index = torch.argmax(self.assign_tensor, dim=-1)
            unique_max_indices, sort_indices = torch.unique(self.max_index, sorted=True, return_inverse=True)
            new_indices = torch.arange(len(unique_max_indices))
            self.node_label = new_indices[sort_indices]
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor

        embedding_tensor = self.gcn_forward(x, adj, self.conv_first_after_pool[i], self.conv_block_after_pool[i], self.conv_last_after_pool[i])
        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)
        self.x_cluster = x
        self.adj_cluster = adj
        n_graphs = self.x_cluster.shape[0]
        counts = [self.x_cluster[i].shape[0] for i in range(n_graphs)]

        if self.normalize:
            norm = counts.unsqueeze(1).repeat(1, self.hidden_graphs)
        adj_hidden_norm = torch.zeros(self.hidden_graphs, self.size_hidden_graphs, self.size_hidden_graphs).to(self.device)
        idx = torch.triu_indices(self.size_hidden_graphs, self.size_hidden_graphs, 1)
        adj_hidden_norm[:, idx[0], idx[1]] = self.relu(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 1, 2)
        # x为经过压缩的节点特征，【图的个数， 节点个数，特征维度】
        x = self.sigmoid(self.fc(self.x_cluster))
        # z为待训练的原型图的节点特征维度，【原型图个数，原型图中的节点个数，特征维度】
        z = self.features_hidden
        # zx, 【原型图个数，原型图中的节点个数，图的个数， 节点个数】
        zx = torch.einsum("abc,edc->abed", (z, x))
        out = list()

        # random walk
        for i in range(self.max_step):
            if i == 0:
                # eye：【原型图个数，原型图中的节点个数， 原型图中的节点个数】
                eye = torch.eye(self.size_hidden_graphs, device=self.device)
                eye = eye.repeat(self.hidden_graphs, 1, 1)
                # o,【原型图个数，原型图中的节点个数，特征维度】
                o = torch.einsum("abc,acd->abd", (eye, z))
                # t, 【原型图个数，原型图中的节点个数，图的个数， 节点个数】
                t = torch.einsum("abc,edc->abed", (o, x))
            else:
                # adj, 【图的个数，节点个数，节点个数】
                # x, 【图的个数， 节点个数，特征维度】
                x = torch.einsum("abc,abd->abd", (adj, x))
                # adj_hidden_norm: 【原型图个数，原型图中节点个数，原型图中节点个数】
                # z：【原型图个数，原型图中的节点个数，特征维度】
                z = torch.einsum("abc,acd->abd", (adj_hidden_norm, z))
                # t: 【原型图个数，原型图中的节点个数，图的个数， 节点个数】
                t = torch.einsum("abc,edc->abed", (z, x))
            # ([18, 8, 20, 20])
            t = self.dropout(t)
            # ([18, 8, 20, 20])
            t = torch.mul(zx, t)
            # ([18, 8, 20])：【原型图个数，原型图中的节点个数，图的个数】
            t = torch.sum(t, dim=3)
            # ([18, 20])
            t = torch.sum(t, dim=1)
            # ([20, 18]): 每个图和各个原型图之间的相似度
            t = torch.transpose(t, 0, 1)

            if self.normalize:
                t /= norm
            out.append(t)
        self.adj_hidden_norm = adj_hidden_norm
        out = torch.stack(out)
        out = torch.sum(out, dim=0)
        out = self.bn(out)
        # logits
        out = self.fc2(out)
        self.pairwise_similarity = t
        self.prototype_features = z
        self.compressed_features = x
        self.prototype_adj = adj_hidden_norm
        self.compressed_adj = adj
        return F.log_softmax(out, dim=1)

    def cal_diversity_loss(self):
        diversity_loss = 0.0
        groups = []
        for i in range(self.num_of_classes):
            groups.append([num for num in range(i * self.num_prototypes_per_class, (i+1) * self.num_prototypes_per_class)])
        for group in groups:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    sim = self.prototype_similarity[group[i]][group[j]]
                    sim = math.tanh(sim)
                    diversity_loss = diversity_loss + max(0, sim - 2)
        return diversity_loss

    def cal_multi_similarity_loss(self, labels):
        # m个class，分别计算graph与每一个原型图之间的核空间距离（基于随机游走图核计算）
        # 计算每个graph之间的随机游走图核相似度，同时计算每个原型图彼此之间的随机游走图核相似度
        loss = 0.0
        xx = torch.einsum("abc,edc->abed", (self.compressed_features, self.compressed_features))
        counts = [self.x_cluster[i].shape[0] for i in range(self.compressed_features.shape[0])]
        if self.normalize:
            norm = counts.unsqueeze(1).repeat(1, self.compressed_features.shape[0])
        for i in range(self.max_step):
            if i == 0:
                # eye：【图个数，图中的节点个数， 图中的节点个数】
                eye = torch.eye(self.compressed_features.shape[1], device=self.device)
                eye = eye.repeat(self.compressed_features.shape[0], 1, 1)
                # o,【图个数，图中的节点个数，特征维度】
                o = torch.einsum("abc,acd->abd", (eye, self.compressed_features))
                # t, 【图个数，图中的节点个数，图的个数， 节点个数】
                t = torch.einsum("abc,edc->abed", (o, self.compressed_features))
            else:
                # adj, 【图的个数，节点个数，节点个数】
                # x, 【图的个数， 节点个数，特征维度】
                x = torch.einsum("abc,abd->abd", (self.compressed_adj, self.compressed_features))
                # adj_hidden_norm: 【原型图个数，原型图中节点个数，原型图中节点个数】
                # z：【原型图个数，原型图中的节点个数，特征维度】
                z = torch.einsum("abc,acd->abd", (self.compressed_adj, self.compressed_features))
                # t: 【原型图个数，原型图中的节点个数，图的个数， 节点个数】
                t = torch.einsum("abc,edc->abed", (z, x))
            # ([18, 8, 20, 20])
            t = self.dropout(t)
            # ([18, 8, 20, 20])
            t = torch.mul(xx, t)
            # ([18, 8, 20])：【原型图个数，原型图中的节点个数，图的个数】
            t = torch.sum(t, dim=3)
            # ([18, 20])
            t = torch.sum(t, dim=1)
            # ([20, 18]): 每个图和各个原型图之间的相似度
            t = torch.transpose(t, 0, 1)
            self.graph_similarity = t
            if self.normalize:
                self.graph_similarity /= norm

        zz = torch.einsum("abc,edc->abed", (self.prototype_features, self.prototype_features))
        counts = [self.size_hidden_graphs for i in range(self.prototype_features.shape[0])]
        if self.normalize:
            norm = counts.unsqueeze(1).repeat(1, self.prototype_features.shape[0])
        for i in range(self.max_step):
            if i == 0:
                # eye：【图个数，图中的节点个数， 图中的节点个数】
                eye = torch.eye(self.prototype_features.shape[1], device=self.device)
                eye = eye.repeat(self.prototype_features.shape[0], 1, 1)
                # o,【图个数，图中的节点个数，特征维度】
                o = torch.einsum("abc,acd->abd", (eye, self.prototype_features))
                # t, 【图个数，图中的节点个数，图的个数， 节点个数】
                t = torch.einsum("abc,edc->abed", (o, self.prototype_features))
            else:
                # adj, 【图的个数，节点个数，节点个数】
                # x, 【图的个数， 节点个数，特征维度】
                x = torch.einsum("abc,abd->abd", (self.prototype_adj, self.prototype_features))
                # adj_hidden_norm: 【原型图个数，原型图中节点个数，原型图中节点个数】
                # z：【原型图个数，原型图中的节点个数，特征维度】
                z = torch.einsum("abc,acd->abd", (self.prototype_adj, self.prototype_features))
                # t: 【原型图个数，原型图中的节点个数，图的个数， 节点个数】
                t = torch.einsum("abc,edc->abed", (z, x))
            # ([18, 8, 20, 20])
            t = self.dropout(t)
            # ([18, 8, 20, 20])
            t = torch.mul(zz, t)
            # ([18, 8, 20])：【原型图个数，原型图中的节点个数，图的个数】
            t = torch.sum(t, dim=3)
            # ([18, 20])
            t = torch.sum(t, dim=1)
            # ([20, 18]): 每个图和各个原型图之间的相似度
            t = torch.transpose(t, 0, 1)
            self.prototype_similarity = t
            if self.normalize:
                self.prototype_similarity /= norm
        for i in range(self.compressed_features.shape[0]):
            label = labels[i]
            k_graph = self.graph_similarity[i][i]
            single_loss = 0.0
            for j in range(self.num_prototypes_per_class):
                k_prototype = self.prototype_similarity[self.num_prototypes_per_class * label + j][self.num_prototypes_per_class * label + j]
                k_pairwise = self.pairwise_similarity[i][self.num_prototypes_per_class * label + j]
                d_mi = math.sqrt(math.fabs((k_graph+k_prototype)/2-k_pairwise))
                d_mi = math.tanh(d_mi)
                single_loss = single_loss + math.exp(0.5*math.fabs(d_mi-0.3))
            positive_loss = 0.1 * math.log(1+single_loss, 2)
            range1 = range(0, self.num_prototypes_per_class * label)
            range2 = range(self.num_prototypes_per_class * label+1, self.hidden_graphs)
            merged_range = list(range1)
            merged_range.extend(range2)
            single_loss = 0.0
            for j in merged_range:
                k_prototype = self.prototype_similarity[j][j]
                k_pairwise = self.pairwise_similarity[i][j]
                d_mi = math.sqrt(math.fabs((k_graph+k_prototype)/2-k_pairwise))
                d_mi = math.tanh(d_mi)
                single_loss = single_loss + math.exp(-0.5*math.fabs(d_mi-0.3))
            negative_loss = 0.1 * math.log(1+single_loss, 2)
            loss = loss + positive_loss + negative_loss
        loss = loss/self.compressed_features.shape[0]
        return loss

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(GIPMatching, self).loss(pred, label)
        alpha = 0.5
        motif_adj = torch.mul(torch.matmul(adj, adj), adj)
        s = torch.softmax(self.assign_tensor, dim=-1)
        out_adj = self.adj_cluster
        motif_out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), motif_adj), s)
        diag_SAS = torch.einsum("ijj->ij", out_adj.clone())
        d_flat = torch.einsum("ijk->ij", adj.clone())
        d = _rank3_diag(d_flat)
        sds = torch.matmul(torch.matmul(s.transpose(1, 2), d), s)
        diag_SDS = torch.einsum("ijk->ij", sds) + EPS
        mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        (batch_size, num_nodes, _), k = self.x_cluster.size(), s.size(-1)
        mincut_loss = 1 / k * torch.mean(mincut_loss)
        diag_SAS = torch.einsum("ijj->ij", motif_out_adj)
        d_flat = torch.einsum("ijk->ij", motif_adj)
        d = _rank3_diag(d_flat)
        diag_SDS = (torch.einsum("ijk->ij", torch.matmul(torch.matmul(s.transpose(1, 2), d), s)) + EPS)
        ho_mincut_loss = -torch.sum(diag_SAS / diag_SDS, axis=1)
        ho_mincut_loss = 1 / k * torch.mean(ho_mincut_loss)
        clustering_loss = (1 - alpha) * mincut_loss + alpha * ho_mincut_loss
        balanced_loss = sum([((-torch.sum(torch.norm(s[i], p="fro", dim=-2)) / (num_nodes ** 0.5) + k ** 0.5) / (k ** 0.5 - 1)) for i in range(batch_size)]) / float(batch_size)
        multi_similarity_loss = self.cal_multi_similarity_loss(label)
        diversity_loss = self.cal_diversity_loss()
        loss_ca = self.alpha_1 * clustering_loss + self.alpha_2 * balanced_loss
        loss_cpm = self.alpha_3 * multi_similarity_loss + self.alpha_4 * diversity_loss
        loss = loss + self.beta_1 * loss_ca + self.beta_2 * loss_cpm
        return loss