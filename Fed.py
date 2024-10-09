import torch
import numpy as np
from metrics import get_gt_matrix, get_statistics, top_k, compute_precision_k
from utils import get_adj, get_edgeindex
from model.node2vec import node2vec
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import argparse
import os
import networkx as nx

def parse_args():
    parser = argparse.ArgumentParser(description="CCNE")
    parser.add_argument('--gt_path', default='./data/douban/anchor/node,split=0.8.test.dict')
    parser.add_argument('--train_path', default='./data/douban/anchor/node,split=0.8.train.dict')
    parser.add_argument('--out_path', default='./data/douban/anchor/embeddings')
    parser.add_argument('--s_edge', default='./data/douban/online/raw/edgelist')
    parser.add_argument('--t_edge', default='./data/douban/offline/raw/edgelist')
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=3000, type=int)
    parser.add_argument('--rounds', default=5, type=int)
    return parser.parse_args()

class FedUA(torch.nn.Module):
    def __init__(self, s_input, t_input, output):#, not_share=False, is_bn=False, dropout=0):
        super().__init__()
        self.conv1 = GCNConv(s_input, 2 * output)
        self.conv2 = GCNConv(t_input, 2 * output)
        self.conv3 = GCNConv(2 * output, output)
        self.activation = nn.ReLU()

    def s_forward(self, x, edge_index):
        '''
        embeddings of source network g_s(V_s, E_s):
        x is node feature vectors of g_s, with shape [V_s, n_feats], V_s is the number of nodes,
            and n_feats is the dimension of features;
        edge_index is edges, with shape [2, 2 * E_s], E_s is the number of edges
        '''
        x = self.conv1(x, edge_index)
        # if self.is_bn:
        #     x = self.bn1(x)
        x = self.activation(x)
        # if self.dropout > 0:
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)

    def t_forward(self, x, edge_index):
        '''
        embeddings of target network g_t(V_t, E_t):
        x is node feature vectors of g_t, with shape [V_t, n_feats], V_t is the number of nodes,
            and n_feats is the dimension of features;
        edge_index is edges, with shape [2, 2 * E_t], E_t is the number of edges
        '''
        x = self.conv2(x, edge_index)
        # if self.is_bn:
        #     x = self.bn2(x)
        x = self.activation(x)
        # if self.dropout > 0:
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # if self.share:
        #     return self.conv3(x, edge_index)
        # return self.conv4(x, edge_index)
        return self.conv3(x, edge_index)

    def decoder(self, z, edge_index, sigmoid=True):
        '''
        reconstuct the original network by calculating the pairwise similarity of embedding vectors
        '''
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def single_recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        '''
        cross-entropy loss of positive and negative edges:
        z: the output of decoder
        pos_edge_index: index of positive edges
        neg_edge_index: index of negative edges
        '''
        EPS = 1e-15  # avoid zero when calculating logarithm

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean()  # loss of positive samples

        if neg_edge_index is None:
            neg_edge_index = torch_geometric.utils.negative_sampling(pos_edge_index, z.size(0))  # negative sampling
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index) + EPS).mean()  # loss of negative samples

        return pos_loss + neg_loss

    def intra_loss(self, zs, zt, s_pos_edge_index, t_pos_edge_index):
        '''
        intra-network loss to preserve intra-network structural features:
        zs: embeddings of source network g_s;
        zt: embeddings of target network g_t
        '''
        s_recon_loss = self.single_recon_loss(zs, s_pos_edge_index)
        t_recon_loss = self.single_recon_loss(zt, t_pos_edge_index)
        return s_recon_loss + t_recon_loss


def get_embedding(s_x, t_x, s_e, t_e, g_s, g_t, anchor, gt_mat, dim=64, lr=0.001, lamda=1, margin=0.8, neg=1, epochs=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    s_x = s_x.to(device)
    t_x = t_x.to(device)
    s_e = s_e.to(device)
    t_e = t_e.to(device)
    s_input = s_x.shape[1]
    t_input = t_x.shape[1]
    model = FedUA(s_input, t_input, dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cosine_loss=nn.CosineEmbeddingLoss(margin=margin)
    in_a, in_b, anchor_label = sample(anchor, g_s, g_t, neg=neg) # hard negative sampling

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # in_a, in_b, anchor_label = sample(anchor, g_s, g_t, neg=neg)
        zs = model.s_forward(s_x, s_e)
        zt = model.t_forward(t_x, t_e)

        intra_loss = model.intra_loss(zs, zt, s_e, t_e)
        anchor_label = anchor_label.view(-1).to(device)
        inter_loss = cosine_loss(zs[in_a], zt[in_b], anchor_label)
        loss = intra_loss + lamda * inter_loss
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            p10 = evaluate(zs, zt, gt_mat)
            print('Epoch: {:03d}, intra_loss: {:.8f}, inter_loss: {:.8f}, loss_train: {:.8f}, precision_10: {:.8f}'.format(epoch,\
                intra_loss, inter_loss, loss, p10))
    
    model.eval()
    s_embedding = model.s_forward(s_x, s_e)
    t_embedding = model.t_forward(t_x, t_e)
    s_embedding = s_embedding.detach().cpu()
    t_embedding = t_embedding.detach().cpu()
    return s_embedding, t_embedding

@torch.no_grad()
def evaluate(zs, zt, gt):
    '''
    calculate Precision@10 for evaluation
    '''
    z1 = zs.detach().cpu()
    z2 = zt.detach().cpu()
    S = cosine_similarity(z1, z2)
    pred_top_10 = top_k(S, 10)
    precision_10 = compute_precision_k(pred_top_10, gt)
    return precision_10

def sample(anchor_train, gs, gt, neg=1):
    '''
    sample non-anchors for each anchor
    '''
    triplet_neg = neg  # number of non-anchors for each anchor, when neg=1, there are two negtives for each anchor
    anchor_flag = 1
    anchor_train_len = anchor_train.shape[0]
    anchor_train_a_list = np.array(anchor_train.T[0])
    anchor_train_b_list = np.array(anchor_train.T[1])
    input_a = []
    input_b = []
    classifier_target = torch.empty(0)
    np.random.seed(5)
    index = 0
    while index < anchor_train_len:
        a = anchor_train_a_list[index]
        b = anchor_train_b_list[index]
        input_a.append(a)
        input_b.append(b)
        an_target = torch.ones(anchor_flag)
        classifier_target = torch.cat((classifier_target, an_target), dim=0)
        # an_negs_index = list(set(node_t) - {b}) # all nodes except anchor node
        an_negs_index = list(gt.neighbors(b)) # neighbors of each anchor node
        an_negs_index_sampled = list(np.random.choice(an_negs_index, triplet_neg, replace=True)) # randomly sample negatives
        an_as = triplet_neg * [a]
        input_a += an_as
        input_b += an_negs_index_sampled

        # an_negs_index1 = list(set(node_f) - {a})
        an_negs_index1 = list(gs.neighbors(a))
        an_negs_index_sampled1 = list(np.random.choice(an_negs_index1, triplet_neg, replace=True))
        an_as1 = triplet_neg * [b]
        input_b += an_as1
        input_a += an_negs_index_sampled1

        un_an_target = torch.zeros(triplet_neg * 2)
        classifier_target = torch.cat((classifier_target, un_an_target), dim=0)
        index += 1

    cosine_target = torch.unsqueeze(2 * classifier_target - 1, dim=1)  # labels are [1,-1,-1]
    # classifier_target = torch.unsqueeze(classifier_target, dim=1)  # labels are [1,0,0]

    # [ina, inb] is all anchors and sampled non-anchors, cosine_target is their labels
    ina = torch.LongTensor(input_a)
    inb = torch.LongTensor(input_b)

    return ina, inb, cosine_target


def local_training(s_x, t_x, s_e, t_e, epochs=1000, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FedUA(s_x.shape[1], t_x.shape[1], 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        zs = model.s_forward(s_x.to(device), s_e.to(device))
        zt = model.t_forward(t_x.to(device), t_e.to(device))
        loss = model.intra_loss(zs, zt, s_e, t_e)
        loss.backward()
        optimizer.step()

    # Return local model state dict and loss
    return model.state_dict(), loss.item()

def upload_weights(model_weights):
    # Simulate uploading model weights to server
    # In practice, you would use a secure communication method
    return model_weights

def aggregate_weights(local_weights):
    global_weights = {}
    for key in local_weights[0].keys():
        global_weights[key] = torch.mean(torch.stack([local_weights[i][key] for i in range(len(local_weights))]), dim=0)
    return global_weights

def federated_training(s_data, t_data, epochs=5):
    local_weights = []
    for _ in range(epochs):
        s_weights, s_loss = local_training(*s_data)
        t_weights, t_loss = local_training(*t_data)

        # Upload weights to server
        local_weights.append(upload_weights(s_weights))
        local_weights.append(upload_weights(t_weights))

        # Aggregate weights
        global_weights = aggregate_weights(local_weights)
        
        # Update global model (simulated)
        # In practice, you'd load these weights back into the model

    return global_weights

def evaluate(model, s_x, t_x, s_e, t_e, gt_matrix):
    model.eval()
    with torch.no_grad():
        zs = model.s_forward(s_x, s_e)
        zt = model.t_forward(t_x, t_e)
        S = cosine_similarity(zs.cpu(), zt.cpu())
        result = get_statistics(S, gt_matrix)
    return result

if __name__ == "__main__":
    results = dict.fromkeys(('Acc', 'MRR', 'AUC', 'Hit', 'Precision@1', 'Precision@5', 'Precision@10', 'Precision@15', \
        'Precision@20', 'Precision@25', 'Precision@30', 'time'), 0) # save results
    N = 1 # repeat times for average, default: 1
    for i in range(N):
        start_time = time()
        args = parse_args()

        print('Load data...')
        # genetate adjacency matrix
        s_adj = get_adj(args.s_edge)
        t_adj = get_adj(args.t_edge)
        # generate edge_index(pyG version)
        s_e = get_edgeindex(args.s_edge)
        t_e = get_edgeindex(args.t_edge)
        s_num = s_adj.shape[0]
        t_num = t_adj.shape[0]
        # load train anchor links
        train_anchor = torch.LongTensor(np.loadtxt(args.train_path, dtype=int))
        # generate test anchor matrix for evaluation
        groundtruth_matrix = get_gt_matrix(args.gt_path, (s_num, t_num))

        # generate graph for negative sampling
        s_edge = np.loadtxt(args.s_edge, dtype=int)
        t_edge = np.loadtxt(args.t_edge, dtype=int)
        g_s = nx.Graph()
        g_s.add_edges_from(s_edge)
        g_t = nx.Graph()
        g_t.add_edges_from(t_edge)

        print('Generate deepwalk embeddings as input X...')
        s_x = node2vec(s_adj, P=1, Q=1, WINDOW_SIZE=10, NUM_WALKS=10, WALK_LENGTH=80, DIMENSIONS=128, \
            WORKERS=8, ITER=5, verbose=1)
        t_x = node2vec(t_adj, P=1, Q=1, WINDOW_SIZE=10, NUM_WALKS=10, WALK_LENGTH=80, DIMENSIONS=128, \
            WORKERS=8, ITER=5, verbose=1)
        s_x = torch.FloatTensor(s_x)
        t_x = torch.FloatTensor(t_x)
        time1 = time()
        t1 = time1 - start_time
        print('Finished in %.4f s!'%(t1))

        print('Generate embeddings...')
        s_embedding, t_embedding= get_embedding(s_x, t_x, s_e, t_e, g_s, g_t, train_anchor, groundtruth_matrix, dim=args.dim,\
             lr=args.lr, epochs=args.epochs, lamda=args.lamda, margin=args.margin, neg=args.neg)
        s_data = (s_x, t_x, s_e, t_e)
        t_data = (t_x, s_x, t_e, s_e)  # Simulate the data for the second user
        global_model_weights = federated_training(s_data, t_data, epochs=5)
        t2 = time() - time1
        print('Finished in %.4f s!'%(t2))

        # Save the global model weights
        torch.save(global_model_weights, os.path.join(args.out_path, 'global_model_weights.pth'))
        print('Global model weights saved.')

        # Load the model with the global weights for evaluation
        model = FedUA(s_x.shape[1], t_x.shape[1], 128)
        model.load_state_dict(global_model_weights)

        print('Evaluating the global model...')
        evaluation_results = evaluate(model, s_x, t_x, s_e, t_e, groundtruth_matrix)
        t3 = time() - start_time
        for k, v in evaluation_results.items():
            print(f'{k}: {v:.4f}')
        results['time'] += t3
        print('Total runtime: %.4f s'%(t3))

    for k in results.keys():
        results[k] /= N
        
    print('\nFedUA')
    print(args)
    print('Average results:')
    for k, v in results.items():
        print('%s: %.4f' % (k, v))