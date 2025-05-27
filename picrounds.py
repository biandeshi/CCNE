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
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="CCNE")
    parser.add_argument('--gt_path', default='./data/douban/anchor/node,split=0.8.test.dict')
    parser.add_argument('--train_path', default='./data/douban/anchor/node,split=0.8.train.dict')
    parser.add_argument('--out_path', default='./data/douban/anchor/embeddings')
    parser.add_argument('--s_edge', default='./data/douban/online/raw/edgelist')
    parser.add_argument('--t_edge', default='./data/douban/offline/raw/edgelist')
    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--rounds', default=5, type=int)
    parser.add_argument('--lamda', default=1, type=float)
    parser.add_argument('--margin', default=0.9, type=float)
    parser.add_argument('--neg', default=1, type=int)
    # Percentage of the globel model
    parser.add_argument('--alpha', default=1.0, type=float)
    return parser.parse_args()

class Critic(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
    
    def forward(self, z):
        return self.fc(z).squeeze(-1)

class FedUA(torch.nn.Module):
    def __init__(self, input, output):#, not_share=False, is_bn=False, dropout=0):
        super().__init__()
        self.conv1 = GCNConv(input, 2 * output)
        self.conv2 = GCNConv(2 * output, output)
        self.activation = nn.ReLU()
        self.intra_loss = 0

    def forward(self, x, edge_index):
        '''
        embeddings of source network g_s(V_s, E_s):
        x is node feature vectors of g_s, with shape [V_s, n_feats], V_s is the number of nodes,
            and n_feats is the dimension of features;
        edge_index is edges, with shape [2, 2 * E_s], E_s is the number of edges
        '''
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        return self.conv2(x, edge_index)

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

        self.intra_loss =  pos_loss + neg_loss

def get_intra_loss(s_model, t_model):
    '''
    intra-network loss to preserve intra-network structural features:
    zs: embeddings of source network g_s;
    zt: embeddings of target network g_t
    '''
    return s_model.intra_loss + t_model.intra_loss


def get_embedding(s_x, t_x, s_e, t_e, g_s, g_t, s_model, t_model,anchor, gt_mat, dim=64, lr=0.001, lamda=1, margin=0.8, neg=1, epochs=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化Critic
    critic = Critic(dim).to(device)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.999))
    lambda_gp = 10  # 梯度惩罚系数

    s_x = s_x.to(device)
    t_x = t_x.to(device)
    s_e = s_e.to(device)
    t_e = t_e.to(device)
    s_model = s_model.to(device)
    t_model = t_model.to(device)

    s_optimizer = torch.optim.Adam(s_model.parameters(), lr=lr)
    t_optimizer = torch.optim.Adam(t_model.parameters(), lr=lr)
    cosine_loss=nn.CosineEmbeddingLoss(margin=margin)
    in_a, in_b, anchor_label = sample(anchor) # no hard negative sampling

    # print("Federated local learning...")
    for epoch in range(epochs):
        # 训练Critic 5次
        for _ in range(5):
            # 生成嵌入
            with torch.no_grad():
                zs = s_model(s_x, s_e)
                zt = t_model(t_x, t_e)
            
            # WGAN-GP训练
            real_samples = torch.randn_like(zs).to(device)
            
            # 计算Critic分数
            real_scores = critic(real_samples)
            fake_scores = critic(zs.detach())
            
            # 梯度惩罚
            alpha = torch.rand(zs.size(0), 1).to(device)
            interpolates = (alpha * real_samples + (1 - alpha) * zs.detach()).requires_grad_(True)
            crit_interpolates = critic(interpolates)
            gradients = torch.autograd.grad(
                outputs=crit_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(crit_interpolates),
                create_graph=True,
            )[0]
            grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            # Critic损失
            loss_c = -(torch.mean(real_scores) - torch.mean(fake_scores)) + lambda_gp * grad_penalty
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        s_model.train()
        t_model.train()
        s_optimizer.zero_grad()
        t_optimizer.zero_grad()
        # in_a, in_b, anchor_label = sample(anchor, g_s, g_t, neg=neg)
        zs = s_model.forward(s_x, s_e)
        zt = t_model.forward(t_x, t_e)
        
        s_model.single_recon_loss(zs, s_e)
        t_model.single_recon_loss(zt, t_e)

        intra_loss = s_model.intra_loss + t_model.intra_loss
        anchor_label = anchor_label.view(-1).to(device)
        inter_loss = cosine_loss(zs[in_a], zt[in_b], anchor_label)

        # 对抗损失
        fake_scores = critic(zs)
        loss_adv = -torch.mean(fake_scores)

        loss = intra_loss + lamda * inter_loss + 0.1 * loss_adv
        loss.backward()
        s_optimizer.step()
        t_optimizer.step()
        # if epoch % 100 == 0:
        #     p10 = evaluate(zs, zt, gt_mat)
        #     print('Epoch: {:03d}, intra_loss: {:.8f}, inter_loss: {:.8f}, loss_train: {:.8f}, precision_10: {:.8f}'.format(epoch,\
        #         intra_loss, inter_loss, loss, p10))
    
    # print("Federated local learning has been done...\n")
    s_model.eval()
    t_model.eval()
    s_embedding = s_model.forward(s_x, s_e)
    t_embedding = t_model.forward(t_x, t_e)
    s_embedding = s_embedding.detach().cpu()
    t_embedding = t_embedding.detach().cpu()
    return s_model.state_dict(), t_model.state_dict(), s_embedding, t_embedding

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

def sample(anchor_train):
    '''
    sample for each anchor without negative sampling
    '''
    anchor_train_len = anchor_train.shape[0]
    anchor_train_a_list = np.array(anchor_train.T[0])
    anchor_train_b_list = np.array(anchor_train.T[1])
    input_a = []
    input_b = []
    classifier_target = torch.empty(0)
    
    for index in range(anchor_train_len):
        a = anchor_train_a_list[index]
        b = anchor_train_b_list[index]
        
        input_a.append(a)
        input_b.append(b)
        an_target = torch.ones(1)  
        classifier_target = torch.cat((classifier_target, an_target), dim=0)

    cosine_target = torch.unsqueeze(2 * classifier_target - 1, dim=1)  # labels are [1]
    
    # [ina, inb] is all anchors and sampled non-anchors, cosine_target is their labels
    ina = torch.LongTensor(input_a)
    inb = torch.LongTensor(input_b)

    return ina, inb, cosine_target


if __name__ == "__main__":
    results = dict.fromkeys(('Acc', 'MRR', 'AUC', 'Hit', 'Precision@1', 'Precision@5', 'Precision@10', 'Precision@15', \
        'Precision@20', 'Precision@25', 'Precision@30', 'time'), 0) # save results
    
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

    start_round = 63
    end_round = 100
    for rounds in range(start_round, end_round):
        # initial model
        s_model = FedUA(s_x.shape[1], args.dim)
        t_model = FedUA(t_x.shape[1], args.dim)
        globel_model = s_model
        global_model_state_dict = globel_model.state_dict()

        # Perform federated training
        # print("Performing federated learning...\n")
        for round in range(rounds):
            s_state_dict, t_state_dict, s_embedding, t_embedding = get_embedding(s_x, t_x, s_e, t_e, g_s, g_t, s_model, t_model, train_anchor, groundtruth_matrix, args.dim, 
                            args.lr, args.lamda, args.margin, args.neg, args.epochs)
            # Merge local model
            for key in global_model_state_dict.keys():
                global_model_state_dict[key] = (s_state_dict[key] + t_state_dict[key]) / 2
            # Distribute globel model to local
            for key in s_state_dict.keys():
                s_model.state_dict()[key] = global_model_state_dict[key] * args.alpha + s_state_dict[key] * (1 - args.alpha)
            for key in t_state_dict.keys():
                t_model.state_dict()[key] = global_model_state_dict[key] * args.alpha + t_state_dict[key] * (1 - args.alpha)

        # print("Finished federated learning!\n")
        S = cosine_similarity(s_embedding, t_embedding)  # Example evaluation logic
        result = get_statistics(S, groundtruth_matrix)
        t3 = time() - start_time
        # for k, v in result.items():
            # print(f'{k}: {v:.4f}')
            # results[k] += v

        # results['time'] += t3
        # print(f'Total runtime: {t3:.4f} s')
        print(f"round {rounds}: {result['Precision@10']:.4f}")

    # print('\nCCNE with Federated Learning')
    # print(args)
    # print('Average results:')
    # for k, v in results.items():
    #     print(f'{k}: {v:.4f}')