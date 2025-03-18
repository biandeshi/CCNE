import numpy as np
import networkx as nx
from collections import defaultdict
import random

def line(adj_train,
        P=1,  # 兼容参数，LINE不使用
        Q=1,  # 兼容参数，LINE不使用 
        WINDOW_SIZE=10,  # 兼容参数
        NUM_WALKS=10,  # 兼容参数
        WALK_LENGTH=80,  # 兼容参数
        DIMENSIONS=128,
        WORKERS=4,  # 兼容参数
        ITER=5,
        verbose=1,
        order=2,      # 1: 一阶相似性, 2: 二阶相似性
        neg_samples=5, # 负采样数量
        lr=0.025      # 学习率
    ):
    
    # 构建图并初始化嵌入
    G = nx.Graph(adj_train)
    num_nodes = G.number_of_nodes()
    node_list = list(G.nodes())
    
    # 初始化嵌入向量 (节点嵌入 + 上下文嵌入)
    embed_u = np.random.randn(num_nodes, DIMENSIONS) * 0.01  # 节点嵌入
    embed_v = np.random.randn(num_nodes, DIMENSIONS) * 0.01 if order==2 else None  # 上下文嵌入
    
    # 构建边数据集（带权）
    edges = []
    weights = []
    for u, v, data in G.edges(data=True):
        edges.append((u, v))
        weights.append(data.get('weight', 1.0))
    weights = np.array(weights)
    total_weight = weights.sum()
    
    # 负采样分布（按节点度数^0.75）
    node_degrees = np.array([G.degree(node) ** 0.75 for node in node_list])
    neg_dist = node_degrees / node_degrees.sum()
    
    # 训练主循环
    for epoch in range(ITER):
        loss = 0.0
        # 随机打乱边顺序
        indices = np.random.permutation(len(edges))
        
        for idx in indices:
            u, v = edges[idx]
            w = weights[idx]
            
            # 正样本梯度
            if order == 1:
                # 一阶相似性：直接计算u和v的相似度
                vec_u = embed_u[u]
                vec_v = embed_u[v]
                grad = w * (1 / (1 + np.exp(np.dot(vec_u, vec_v)))) - 1
                embed_u[u] += lr * grad * vec_v
                embed_u[v] += lr * grad * vec_u
                loss += -np.log(1 / (1 + np.exp(-grad)))
                
            else:
                # 二阶相似性：u与v的上下文相似度
                vec_u = embed_u[u]
                vec_v = embed_v[v]
                
                # 正样本梯度
                pos_grad = w * (1 / (1 + np.exp(np.dot(vec_u, vec_v)))) - 1
                embed_u[u] += lr * pos_grad * vec_v
                embed_v[v] += lr * pos_grad * vec_u
                loss += -np.log(1 / (1 + np.exp(-pos_grad)))
                
                # 负采样
                for _ in range(neg_samples):
                    neg_node = np.random.choice(node_list, p=neg_dist)
                    vec_neg = embed_v[neg_node]
                    neg_grad = w * (1 / (1 + np.exp(-np.dot(vec_u, vec_neg))))
                    embed_u[u] -= lr * neg_grad * vec_neg
                    embed_v[neg_node] -= lr * neg_grad * vec_u
                    loss += -np.log(1 / (1 + np.exp(neg_grad)))
        
        if verbose:
            print(f"Epoch {epoch+1}, Loss: {loss/len(edges):.4f}")
    
    # 返回拼接后的嵌入（一阶直接返回，二阶拼接节点嵌入）
    return embed_u if order == 1 else np.concatenate([embed_u, embed_v], axis=1)

# 保持与原node2vec相同的接口
def node2vec(adj_train, 
        P=1, Q=1, 
        WINDOW_SIZE=10, 
        NUM_WALKS=10, 
        WALK_LENGTH=80, 
        DIMENSIONS=128, 
        WORKERS=4, 
        ITER=5, 
        verbose=1):
    
    return line(adj_train, 
                DIMENSIONS=DIMENSIONS//2 if DIMENSIONS%2==0 else DIMENSIONS//2+1,
                ITER=ITER,
                verbose=verbose,
                order=2)