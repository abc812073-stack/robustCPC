import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph

class ManifoldRegularization:
    def __init__(self, k1=7, k2=7, regularization_lambda=1.1):
        self.k1 = k1
        self.k2 = k2
        self.regularization_lambda = regularization_lambda

    def construct_affinity_graphs(self, X, y):
        # 内在亲和图
        intrinsic_affinity = kneighbors_graph(X, self.k1, mode='connectivity', include_self=True).toarray()
        intrinsic_affinity = np.where((intrinsic_affinity + intrinsic_affinity.T) > 0, 1, 0)
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if y[i] != y[j]:
                    intrinsic_affinity[i, j] = 0

        # 外在亲和图
        extrinsic_affinity = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            distances = []
            for j in range(X.shape[0]):
                if i != j:
                    distances.append((j, np.linalg.norm(X[i] - X[j])))
            distances = sorted(distances, key=lambda x: x[1])
            for neighbor, _ in distances[:self.k2]:
                if y[i] != y[neighbor]:
                    extrinsic_affinity[i, neighbor] = 1
                    extrinsic_affinity[neighbor, i] = 1

        return intrinsic_affinity, extrinsic_affinity

    def compute_manifold_regularization(self, T, X, y):
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        intrinsic_affinity, extrinsic_affinity = self.construct_affinity_graphs(X_np, y_np)

        intrinsic_affinity = torch.tensor(intrinsic_affinity).to(X.device)
        extrinsic_affinity = torch.tensor(extrinsic_affinity).to(X.device)

        MI = 0
        MB = 0
        for i in range(T.shape[0]):
            for j in range(T.shape[0]):
                MI += intrinsic_affinity[i, j] * torch.norm(T[i] - T[j], p=2)
                MB += extrinsic_affinity[i, j] * torch.norm(T[i] - T[j], p=2)

        manifold_reg = self.regularization_lambda * (MI - MB)
        return manifold_reg