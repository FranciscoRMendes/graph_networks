import numpy as np
import torch


def create_adjacency_matrix(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    num_nodes = len(vertices)
    adj = np.zeros((num_nodes, num_nodes))
    for face in faces:
        for i in range(3):
            for j in range(i+1, 3):
                adj[face[i], face[j]] = 1
                adj[face[j], face[i]] = 1

    adj = torch.tensor(adj, dtype=torch.float32)
    adj += torch.eye(num_nodes)  # self-loops
    # Normalized Laplacian
    deg = adj.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
    L = torch.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt

    return adj, L


def create_adjacency_matrix_tilde(mesh):
    """
    Creates adjacency matrix and scaled Laplacian L_tilde for a mesh
    """
    vertices = mesh.vertices
    faces = mesh.faces
    num_nodes = len(vertices)

    # Build adjacency
    adj = np.zeros((num_nodes, num_nodes))
    for face in faces:
        for i in range(3):
            for j in range(i+1, 3):
                adj[face[i], face[j]] = 1
                adj[face[j], face[i]] = 1

    adj = torch.tensor(adj, dtype=torch.float32)
    adj += torch.eye(num_nodes)  # add self-loops

    # Normalized Laplacian
    deg = adj.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-8))
    L = torch.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt

    # Scaled Laplacian for Chebyshev
    lambda_max = torch.linalg.eigvals(L).real.max()
    L_tilde = (2.0 / lambda_max) * L - torch.eye(num_nodes)

    return adj, L_tilde

