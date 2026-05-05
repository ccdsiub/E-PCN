import numpy as np
import h5py
import torch
from tqdm import tqdm
import os
import dgl
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import math
import gc
import time

from scipy.spatial import cKDTree
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, davies_bouldin_score, silhouette_score
from scipy.spatial.distance import cdist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

batch_size = 2048

# Argument Parser

parser = argparse.ArgumentParser(description='Aspen EPCN DeepCluster: Multi-Graph DECEncoder + DeepCluster on ASPEN Open Jets (unsupervised)')
parser.add_argument('--h5_path', type=str, default='downloaded_datasets/RunG_batch0.h5', help='Path to HDF5 dataset')
parser.add_argument('--max_iterations', type=int, default=500, help='Maximum DeepCluster iterations (default: 500)')
parser.add_argument('--epochs_per_iter', type=int, default=1, help='Training epochs per iteration (default: 1)')
parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size (default: 2048)')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (default: cuda)')
parser.add_argument('--model_architecture', type=str, default=f'EPCN-DeepCluster-{batch_size}-BN', help='Model architecture name')
parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters (default: 5)')
parser.add_argument('--knn_k', type=int, default=3, help='KNN k for graph construction (default: 3)')
parser.add_argument('--num_workers', type=int, default=4, help='DataLoader num_workers (default: 4)')
parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding dimension (default: 32)')
parser.add_argument('--encoder_lr', type=float, default=1e-3, help='Encoder learning rate (default: 1e-3)')
parser.add_argument('--classifier_lr_mult', type=float, default=10.0, help='Classifier LR multiplier vs encoder (default: 10)')
parser.add_argument('--lr_milestones', nargs='+', type=int, default=[200, 350], help='LR decay milestones (default: 200 350)')
parser.add_argument('--lr_gamma', type=float, default=0.1, help='LR decay factor (default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (default: 1e-5)')
parser.add_argument('--min_cluster_size', type=int, default=100, help='Min samples per cluster (default: 100)')
parser.add_argument('--convergence_nmi', type=float, default=0.99, help='NMI threshold for convergence (default: 0.99)')
parser.add_argument('--plateau_patience', type=int, default=15, help='Stop if DBI not improved for N iterations (default: 15)')

args = parser.parse_args()

maxIterations = args.max_iterations
epochsPerIter = args.epochs_per_iter
batchSize = args.batch_size
device = args.device
modelArchitecture = args.model_architecture
n_clusters = args.n_clusters
knn_k = args.knn_k
num_workers = args.num_workers
embedding_dim = args.embedding_dim
encoder_lr = args.encoder_lr
classifier_lr_mult = args.classifier_lr_mult
lr_milestones = args.lr_milestones
lr_gamma = args.lr_gamma
weight_decay = args.weight_decay
min_cluster_size = args.min_cluster_size
convergence_nmi = args.convergence_nmi
plateau_patience = args.plateau_patience


# Physics Functions: edge weight computation

def get_pTmin(part_i, part_j):
    pT_i = part_i[:, 4]
    pT_j = part_j[:, 4]
    pTmin = torch.minimum(pT_i, pT_j)
    return pTmin

def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

def rapidity(part_n):
    energy = part_n[:, 3]
    pz = part_n[:, 2]
    rapidity = 0.5 * torch.log(1 + (2 * pz) / (energy - pz).clamp(min=1e-20))
    return rapidity

def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2

def get_delta(part_i, part_j, eps=1e-8):
    rap_i = rapidity(part_i)
    rap_j = rapidity(part_j)
    phi_i = part_i[:, 6]
    phi_j = part_j[:, 6]
    delta = delta_r2(rap_i, phi_i, rap_j, phi_j).sqrt()
    return delta

def delta_weight(part_i, part_j, eps=1e-8):
    lndelta = torch.log(get_delta(part_i, part_j, eps))
    return lndelta

def kT_weight(part_i, part_j, eps=1e-8):
    pTmin = get_pTmin(part_i, part_j)
    delta_ij = get_delta(part_i, part_j)
    lnkT = torch.log((pTmin * delta_ij).clamp(min=eps))
    return lnkT

def Z_weight(part_i, part_j, eps=1e-8):
    pTi = part_i[:, 4]
    pTj = part_j[:, 4]
    pTmin = get_pTmin(part_i, part_j)
    lnZ = torch.log((pTmin / (pTi + pTj).clamp(min=eps)).clamp(min=eps))
    return lnZ

def to_m2(part_i, part_j, eps=1e-8):
    energy_i = part_i[:, 3]
    energy_j = part_j[:, 3]
    p_i = part_i[:, 0:3]
    p_j = part_j[:, 0:3]
    m2 = (energy_i + energy_j).square() - (p_i + p_j).square().sum(dim=1)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2

def mSquare_weight(part_i, part_j, eps=1e-8):
    lnm2 = torch.log(to_m2(part_i, part_j, eps=eps))
    return lnm2


# Dataset: dynamic HDF5 graph construction with 4 physics edge weights

class AspenEPCNDataset(torch.utils.data.Dataset):
    """Read ASPEN HDF5 on the fly and build KNN graphs dynamically per jet,
    then create 4 edge-weighted graph views (delta, kT, Z, mSquare).
    The h5py file handle is opened lazily per worker process (fork-safe).
    """

    def __init__(self, h5_path, knn_k=3):
        self.h5_path = h5_path
        self.knn_k = knn_k
        self._h5file = None
        self._pfcands = None
        self._jet_kin = None
        self._cache = {}

        # Read only the dataset length (close immediately)
        with h5py.File(h5_path, 'r') as f:
            self.length = f['PFCands'].shape[0]

    def _open_h5(self):
        """Lazy open. Called once per worker process on first __getitem__."""
        if self._h5file is None:
            self._h5file = h5py.File(self.h5_path, 'r')
            self._pfcands = self._h5file['PFCands']
            self._jet_kin = self._h5file['jet_kinematics']

    def is_valid(self, idx):
        """Fast check that only reads particle data and counts non-zero rows."""
        self._open_h5()
        particles = self._pfcands[idx]
        n_particles = np.count_nonzero(np.any(particles != 0, axis=1))
        return n_particles >= 2

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        try:
            self._open_h5()

            # Load raw data for this jet
            particles = self._pfcands[idx].astype(np.float32)  # (150, 11)
            jet_kin = self._jet_kin[idx].astype(np.float32)     # (4,) [pT, eta, phi, msoftdrop]

            # Remove zero-padded particles (rows where all 11 features are 0)
            mask = np.any(particles != 0, axis=1)
            particles = particles[mask]

            # Handle inf/nan in remaining real particles
            particles = np.nan_to_num(particles, nan=0.0, posinf=0.0, neginf=0.0)

            n_particles = particles.shape[0]

            if n_particles < 2:
                self._cache[idx] = None
                return None

            # Extract momentum components
            px = particles[:, 0]
            py = particles[:, 1]
            pz = particles[:, 2]

            jet_eta = jet_kin[1]
            jet_phi = jet_kin[2]

            # Compute 3 additional features
            pT = np.sqrt(px**2 + py**2)
            p = np.sqrt(px**2 + py**2 + pz**2)

            # Particle eta (safe for p ~ |pz|)
            with np.errstate(divide='ignore', invalid='ignore'):
                eta_particle = np.where(
                    (p - np.abs(pz)) > 1e-10,
                    0.5 * np.log((p + pz) / (p - pz + 1e-10)),
                    0.0
                )
            phi_particle = np.arctan2(py, px)

            deta = eta_particle - jet_eta
            dphi = np.arctan2(np.sin(phi_particle - jet_phi), np.cos(phi_particle - jet_phi))

            # Reorder features so physics functions work:
            # [px(0), py(1), pz(2), energy(3), pT(4), deta(5), dphi(6), original[4:11]]
            # Physics functions expect: pT at index 4, phi(dphi) at index 6
            features = np.column_stack([
                particles[:, 0:4],   # px(0), py(1), pz(2), energy(3)
                pT,                  # index 4: needed by get_pTmin, Z_weight
                deta,                # index 5
                dphi,                # index 6: needed by get_delta (as phi)
                particles[:, 4:],    # original indices 4-10
            ]).astype(np.float32)

            # Build KNN graph in (deta, dphi) space
            coords = np.column_stack([deta, dphi])
            k = min(self.knn_k, n_particles - 1)

            tree = cKDTree(coords)
            _, indices = tree.query(coords, k + 1)  # +1 to exclude self

            # Build edge lists (bidirectional, vectorized)
            if indices.ndim == 2:
                neighbors = indices[:, 1:k+1]
            else:
                neighbors = indices[np.newaxis, :]

            sources = np.repeat(np.arange(n_particles), k)
            targets = neighbors.flatten()

            src = np.concatenate([sources, targets])
            dst = np.concatenate([targets, sources])

            # Create base graph with reordered features
            feat_tensor = torch.tensor(features, dtype=torch.float32)

            # Compute 4 edge-weighted graph views inline
            src_feats = feat_tensor[src]
            dst_feats = feat_tensor[dst]

            weight_functions = {
                'delta': delta_weight, 'kT': kT_weight,
                'Z': Z_weight, 'mSquare': mSquare_weight,
            }

            graphs = {}
            for wtype, wfunc in weight_functions.items():
                g = dgl.graph((src, dst), num_nodes=n_particles)
                g.ndata['feat'] = feat_tensor.clone()
                edge_w = wfunc(src_feats, dst_feats)
                # Sanitize edge weights (Aspen data can have inf/nan in particle features)
                edge_w = torch.nan_to_num(edge_w, nan=0.0, posinf=0.0, neginf=0.0)
                g.edata['weight'] = edge_w.float()
                graphs[wtype] = g

            result = {
                'graph_delta': graphs['delta'],
                'graph_kT': graphs['kT'],
                'graph_Z': graphs['Z'],
                'graph_mSquare': graphs['mSquare'],
            }
            self._cache[idx] = result
            return result
        except Exception:
            self._cache[idx] = None
            return None

    def __len__(self):
        return self.length


# Collate functions

def collateMultiGraphsOnly(batch):
    """For embedding collection (no labels)."""
    valid = [item for item in batch if item is not None]
    if len(valid) == 0:
        return None
    g_delta = dgl.batch([item['graph_delta'] for item in valid])
    g_kT = dgl.batch([item['graph_kT'] for item in valid])
    g_mSquare = dgl.batch([item['graph_mSquare'] for item in valid])
    g_Z = dgl.batch([item['graph_Z'] for item in valid])
    return (g_delta, g_kT, g_mSquare, g_Z)


def collateMultiGraphPseudoLabels(batch):
    """For DeepCluster training. Returns ((g_delta, g_kT, g_mSquare, g_Z), pseudo_labels)."""
    valid = [item for item in batch if item is not None]
    if len(valid) == 0:
        return None, None
    g_delta = dgl.batch([item['graph_delta'] for item in valid])
    g_kT = dgl.batch([item['graph_kT'] for item in valid])
    g_mSquare = dgl.batch([item['graph_mSquare'] for item in valid])
    g_Z = dgl.batch([item['graph_Z'] for item in valid])
    labels = torch.tensor([item['label'] for item in valid])
    return (g_delta, g_kT, g_mSquare, g_Z), labels


# GNN feature extractor

class GNNFeatureExtractor(nn.Module):
    def __init__(self, in_feats, hidden_feats, k):
        super(GNNFeatureExtractor, self).__init__()
        self.conv1 = dgl.nn.ChebConv(in_feats, hidden_feats, k)
        self.bn1 = nn.BatchNorm1d(hidden_feats)
        self.conv2 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.bn2 = nn.BatchNorm1d(hidden_feats)
        self.conv3 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.bn3 = nn.BatchNorm1d(hidden_feats)

        self.edgeconv1 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        self.bn_edge1 = nn.BatchNorm1d(hidden_feats)
        self.edgeconv2 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        self.bn_edge2 = nn.BatchNorm1d(hidden_feats)

    def forward(self, g):
        h = self.conv1(g, g.ndata['feat'])
        h = self.bn1(h)
        h = F.relu(h)

        h = self.edgeconv1(g, h)
        h = self.bn_edge1(h)
        h = F.relu(h)

        h = self.conv2(g, h)
        h = self.bn2(h)
        h = F.relu(h)

        h = self.edgeconv2(g, h)
        h = self.bn_edge2(h)
        h = F.relu(h)

        h = self.conv3(g, h)
        h = self.bn3(h)
        h = F.relu(h)

        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')

        return hg


# DECEncoder: 4 GNN branches + Conv1d fusion + FC layers

class DECEncoder(nn.Module):
    """Encoder: 4 GNN branches + Conv1d fusion + FC layers to embedding_dim."""
    def __init__(self, in_feats, hidden_feats, k, embedding_dim):
        super(DECEncoder, self).__init__()
        self.model_delta = GNNFeatureExtractor(in_feats, hidden_feats, k)
        self.model_kT = GNNFeatureExtractor(in_feats, hidden_feats, k)
        self.model_mSquare = GNNFeatureExtractor(in_feats, hidden_feats, k)
        self.model_Z = GNNFeatureExtractor(in_feats, hidden_feats, k)

        self.conv1d = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)
        self.bn_conv = nn.BatchNorm1d(4)

        self.fc1 = nn.Linear(4 * hidden_feats, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, embedding_dim)

    def forward(self, graph_delta, graph_kT, graph_mSquare, graph_Z):
        hg_delta = self.model_delta(graph_delta)
        hg_kT = self.model_kT(graph_kT)
        hg_mSquare = self.model_mSquare(graph_mSquare)
        hg_Z = self.model_Z(graph_Z)

        stacked = torch.stack([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)

        fused = self.conv1d(stacked)
        fused = self.bn_conv(fused)
        fused = self.relu(fused)

        fused = fused.view(fused.size(0), -1)

        h = self.fc1(fused)
        h = self.bn_fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        embedding = self.fc2(h)

        return embedding


# DeepCluster-specific classes

class ClassifierHead(nn.Module):
    """Linear classifier head. Reinitialized each DeepCluster iteration."""
    def __init__(self, embedding_dim, n_clusters):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(embedding_dim, n_clusters)

    def forward(self, z):
        return self.fc(z)


class PseudoLabelDataset(torch.utils.data.Dataset):
    """Wraps base dataset and overrides/adds labels with pseudo-labels from k-means."""
    def __init__(self, base_dataset, pseudo_labels):
        self.base_dataset = base_dataset
        self.pseudo_labels = pseudo_labels

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if item is None:
            return None
        item['label'] = int(self.pseudo_labels[idx])
        return item

    def __len__(self):
        return len(self.base_dataset)


# Helper functions

def reassign_empty_clusters(pseudo_labels, centroids, embeddings, min_cluster_size=100):
    """Handle empty or near-empty clusters by splitting the largest cluster."""
    n_clusters = centroids.shape[0]
    all_counts = np.zeros(n_clusters, dtype=np.int64)
    unique, counts = np.unique(pseudo_labels, return_counts=True)
    for u, c in zip(unique, counts):
        all_counts[u] = c

    empty_clusters = np.where(all_counts < min_cluster_size)[0]

    if len(empty_clusters) == 0:
        return pseudo_labels, centroids

    print(f"  Reassigning {len(empty_clusters)} empty/tiny clusters: {empty_clusters}")

    for empty_id in empty_clusters:
        largest_id = np.argmax(all_counts)
        largest_mask = (pseudo_labels == largest_id)
        largest_embeddings = embeddings[largest_mask]

        centroid = largest_embeddings.mean(axis=0)
        noise = np.random.randn(centroid.shape[0]) * 0.01
        centroids[largest_id] = centroid + noise
        centroids[empty_id] = centroid - noise

        dists_to_new = cdist(largest_embeddings, centroids[[largest_id, empty_id]])
        new_assignments = np.argmin(dists_to_new, axis=1)

        idx_in_full = np.where(largest_mask)[0]
        pseudo_labels[idx_in_full[new_assignments == 1]] = empty_id

        all_counts[empty_id] = (new_assignments == 1).sum()
        all_counts[largest_id] = (new_assignments == 0).sum()

    return pseudo_labels, centroids


def dunn_index(embeddings, labels):
    """Compute Dunn Index (on a subsample for efficiency)."""
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    max_intra = 0.0
    for label in unique_labels:
        cluster_points = embeddings[labels == label]
        if len(cluster_points) > 1:
            dists = cdist(cluster_points, cluster_points, 'euclidean')
            max_intra = max(max_intra, dists.max())

    if max_intra == 0:
        return 0.0

    min_inter = float('inf')
    for i, label_i in enumerate(unique_labels):
        for label_j in unique_labels[i+1:]:
            points_i = embeddings[labels == label_i]
            points_j = embeddings[labels == label_j]
            dists = cdist(points_i, points_j, 'euclidean')
            min_inter = min(min_inter, dists.min())

    return min_inter / max_intra


def evaluate_clustering_unsupervised(embeddings, pred_labels, kmeans_inertia, compute_dunn=False, dunn_sample_size=10000):
    """Compute unsupervised clustering evaluation metrics."""
    pred_labels = np.array(pred_labels)

    metrics = {}

    # Davies-Bouldin Index (lower is better)
    unique_pred = np.unique(pred_labels)
    if len(unique_pred) >= 2:
        metrics['DBI'] = davies_bouldin_score(embeddings, pred_labels)
    else:
        metrics['DBI'] = float('nan')

    # Silhouette Score (higher is better, subsample for speed)
    if len(unique_pred) >= 2:
        metrics['Silhouette'] = silhouette_score(
            embeddings, pred_labels,
            sample_size=min(50000, len(embeddings)),
            random_state=42
        )
    else:
        metrics['Silhouette'] = float('nan')

    # K-means inertia (lower is better)
    metrics['Inertia'] = kmeans_inertia

    # Dunn Index (higher is better, expensive; only on request with subsampling)
    if compute_dunn:
        if len(embeddings) > dunn_sample_size:
            sample_idx = np.random.RandomState(42).choice(len(embeddings), dunn_sample_size, replace=False)
            metrics['Dunn'] = dunn_index(embeddings[sample_idx], pred_labels[sample_idx])
        else:
            metrics['Dunn'] = dunn_index(embeddings, pred_labels)

    return metrics


def collect_embeddings(encoder, data_loader, device):
    """Pass all data through encoder (multi-graph), return embeddings as numpy array."""
    encoder.eval()
    all_embeddings = []
    with torch.no_grad():
        for graphs in tqdm(data_loader, desc="Collecting embeddings", leave=False):
            if graphs is None:
                continue
            g_delta, g_kT, g_mSquare, g_Z = graphs
            g_delta = g_delta.to(device)
            g_kT = g_kT.to(device)
            g_mSquare = g_mSquare.to(device)
            g_Z = g_Z.to(device)
            embedding = encoder(g_delta, g_kT, g_mSquare, g_Z)
            all_embeddings.append(embedding.cpu().numpy())
            del g_delta, g_kT, g_mSquare, g_Z, embedding
    return np.concatenate(all_embeddings, axis=0)


# Dataset setup and data split

print(f"Loading dataset from: {args.h5_path}")
dataset = AspenEPCNDataset(args.h5_path, knn_k=knn_k)
print(f"Total jets: {len(dataset)}")

# Random 80/10/10 split
rng = np.random.RandomState(42)
indices = rng.permutation(len(dataset))
n_train = int(0.8 * len(dataset))
n_val = int(0.1 * len(dataset))
train_indices = indices[:n_train]
val_indices = indices[n_train:n_train + n_val]
test_indices = indices[n_train + n_val:]

train = Subset(dataset, train_indices)
val = Subset(dataset, val_indices)
test = Subset(dataset, test_indices)

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Pre-scan all jets at once to find which have >= 2 real particles
print("Scanning for valid jets (vectorized)...")
with h5py.File(args.h5_path, 'r') as f:
    all_pfcands = f['PFCands'][:]  # shape (N, 150, 11); load full array once
particle_counts = np.count_nonzero(np.any(all_pfcands != 0, axis=2), axis=1)  # (N,)
valid_mask = particle_counts >= 2
del all_pfcands, particle_counts

valid_train_indices = [i for i in range(len(train)) if valid_mask[train_indices[i]]]
valid_train = Subset(train, valid_train_indices)

valid_val_indices = [i for i in range(len(val)) if valid_mask[val_indices[i]]]
valid_val = Subset(val, valid_val_indices)

valid_test_indices = [i for i in range(len(test)) if valid_mask[test_indices[i]]]
valid_test = Subset(test, valid_test_indices)

del valid_mask
print(f"Valid jets - Train: {len(valid_train)}/{len(train)}, Val: {len(valid_val)}/{len(val)}, Test: {len(valid_test)}/{len(test)}")

# Model save file

os.makedirs("modelSaveFiles", exist_ok=True)
base_filename = f"Aspen_EPCN_DeepCluster_{modelArchitecture}"
base_model_path = f"modelSaveFiles/{base_filename}.pt"

if os.path.exists(base_model_path):
    version = 1
    while True:
        versioned_filename = f"{base_filename}_{version}"
        versioned_model_path = f"modelSaveFiles/{versioned_filename}.pt"
        if not os.path.exists(versioned_model_path):
            modelSaveFile = versioned_model_path
            print(f"Model will be saved as: {versioned_filename}.pt (version {version})")
            break
        version += 1
else:
    modelSaveFile = base_model_path
    print(f"Model will be saved as: {base_filename}.pt (first version)")

versioned_model_name = os.path.splitext(os.path.basename(modelSaveFile))[0]

# wandb logging

in_feats = 14
hidden_feats = 64
chebFilterSize = 16

wandb.init(
    project="Aspen EPCN DeepCluster",
    name=versioned_model_name,
    config={
        "dataset": args.h5_path,
        "max_iterations": maxIterations,
        "epochs_per_iter": epochsPerIter,
        "batch_size": batchSize,
        "model": modelArchitecture,
        "device": device,
        "n_clusters": n_clusters,
        "knn_k": knn_k,
        "num_workers": num_workers,
        "embedding_dim": embedding_dim,
        "encoder_lr": encoder_lr,
        "classifier_lr_mult": classifier_lr_mult,
        "lr_milestones": lr_milestones,
        "lr_gamma": lr_gamma,
        "weight_decay": weight_decay,
        "min_cluster_size": min_cluster_size,
        "convergence_nmi": convergence_nmi,
        "plateau_patience": plateau_patience,
        "method": "DeepCluster (Caron et al., ECCV 2018)",
        "model_type": "EPCN (4-branch DECEncoder with BatchNorm)",
        "in_feats": in_feats,
        "hidden_feats": hidden_feats,
        "chebFilterSize": chebFilterSize,
        "train_size": len(valid_train),
        "val_size": len(valid_val),
        "test_size": len(valid_test),
        "model_save_file": modelSaveFile,
    }
)

print("wandb logging initialized.")

# Data loaders

trainLoaderNoShuffle = DataLoader(
    valid_train, batch_size=batchSize, shuffle=False,
    collate_fn=collateMultiGraphsOnly, drop_last=False,
    num_workers=0,
)
testLoader = DataLoader(
    valid_test, batch_size=batchSize, shuffle=False,
    collate_fn=collateMultiGraphsOnly, drop_last=False,
    num_workers=0,
)


# DeepCluster training (Caron et al., ECCV 2018)

print("DeepCluster training (EPCN DECEncoder, dynamic graphs, ASPEN Open Jets)")

# Create encoder (random initialization)
encoder = DECEncoder(in_feats, hidden_feats, chebFilterSize, embedding_dim).to(device)

# Watch encoder for wandb
wandb.watch(encoder, log='gradients', log_freq=100)

# Trackers
dbi_tracker = []
silhouette_tracker = []
loss_tracker = []
change_tracker = []
best_dbi = float('inf')
prev_pseudo_labels = None

# Create output directory for plots
imageSavePath = f'{versioned_model_name}'
os.makedirs(imageSavePath, exist_ok=True)

print(f"Starting DeepCluster (max {maxIterations} iterations, {n_clusters} clusters)...")
print(f"Encoder LR: {encoder_lr}, Classifier LR: {encoder_lr * classifier_lr_mult}")

for iteration in range(maxIterations):
    iterStartTime = time.time()

    # Extract embeddings
    print(f"\nIteration {iteration+1}/{maxIterations}")
    embeddings = collect_embeddings(encoder, trainLoaderNoShuffle, device)
    print(f"  Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    # L2-normalize embeddings before k-means
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    embeddings_normalized = embeddings / norms

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=iteration)
    pseudo_labels = kmeans.fit_predict(embeddings_normalized)

    # Handle empty clusters
    pseudo_labels, centroids = reassign_empty_clusters(
        pseudo_labels, kmeans.cluster_centers_, embeddings_normalized, min_cluster_size=min_cluster_size
    )

    # Print cluster distribution
    unique, counts = np.unique(pseudo_labels, return_counts=True)
    print(f"  Cluster sizes: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Evaluate clustering quality (unsupervised)
    metrics = evaluate_clustering_unsupervised(embeddings, pseudo_labels, kmeans.inertia_)
    print(f"  DBI={metrics['DBI']:.4f} Silhouette={metrics['Silhouette']:.4f} Inertia={metrics['Inertia']:.2f}")

    dbi_tracker.append(metrics['DBI'])
    silhouette_tracker.append(metrics['Silhouette'])

    # Check convergence
    if prev_pseudo_labels is not None:
        label_nmi = normalized_mutual_info_score(prev_pseudo_labels, pseudo_labels)
        change_pct = (pseudo_labels != prev_pseudo_labels).sum() / len(pseudo_labels)
        change_tracker.append(change_pct)
        print(f"  Label NMI vs prev: {label_nmi:.4f}, Change: {change_pct*100:.2f}%")

        if label_nmi > convergence_nmi and iteration > 10:
            print(f"  Converged: pseudo-labels stabilized (NMI={label_nmi:.4f} > {convergence_nmi})")
            prev_pseudo_labels = pseudo_labels.copy()
            break
    else:
        change_tracker.append(1.0)

    prev_pseudo_labels = pseudo_labels.copy()

    # Compute class weights (inverse cluster size)
    class_weights = torch.zeros(n_clusters, device=device)
    for u, c in zip(*np.unique(pseudo_labels, return_counts=True)):
        class_weights[u] = 1.0 / c
    class_weights = class_weights / class_weights.sum() * n_clusters
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Reinitialize classifier head
    classifier = ClassifierHead(embedding_dim, n_clusters).to(device)

    # Compute current LR based on iteration
    current_lr = encoder_lr
    for milestone in lr_milestones:
        if iteration >= milestone:
            current_lr *= lr_gamma

    optimizer = torch.optim.SGD([
        {'params': encoder.parameters(), 'lr': current_lr},
        {'params': classifier.parameters(), 'lr': current_lr * classifier_lr_mult},
    ], momentum=0.9, weight_decay=weight_decay)

    # Create pseudo-label DataLoader
    pseudo_dataset = PseudoLabelDataset(valid_train, pseudo_labels)
    pseudo_loader = DataLoader(
        pseudo_dataset, batch_size=batchSize, shuffle=True,
        collate_fn=collateMultiGraphPseudoLabels, drop_last=True,
        num_workers=0,
    )

    # Train encoder + classifier for 1 epoch
    encoder.train()
    classifier.train()
    runningLoss = 0.0

    for graphs, batch_pseudo_labels in tqdm(pseudo_loader, desc=f"  Training iter {iteration+1}", leave=False):
        if graphs is None:
            continue
        g_delta, g_kT, g_mSquare, g_Z = graphs
        g_delta = g_delta.to(device)
        g_kT = g_kT.to(device)
        g_mSquare = g_mSquare.to(device)
        g_Z = g_Z.to(device)
        batch_pseudo_labels = batch_pseudo_labels.to(device).long()

        optimizer.zero_grad()

        embeddings_batch = encoder(g_delta, g_kT, g_mSquare, g_Z)
        logits = classifier(embeddings_batch)
        loss = criterion(logits, batch_pseudo_labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(classifier.parameters()),
            max_norm=5.0
        )

        optimizer.step()
        runningLoss += loss.item()

        del g_delta, g_kT, g_mSquare, g_Z, batch_pseudo_labels, embeddings_batch, logits, loss

    avgLoss = runningLoss / len(pseudo_loader)
    loss_tracker.append(avgLoss)

    # Save best model (by DBI, lower is better)
    if metrics['DBI'] < best_dbi:
        best_dbi = metrics['DBI']
        torch.save({
            'encoder': encoder.state_dict(),
            'iteration': iteration,
            'dbi': best_dbi,
            'pseudo_labels': pseudo_labels,
        }, modelSaveFile)
        print(f"  Saved best model (DBI: {best_dbi:.4f})")

    # Timing and logging
    iterTime = time.time() - iterStartTime
    print(f"  Loss={avgLoss:.4f} LR={current_lr:.6f} Time={iterTime/60:.2f}min")

    wandb.log({
        "DeepCluster/Iteration": iteration + 1,
        "DeepCluster/Train_Loss": avgLoss,
        "DeepCluster/DBI": metrics['DBI'],
        "DeepCluster/Silhouette": metrics['Silhouette'],
        "DeepCluster/Inertia": metrics['Inertia'],
        "DeepCluster/LR": current_lr,
        "DeepCluster/Assignment_Change_%": change_tracker[-1] * 100,
    })

    # Plateau early stopping (DBI, lower is better)
    if len(dbi_tracker) > plateau_patience:
        best_in_recent = min(dbi_tracker[-plateau_patience:])
        best_before_recent = min(dbi_tracker[:-plateau_patience])
        if best_in_recent >= best_before_recent - 0.001:
            print(f"  Plateau: no DBI improvement for {plateau_patience} iterations. Stopping.")
            break

    # Cleanup
    del embeddings, embeddings_normalized, pseudo_labels, pseudo_dataset, pseudo_loader
    del classifier, optimizer, criterion, kmeans
    torch.cuda.empty_cache()
    gc.collect()

print(f"\nDeepCluster complete. Best DBI: {best_dbi:.4f}")


# Final evaluation

print("Final evaluation")

# Load best model
print("Loading best model...")
best_model = torch.load(modelSaveFile, map_location=device)
encoder.load_state_dict(best_model['encoder'])
print(f"Loaded model from iteration {best_model['iteration']+1} (DBI: {best_model['dbi']:.4f})")

encoder.eval()

# Evaluate on train set
print("Evaluating on train set...")
train_embeddings = collect_embeddings(encoder, trainLoaderNoShuffle, device)
train_norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-10
train_normalized = train_embeddings / train_norms
train_kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
train_pred = train_kmeans.fit_predict(train_normalized)

train_metrics = evaluate_clustering_unsupervised(train_embeddings, train_pred, train_kmeans.inertia_)
print(f"Train metrics: DBI={train_metrics['DBI']:.4f} Silhouette={train_metrics['Silhouette']:.4f} "
      f"Inertia={train_metrics['Inertia']:.2f}")

# Evaluate on test set
print("Evaluating on test set...")
test_embeddings = collect_embeddings(encoder, testLoader, device)
test_norms = np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-10
test_normalized = test_embeddings / test_norms
test_pred = train_kmeans.predict(test_normalized)

test_metrics = evaluate_clustering_unsupervised(test_embeddings, test_pred, 0.0, compute_dunn=True)
print(f"Test metrics:  DBI={test_metrics['DBI']:.4f} Silhouette={test_metrics['Silhouette']:.4f} "
      f"Dunn={test_metrics['Dunn']:.4f}")

# Log final metrics
wandb.log({
    "Final/Train_DBI": train_metrics['DBI'],
    "Final/Train_Silhouette": train_metrics['Silhouette'],
    "Final/Train_Inertia": train_metrics['Inertia'],
    "Final/Test_DBI": test_metrics['DBI'],
    "Final/Test_Silhouette": test_metrics['Silhouette'],
    "Final/Test_Dunn": test_metrics['Dunn'],
})

# Print comparison table
print("\nTrain vs test comparison")
import pandas as pd
comparison_data = {
    'Metric': ['DBI', 'Silhouette'],
    'Train': [train_metrics['DBI'], train_metrics['Silhouette']],
    'Test': [test_metrics['DBI'], test_metrics['Silhouette']],
}
comparison_df = pd.DataFrame(comparison_data)
comparison_df['Diff'] = comparison_df['Train'] - comparison_df['Test']
print(comparison_df.to_string(index=False))
print(f"\nTest-only Dunn Index: {test_metrics['Dunn']:.4f}")


# Visualizations

print("\nCreating visualizations...")

# Cluster size distribution (train)
fig, ax = plt.subplots(figsize=(10, 6))
unique, counts = np.unique(train_pred, return_counts=True)
ax.bar(unique, counts, color='steelblue')
ax.set_title(f'{versioned_model_name} Cluster Size Distribution (Train)')
ax.set_xlabel('Cluster')
ax.set_ylabel('Count')
ax.set_xticks(unique)
plt.tight_layout()
plt.savefig(f'{imageSavePath}/Cluster_Size_Distribution_Train.png')
plt.close()

# Cluster size distribution (test)
fig, ax = plt.subplots(figsize=(10, 6))
unique_test, counts_test = np.unique(test_pred, return_counts=True)
ax.bar(unique_test, counts_test, color='coral')
ax.set_title(f'{versioned_model_name} Cluster Size Distribution (Test)')
ax.set_xlabel('Cluster')
ax.set_ylabel('Count')
ax.set_xticks(unique_test)
plt.tight_layout()
plt.savefig(f'{imageSavePath}/Cluster_Size_Distribution_Test.png')
plt.close()

# Training curves
if len(loss_tracker) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(loss_tracker)
    axes[0, 0].set_title('Cross-Entropy Loss (pseudo-labels)')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')

    axes[0, 1].plot(dbi_tracker)
    axes[0, 1].set_title('Davies-Bouldin Index (lower is better)')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('DBI')

    axes[1, 0].plot(silhouette_tracker)
    axes[1, 0].set_title('Silhouette Score (higher is better)')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Silhouette')

    axes[1, 1].plot([c * 100 for c in change_tracker])
    axes[1, 1].set_title('Assignment Change %')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Change %')

    plt.suptitle(f'{versioned_model_name} DeepCluster Training Curves')
    plt.tight_layout()
    plt.savefig(f'{imageSavePath}/DeepCluster_Training_Curves.png')
    plt.close()

# t-SNE visualization of test embeddings
print("Computing t-SNE...")
from sklearn.manifold import TSNE

tsne_sample_size = min(20000, len(test_embeddings))
tsne_idx = np.random.RandomState(42).choice(len(test_embeddings), tsne_sample_size, replace=False)
tsne_embeddings = test_normalized[tsne_idx]
tsne_labels = test_pred[tsne_idx]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_2d = tsne.fit_transform(tsne_embeddings)

fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=tsne_labels, cmap='tab10', s=1, alpha=0.5)
ax.set_title(f'{versioned_model_name} t-SNE of Test Embeddings (colored by cluster)')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
plt.savefig(f'{imageSavePath}/tSNE_Test_Embeddings.png', dpi=150)
plt.close()

# Log images to wandb
wandb.log({
    "Cluster Size Distribution (Train)": wandb.Image(f"{imageSavePath}/Cluster_Size_Distribution_Train.png"),
    "Cluster Size Distribution (Test)": wandb.Image(f"{imageSavePath}/Cluster_Size_Distribution_Test.png"),
})
if len(loss_tracker) > 0:
    wandb.log({"DeepCluster Training Curves": wandb.Image(f"{imageSavePath}/DeepCluster_Training_Curves.png")})
wandb.log({"t-SNE Test Embeddings": wandb.Image(f"{imageSavePath}/tSNE_Test_Embeddings.png")})

wandb.log({
    "Comparison Table": wandb.Table(dataframe=comparison_df),
})

# Save final model
torch.save({
    'encoder': encoder.state_dict(),
}, modelSaveFile)

wandb.save(modelSaveFile)
print(f"Final model saved to {modelSaveFile}")

wandb.finish()

# Final cleanup
del dataset, train, val, test
del trainLoaderNoShuffle, testLoader
del encoder
del train_embeddings, train_pred
del test_embeddings, test_pred
torch.cuda.empty_cache()
gc.collect()

print("Aspen EPCN DeepCluster done.")
