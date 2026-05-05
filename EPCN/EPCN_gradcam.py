import numpy as np
import pandas as pd
from operator import truth
import awkward as ak
import torch
from tqdm import tqdm
import os
import dgl
import pickle
import wandb
import GPUtil
import gc
import json
import argparse
import math

# Add argument parser
parser = argparse.ArgumentParser(
    description='Multi-Graph Neural Network Testing',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Example usage:
  python multi_graph_testing.py --batch_size 256 --device cuda --classification_level MultiGraph --model_architecture PCN-1024

  python multi_graph_testing.py --batch_size 512 --load_model Y --batch_dir /path/to/batches --output_dir results/

  python multi_graph_testing.py --help  # Show this help message
""")
batch_size = 256
parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size (default: 256)')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (default: cuda)')
parser.add_argument('--classification_level', type=str, default='Dynamic_All_Interactions-1x1_Conv', help='Classification level (default: Dynamic_All_Interactions-1x1_Conv)')
parser.add_argument('--model_architecture', type=str, default='PCN-256-OneCycleLR-BN_1', help='Model architecture name (default: PCN-256-OneCycleLR-BN_1)')
parser.add_argument('--model_type', type=str, default='DGCNN', help='Model type (default: DGCNN)')
parser.add_argument('--load_model', type=str, default='Y', choices=['Y', 'N'], help='Load from save file (default: Y)')
parser.add_argument('--batch_dir', type=str, default='batches', help='Directory containing batch files (default: batches)')
parser.add_argument('--wandb_project', type=str, default='Dynamic Multi-Graph Testing 1M', help='Wandb project name (default: Dynamic Multi-Graph Testing 1M)')
parser.add_argument('--checkpoint_freq', type=int, default=5, help='Save intermediate results every N batch sets (default: 5)')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results (default: auto-generated)')

args = parser.parse_args()

# Print configuration
print("Multi-graph testing configuration")
print(f"Batch Size: {args.batch_size}")
print(f"Device: {args.device}")
print(f"Classification Level: {args.classification_level}")
print(f"Model Architecture: {args.model_architecture}")
print(f"Model Type: {args.model_type}")
print(f"Load Model: {args.load_model}")
print(f"Batch Directory: {args.batch_dir}")
print(f"Wandb Project: {args.wandb_project}")
print(f"Checkpoint Frequency: {args.checkpoint_freq}")
print(f"Output Directory: {args.output_dir if args.output_dir else 'auto-generated'}")

# Dynamic graph weight computation functions

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    torch.cuda.empty_cache()
    gc.collect()
    # Force multiple garbage collection cycles
    for _ in range(3):
        gc.collect()

def log_gpu_memory(stage=""):
    """Simple memory logging"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage} - GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

# get pTmin
def get_pTmin(part_i, part_j):
    pT_i = part_i[:, 4]
    pT_j = part_j[:, 4]
    pTmin = torch.minimum(pT_i, pT_j)
    return pTmin

# Delta
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
    phi_i = part_i[:, 6] # part_dphi
    phi_j = part_j[:, 6]

    delta = delta_r2(rap_i, phi_i, rap_j, phi_j).sqrt()
    return delta

def delta_weight(part_i, part_j, eps=1e-8):
    lndelta = torch.log(get_delta(part_i, part_j, eps))
    return lndelta

# kT
def kT_weight(part_i, part_j, eps=1e-8):
    pTmin = get_pTmin(part_i, part_j)
    delta_ij = get_delta(part_i, part_j)
    lnkT = torch.log((pTmin * delta_ij).clamp(min=eps))
    return lnkT

# Z
def Z_weight(part_i, part_j, eps=1e-8):
    pTi = part_i[:, 4]
    pTj = part_j[:, 4]
    pTmin = get_pTmin(part_i, part_j)
    lnZ = torch.log((pTmin / (pTi + pTj).clamp(min=eps)).clamp(min=eps))
    return lnZ

# mSquare
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

# Compute edge weight for a given weight type (GPU calculation only, results returned on CPU)
def compute_edge_weights_gpu(base_graph_cpu, weight_type, device):
    # Compute edge weights on GPU using node features and edge indices; keep graphs on CPU
    weight_functions = {
        'delta': delta_weight,
        'kT': kT_weight,
        'Z': Z_weight,
        'mSquare': mSquare_weight,
    }
    weight_func = weight_functions[weight_type]

    # Edge indices on CPU
    src_nodes_cpu, dst_nodes_cpu = base_graph_cpu.edges()
    src_nodes_cpu = src_nodes_cpu.detach().clone()
    dst_nodes_cpu = dst_nodes_cpu.detach().clone()

    # Move only what is required to GPU
    points_cpu = base_graph_cpu.ndata['feat']
    points_gpu = points_cpu.to(device)
    src_gpu = src_nodes_cpu.to(device)
    dst_gpu = dst_nodes_cpu.to(device)

    # Gather endpoint features for all edges
    src_points = points_gpu.index_select(0, src_gpu)
    dst_points = points_gpu.index_select(0, dst_gpu)

    with torch.no_grad():
        edge_weights_gpu = weight_func(src_points, dst_points)

    # Bring weights back to CPU for CPU graphs
    edge_weights_cpu = edge_weights_gpu.detach().to('cpu')

    # Cleanup GPU tensors
    del src_points, dst_points, edge_weights_gpu, points_gpu, src_gpu, dst_gpu
    torch.cuda.empty_cache()

    return edge_weights_cpu

# Create new graph with same structure but different edge weights (graph stays on CPU)
def create_weighted_graphs(base_graph_cpu, weight_type, device):
    # Compute edge weights on GPU, return on CPU
    edge_weights_cpu = compute_edge_weights_gpu(base_graph_cpu, weight_type, device)

    # Create new graph on CPU with same structure
    src_nodes_cpu, dst_nodes_cpu = base_graph_cpu.edges()
    new_graph_cpu = dgl.graph((src_nodes_cpu, dst_nodes_cpu), num_nodes=base_graph_cpu.num_nodes(), device='cpu')

    # Assign edge weights on CPU
    new_graph_cpu.edata['weight'] = edge_weights_cpu.float().contiguous()

    # Copy node features and any other node data (on CPU)
    # Data is already on CPU, just copy reference
    new_graph_cpu.ndata['feat'] = base_graph_cpu.ndata['feat']

    for key in base_graph_cpu.ndata.keys():
        if key != 'feat':
            new_graph_cpu.ndata[key] = base_graph_cpu.ndata[key]

    # Sanity checks
    assert new_graph_cpu.num_nodes() == base_graph_cpu.num_nodes(), "Node count mismatch!"
    assert new_graph_cpu.num_edges() == base_graph_cpu.num_edges(), "Edge count mismatch!"

    # Final GPU cleanup after this graph
    torch.cuda.empty_cache()

    return new_graph_cpu

# End of dynamic graph weight computation functions

class BatchedMultiGraphDataset(dgl.data.DGLDataset):
    def __init__(self, jetNames, k, batchDir='batches', loadFromDisk=False):

        self.jetNames = jetNames
        self.batchDir = batchDir

        # Collect base graph batch files for each jet type
        self.batch_files = {}
        self.sampleCountPerClass = []

        for jetType in jetNames:
            if type(jetType) != list:
                # Look for base graph files in batchDir/jetType/
                jet_dir = os.path.join(batchDir, jetType)
                if os.path.exists(jet_dir):
                    # Look for files with pattern {jetType}_{index}.pkl
                    batch_files = sorted([f for f in os.listdir(jet_dir)
                                        if f.endswith('.pkl') and f.startswith(jetType + '_')])
                    self.batch_files[jetType] = [os.path.join(jet_dir, f) for f in batch_files]
                    print(f'{jetType}: Found {len(batch_files)} base graph batch files')
                else:
                    print(f'Warning: Directory not found: {jet_dir}')
                    self.batch_files[jetType] = []

                self.sampleCountPerClass.append(len(self.batch_files[jetType]))
            else:
                # Handle list of jet types (if needed)
                combined_files = []
                for item in jetType:
                    jet_dir = os.path.join(batchDir, item)
                    if os.path.exists(jet_dir):
                        # Look for files with pattern {item}_{index}.pkl
                        batch_files = sorted([f for f in os.listdir(jet_dir)
                                            if f.endswith('.pkl') and f.startswith(item + '_')])
                        item_files = [os.path.join(jet_dir, f) for f in batch_files]
                        combined_files.extend(item_files)

                self.batch_files[str(jetType)] = combined_files
                self.sampleCountPerClass.append(len(combined_files))

        # Create a flat list of all batch files with their labels
        # Each entry is a path to a base graph batch file
        self.all_batch_files = []
        label = 0
        for jetType in jetNames:
            jet_key = jetType if type(jetType) != list else str(jetType)

            # Get the number of batch files
            num_batches = len(self.batch_files[jet_key])

            for batch_idx in range(num_batches):
                batch_path = self.batch_files[jet_key][batch_idx]
                self.all_batch_files.append((batch_path, label))
            label += 1

        print(f'Total base graph batch files to process: {len(self.all_batch_files)}')

    def process(self):
        return
    
    def get_all_batch_files(self):
        """Get list of all batch file sets with their labels"""
        return self.all_batch_files
                
    def __getitem__(self, idx):
        # Not used: the batch processing path calls get_all_batch_files() instead
        raise NotImplementedError("Use get_all_batch_files() for memory-efficient processing")

    def __len__(self):
        return len(self.all_batch_files)

# GradCAM computation function
def compute_graph_type_importance(model_delta, model_kT, model_mSquare, model_Z, classifier,
                                   graph_delta, graph_kT, graph_mSquare, graph_Z, target_class=None):
    """
    Compute importance scores for each of the 4 graph types using GradCAM approach.
    Returns importance scores for [delta, kT, mSquare, Z]
    Memory-efficient: processes in eval mode with gradient tracking only for needed tensors
    """
    # Enable gradient computation for this specific computation
    model_delta.eval()
    model_kT.eval()
    model_mSquare.eval()
    model_Z.eval()
    classifier.eval()

    # Enable gradient tracking for GradCAM computation
    with torch.enable_grad():
        # Get embeddings with gradient tracking
        hg_delta = model_delta(graph_delta)
        hg_kT = model_kT(graph_kT)
        hg_mSquare = model_mSquare(graph_mSquare)
        hg_Z = model_Z(graph_Z)

        # Enable gradient tracking and retention for non-leaf tensors
        hg_delta.requires_grad_(True)
        hg_kT.requires_grad_(True)
        hg_mSquare.requires_grad_(True)
        hg_Z.requires_grad_(True)

        # Retain gradients for non-leaf tensors (needed for GradCAM)
        hg_delta.retain_grad()
        hg_kT.retain_grad()
        hg_mSquare.retain_grad()
        hg_Z.retain_grad()

        # Stack embeddings for 1x1 conv classifier
        stacked_features = torch.stack([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)  # (batch_size, 4, 64)

        # Get final logits from classifier
        logits = classifier(stacked_features)

        # If target_class not specified, use predicted class
        if target_class is None:
            target_class = logits.argmax(dim=1)

        # Compute score for target class (average across batch)
        if isinstance(target_class, torch.Tensor):
            score = logits.gather(1, target_class.unsqueeze(1)).mean()
        else:
            score = logits[:, target_class].mean()

        # Backward to compute gradients
        score.backward()

        # Compute importance as gradient * activation (GradCAM style)
        importances = []
        for hg in [hg_delta, hg_kT, hg_mSquare, hg_Z]:
            if hg.grad is not None:
                # Importance = mean(|gradient * activation|) across batch and features
                importance = (hg.grad.abs() * hg.abs()).mean().item()
                importances.append(importance)
            else:
                importances.append(0.0)

        # Clean up gradients immediately
        del hg_delta, hg_kT, hg_mSquare, hg_Z, stacked_features, logits, score
        torch.cuda.empty_cache()

    return importances

# Checkpoint management functions
def save_checkpoint(checkpoint_file, processed_files, results):
    """Save checkpoint with processed files and accumulated results."""
    checkpoint_data = {
        'processed_files': processed_files,
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'total_processed': results['total_processed'],
        'class_counts': results['class_counts'],  # For metrics calculation
        'gradcam_samples_processed': results.get('gradcam_samples_processed', 0)  # Track GradCAM progress
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    print(f"Checkpoint saved: {len(processed_files)} batch sets processed")

def load_checkpoint(checkpoint_file):
    """Load checkpoint and return processed files and results."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            # Check if this is an old checkpoint format (with tracking lists)
            if 'logitsTracker' in checkpoint_data or 'predictionsTracker' in checkpoint_data:
                print("WARNING: Old checkpoint format detected (contains tracking lists)")
                print("This checkpoint will be ignored to save memory. Starting fresh.")
                print(f"Removing old checkpoint: {checkpoint_file}")
                os.remove(checkpoint_file)
                return [], None

            # New format - load with backward compatibility for class_counts and gradcam
            results = {
                'confusion_matrix': np.array(checkpoint_data['confusion_matrix']),
                'total_processed': checkpoint_data['total_processed'],
                'class_counts': checkpoint_data.get('class_counts', {}),  # Default to empty dict if missing
                'gradcam_samples_processed': checkpoint_data.get('gradcam_samples_processed', 0)  # GradCAM progress
            }
            print(f"Checkpoint loaded: {len(checkpoint_data['processed_files'])} batch sets already processed")
            print(f"GradCAM samples already processed: {results['gradcam_samples_processed']}")
            return checkpoint_data['processed_files'], results
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from beginning")
            return [], None
    else:
        print("No checkpoint found, starting from beginning")
        return [], None

# Function to calculate and display current metrics
def calculate_and_display_metrics(cfs, class_counts, jetNames, total_processed, file_count, total_files):
    """Calculate and display current metrics from the confusion matrix."""
    print(f"\nResults after {file_count}/{total_files} batch sets ({total_processed} samples)")

    # Calculate overall accuracy from confusion matrix
    if np.sum(cfs) > 0:
        overall_accuracy = np.trace(cfs) / np.sum(cfs)
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

    # Calculate metrics for each class
    def calculateConfusionMetrics(confusion_matrix):
        num_classes = len(confusion_matrix)
        metrics = []

        for i in range(num_classes):
            true_positive = confusion_matrix[i][i]
            false_positive = np.sum(confusion_matrix[:, i]) - true_positive
            false_negative = np.sum(confusion_matrix[i, :]) - true_positive
            true_negative = np.sum(confusion_matrix) - true_positive - false_positive - false_negative

            accuracy = (true_positive + true_negative) / np.sum(confusion_matrix) if np.sum(confusion_matrix) > 0 else 0
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

            metrics.append([accuracy, precision, recall, specificity])

        return metrics

    metrics = calculateConfusionMetrics(cfs)
    metricsDF = pd.DataFrame(metrics, columns=['Accuracy', 'Precision', 'Recall', 'Specificity'], index=jetNames)

    # Calculate micro and macro averages
    microAvg = metricsDF.mean(axis=0)
    macroAvg = metricsDF.mean(axis=0)

    # Add micro and macro averages to the DataFrame
    metricsDF.loc['Micro Avg'] = microAvg
    metricsDF.loc['Macro Avg'] = macroAvg

    print("\nPer-Class Metrics:")
    print(metricsDF.round(4))

    # Display class distribution from class_counts dict
    print("\nClass Distribution (samples processed so far):")
    total_samples = sum(class_counts.values()) if class_counts else 0
    for jet_name in jetNames:
        count = class_counts.get(jet_name, 0)
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"  {jet_name}: {count} samples ({percentage:.1f}%)")

    return metricsDF

# Function to save intermediate results
def save_intermediate_results(imageSavePath, cfs, jetNames, classificationLevel, modelArchitecture,
                            file_count, total_files):
    """Save the latest intermediate visualization."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Save the latest confusion matrix (raw counts)
    try:
        plt.figure(figsize=(15, 15))
        ax = sns.heatmap(cfs, annot=True, fmt='g', cmap='Blues')
        ax.set_title(f'{classificationLevel} {modelArchitecture} Multi-Graph Confusion Matrix (Batch Sets: {file_count}/{total_files})')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        plt.savefig(f'{imageSavePath}/Confusion Matrix_Latest.png')
        plt.close()  # Use close() instead of clf() to free memory

    except Exception as e:
        print(f"Error saving intermediate confusion matrix: {e}")

# Multi-graph model definitions
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.batch import batch

# GNN feature extractor (returns embeddings)
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
        # Apply graph convolutional layers with batch normalization
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

        # Store the node embeddings in the node data directory
        g.ndata['h'] = h

        # Compute graph-level representations by taking global mean pooling
        hg = dgl.mean_nodes(g, 'h')

        return hg

# Classifier class with 1D convolution and batch normalization
class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        # 1D convolution with kernel_size=1 (1-to-1 convolution)
        # Input: (batch_size, 4, 64) -> Output: (batch_size, 4, 64)
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=1)
        self.bn_conv = nn.BatchNorm1d(4)  # Batch norm for 1D conv (4 channels)

        # Keep the same structure as before
        self.fc1 = torch.nn.Linear(4 * 64, hidden_dim)  # 4*64 = 256, same as before
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)  # Batch norm for fc1
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, 4, 64)
        # Apply 1D convolution across the 4 graph types
        x = self.conv1d(x)  # (batch_size, 4, 64)
        x = self.bn_conv(x)  # Apply batch norm
        x = self.relu(x)

        # Flatten to same dimension as before: (batch_size, 256)
        x = x.view(x.size(0), -1)

        # Same structure as before with batch norm
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Custom collate function for multiple graphs
def collateFunction(batch):
    graphs_delta = [item['graph_delta'] for item in batch]
    graphs_kT = [item['graph_kT'] for item in batch]
    graphs_mSquare = [item['graph_mSquare'] for item in batch]
    graphs_Z = [item['graph_Z'] for item in batch]
    labels = [item['label'] for item in batch]
    
    batched_graph_delta = dgl.batch(graphs_delta)
    batched_graph_kT = dgl.batch(graphs_kT)
    batched_graph_mSquare = dgl.batch(graphs_mSquare)
    batched_graph_Z = dgl.batch(graphs_Z)
    
    return (batched_graph_delta, batched_graph_kT, batched_graph_mSquare, batched_graph_Z), torch.tensor(labels)

# Process all jetTypes
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']

# For testing, use the original jet names
testingSet = Top + Vector + QCD + Higgs
jetNames = testingSet
print("Jet types to test:", jetNames)

# Create multi-graph dataset object
dataset = BatchedMultiGraphDataset(jetNames, 3, batchDir='batches', loadFromDisk=False)
dataset.process()

# Testing path (maxEpochs = 0) uses the batched approach
maxEpochs = 0  # Set to 0 for testing
batchSize = args.batch_size

# Device and model configuration
device = args.device
classificationLevel = args.classification_level
modelArchitecture = args.model_architecture
modelType = args.model_type
modelSaveFile = "modelSaveFiles/" + classificationLevel + modelArchitecture + ".pt"
load = True if args.load_model == 'Y' else False

# Checkpoint file
checkpoint_file = f"checkpoints/{classificationLevel}-{modelArchitecture}-multigraph-checkpoint.json"
os.makedirs("checkpoints", exist_ok=True)

in_feats = 16
hidden_feats = 64
out_feats = len(jetNames)  # Number of output classes
chebFilterSize = 16

# Start wandb logging
wandb.init(
    project=args.wandb_project, 
    name=f"{classificationLevel}-{modelArchitecture}-MultiGraph-Testing",
    config={
        "epochs": maxEpochs,
        "batch_size": batchSize,
        "model": modelArchitecture,
        "in_feats": in_feats,
        "hidden_feats": hidden_feats,
        "out_feats": out_feats,
        "device": device,
        "testing_mode": True,
        "graph_types": ["delta", "kT", "mSquare", "Z"]
    }
)

# Initialize multi-graph models
if modelType == "DGCNN":
    # Create 4 feature extractors for each graph type
    model_delta = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_kT = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_mSquare = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_Z = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    
    # Final classifier that takes concatenated features
    classifier = Classifier(hidden_feats * 4, hidden_feats, out_feats)
else:
    print("Invalid selection. Only DGCNN supported for multi-graph!")
    exit()

# Move models to device
model_delta.to(device)
model_kT.to(device)
model_mSquare.to(device)
model_Z.to(device)
classifier.to(device)

# Create a list of all models for easier handling
all_models = [model_delta, model_kT, model_mSquare, model_Z, classifier]

if load:
    checkpoint = torch.load(modelSaveFile)
    if isinstance(checkpoint, dict) and 'model_delta' in checkpoint:
        # Load multi-graph model
        model_delta.load_state_dict(checkpoint['model_delta'])
        model_kT.load_state_dict(checkpoint['model_kT'])
        model_mSquare.load_state_dict(checkpoint['model_mSquare'])
        model_Z.load_state_dict(checkpoint['model_Z'])
        classifier.load_state_dict(checkpoint['classifier'])
        print(f"Loaded multi-graph model from {modelSaveFile}")
    else:
        print("Error: Model file doesn't contain multi-graph architecture!")
        exit()

# Set all models to eval mode
for model in all_models:
    model.eval()

# Load checkpoint if exists
processed_files, checkpoint_results = load_checkpoint(checkpoint_file)

# Initialize tracking variables for testing
if checkpoint_results:
    cfs = checkpoint_results['confusion_matrix']
    total_processed = checkpoint_results['total_processed']
    class_counts = checkpoint_results['class_counts']
    gradcam_samples_processed = checkpoint_results.get('gradcam_samples_processed', 0)
else:
    cfs = np.zeros((out_feats, out_feats))
    total_processed = 0
    class_counts = {jet_name: 0 for jet_name in jetNames}
    gradcam_samples_processed = 0

# Get all batch file sets (need this to calculate files_processed_count)
all_batch_files = dataset.get_all_batch_files()
total_files = len(all_batch_files)
files_processed_count = len(processed_files)

# Create streaming output files for predictions (append mode)
predictions_stream_file = f'metrics/{classificationLevel}-{modelArchitecture}-MultiGraph-Predictions-Stream.txt'
probabilities_stream_file = f'metrics/{classificationLevel}-{modelArchitecture}-MultiGraph-Probabilities-Stream.txt'
gradcam_stream_file = f'metrics/{classificationLevel}-{modelArchitecture}-MultiGraph-GradCAM-Stream.txt'
os.makedirs('metrics', exist_ok=True)

# Initialize stream files (truncate if starting fresh, keep if resuming)
if files_processed_count == 0 and os.path.exists(predictions_stream_file):
    print(f"Removing old predictions stream file: {predictions_stream_file}")
    os.remove(predictions_stream_file)

if files_processed_count == 0 and os.path.exists(probabilities_stream_file):
    print(f"Removing old probabilities stream file: {probabilities_stream_file}")
    os.remove(probabilities_stream_file)

if gradcam_samples_processed == 0 and os.path.exists(gradcam_stream_file):
    print(f"Removing old GradCAM stream file: {gradcam_stream_file}")
    os.remove(gradcam_stream_file)

# Create results directory early
if args.output_dir:
    imageSavePath = args.output_dir
else:
    imageSavePath = f'{classificationLevel} {modelArchitecture} MultiGraph'
try:
    os.makedirs(imageSavePath, exist_ok=True)
except Exception as e:
    print(e)

print("Starting multi-graph batch-wise testing...")

# Display initial status if resuming
if files_processed_count > 0:
    print(f"\nResuming from checkpoint. Already processed {files_processed_count}/{total_files} batch sets.")
    metricsDF = calculate_and_display_metrics(cfs, class_counts, jetNames,
                                            total_processed, files_processed_count, total_files)

# Process each batch file one at a time, computing weighted graphs on the fly
with torch.no_grad():
    for file_idx, (batch_path, label) in enumerate(tqdm(all_batch_files, desc="Processing base graph batches")):

        # Skip if already processed
        batch_key = batch_path
        if batch_key in processed_files:
            continue

        try:
            print(f"\n{'='*80}")
            print(f"Processing batch {file_idx + 1}/{total_files}: {batch_path}")
            print(f"{'='*80}")

            # Load base graphs from this batch file
            print(f"Loading base graphs from: {batch_path}")
            with open(batch_path, 'rb') as f:
                base_graphs = pickle.load(f)

            batch_size = len(base_graphs)
            print(f"Loaded {batch_size} base graphs")

            # Compute all 4 weighted graph types per base graph
            print(f"Dynamically computing weighted graphs...")
            log_gpu_memory("before_graph_computation")

            batch_graphs = {
                'delta': [],
                'kT': [],
                'mSquare': [],
                'Z': []
            }

            for idx, base_graph in enumerate(tqdm(base_graphs, desc="Computing weighted graphs", leave=False)):
                # Create all 4 weighted graph types for this single base graph
                graph_delta = create_weighted_graphs(base_graph, 'delta', device)
                graph_kT = create_weighted_graphs(base_graph, 'kT', device)
                graph_mSquare = create_weighted_graphs(base_graph, 'mSquare', device)
                graph_Z = create_weighted_graphs(base_graph, 'Z', device)

                # Store the computed graphs
                batch_graphs['delta'].append(graph_delta)
                batch_graphs['kT'].append(graph_kT)
                batch_graphs['mSquare'].append(graph_mSquare)
                batch_graphs['Z'].append(graph_Z)

                # Clean up
                del graph_delta, graph_kT, graph_mSquare, graph_Z, base_graph

                # Periodic cleanup
                if idx % 50000 == 0 and idx > 0:
                    force_memory_cleanup()

            # Clean up base_graphs list
            del base_graphs
            force_memory_cleanup()
            log_gpu_memory("after_graph_computation")

            print(f"Computed all weighted graphs for {batch_size} samples")

            # Create labels for this batch
            batch_labels = [label] * batch_size

            # Create dataset for this batch
            batch_data = []
            for i in range(batch_size):
                batch_data.append({
                    'graph_delta': batch_graphs['delta'][i],
                    'graph_kT': batch_graphs['kT'][i],
                    'graph_mSquare': batch_graphs['mSquare'][i],
                    'graph_Z': batch_graphs['Z'][i],
                    'label': batch_labels[i]
                })

            # Create DataLoader for this batch with specified batch size
            batch_loader = DataLoader(batch_data, batch_size=batchSize, shuffle=False,
                                    collate_fn=collateFunction, drop_last=False)

            print(f"Running inference on {len(batch_loader)} mini-batches...")

            # Process this batch file in mini-batches
            for mini_batch_idx, (mini_batch_graphs, mini_batch_labels) in enumerate(tqdm(batch_loader,
                                                           desc=f"Testing batch {file_idx + 1}",
                                                           leave=False)):
                # Unpack graphs
                graph_delta, graph_kT, graph_mSquare, graph_Z = mini_batch_graphs

                # Move to device
                graph_delta = graph_delta.to(device)
                graph_kT = graph_kT.to(device)
                graph_mSquare = graph_mSquare.to(device)
                graph_Z = graph_Z.to(device)
                mini_batch_labels = mini_batch_labels.to(device)

                # Compute graph type importance for every mini-batch
                # Compute immediately and stream to disk to keep memory bounded
                try:
                    importances = compute_graph_type_importance(
                        model_delta, model_kT, model_mSquare, model_Z, classifier,
                        graph_delta, graph_kT, graph_mSquare, graph_Z, target_class=None
                    )

                    # Get the majority class in this mini-batch for tracking
                    # (Since mini-batch is dominated by one class usually)
                    batch_labels_cpu = mini_batch_labels.cpu()
                    unique_labels, counts = torch.unique(batch_labels_cpu, return_counts=True)
                    majority_class = unique_labels[counts.argmax()].item()

                    # Stream GradCAM results to disk immediately (one line per mini-batch)
                    # Format: class_idx,delta,kT,mSquare,Z
                    with open(gradcam_stream_file, 'a') as f:
                        f.write(f"{majority_class},{importances[0]},{importances[1]},{importances[2]},{importances[3]}\n")

                    gradcam_samples_processed += mini_batch_labels.size(0)

                except Exception as e:
                    print(f"Warning: GradCAM computation failed for mini-batch {mini_batch_idx}: {e}")
                    # Continue with regular inference even if GradCAM fails

                # Regular inference without gradients
                with torch.no_grad():
                    # Get embeddings from each graph type
                    hg_delta = model_delta(graph_delta)
                    hg_kT = model_kT(graph_kT)
                    hg_mSquare = model_mSquare(graph_mSquare)
                    hg_Z = model_Z(graph_Z)

                    # Stack embeddings for 1x1 conv classifier
                    stacked_features = torch.stack([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)  # (batch_size, 4, 64)

                    # Get final logits from classifier
                    logits = classifier(stacked_features)
                    predictions = logits.argmax(dim=1)

                    # Compute probabilities (softmax) for AUC/AUPR
                    probabilities = torch.softmax(logits, dim=1)

                    # Stream predictions to disk (append mode)
                    with open(predictions_stream_file, 'a') as f:
                        for pred, target in zip(predictions.cpu().tolist(), mini_batch_labels.cpu().tolist()):
                            f.write(f"{pred},{target}\n")

                    # Stream probabilities to disk (append mode)
                    # Format: target,prob_class0,prob_class1,...,prob_classN
                    with open(probabilities_stream_file, 'a') as f:
                        probs_cpu = probabilities.cpu()
                        targets_cpu = mini_batch_labels.cpu()
                        for target_idx in range(len(targets_cpu)):
                            target = targets_cpu[target_idx].item()
                            probs = probs_cpu[target_idx].tolist()
                            # Write target followed by all probability values
                            f.write(f"{target},{','.join([f'{p:.6f}' for p in probs])}\n")

                    # Update confusion matrix and class counts
                    predictions_cpu = predictions.cpu()
                    labels_cpu = mini_batch_labels.cpu()
                    for idx in range(len(predictions_cpu)):
                        pred = predictions_cpu[idx].item()
                        target = labels_cpu[idx].item()
                        cfs[pred][target] += 1
                        class_counts[jetNames[target]] += 1

                # Clean up GPU memory
                del graph_delta, graph_kT, graph_mSquare, graph_Z
                del hg_delta, hg_kT, hg_mSquare, hg_Z, stacked_features
                del mini_batch_graphs, mini_batch_labels, logits, predictions, probabilities
                del predictions_cpu, labels_cpu, probs_cpu, targets_cpu
                torch.cuda.empty_cache() if device == 'cuda' else None

            # Update counters
            total_processed += batch_size
            processed_files.append(batch_key)
            files_processed_count += 1

            print(f"Completed batch {file_idx + 1}. Total processed: {total_processed}")

            # Calculate and display current metrics after each batch
            metricsDF = calculate_and_display_metrics(cfs, class_counts, jetNames,
                                                    total_processed, files_processed_count, total_files)

            # Save intermediate results (every args.checkpoint_freq batches or last batch)
            if files_processed_count % args.checkpoint_freq == 0 or files_processed_count == total_files:
                save_intermediate_results(imageSavePath, cfs, jetNames, classificationLevel, modelArchitecture,
                                        files_processed_count, total_files)

                # Log intermediate results to wandb
                wandb.log({
                    "Current_Overall_Accuracy": metricsDF.loc['Micro Avg', 'Accuracy'],
                    "Current_Batches_Processed": files_processed_count,
                    "Current_Samples_Processed": total_processed,
                    "Progress_Percentage": (files_processed_count / total_files) * 100,
                    "Current_Confusion_Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix_Latest.png") if os.path.exists(f"{imageSavePath}/Confusion Matrix_Latest.png") else None
                })

            # Clean up memory after each batch file
            del batch_graphs, batch_labels, batch_data, batch_loader
            force_memory_cleanup()
            log_gpu_memory(f"after_batch_{file_idx + 1}")

            # Save checkpoint after each batch
            results = {
                'confusion_matrix': cfs,
                'total_processed': total_processed,
                'class_counts': class_counts,
                'gradcam_samples_processed': gradcam_samples_processed
            }
            save_checkpoint(checkpoint_file, processed_files, results)

        except Exception as e:
            print(f"Error processing batch {file_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue

print("Multi-graph testing done.")

# Clean up checkpoint file after successful completion
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print("Checkpoint file cleaned up")

# Save results
try:
    os.makedirs(imageSavePath, exist_ok=True)
except Exception as e:
    print(e)

# Load streaming predictions for final analysis
print("Loading predictions from stream file...")
predictions_list = []
targets_list = []

if os.path.exists(predictions_stream_file):
    with open(predictions_stream_file, 'r') as f:
        for line in f:
            pred, target = line.strip().split(',')
            predictions_list.append(int(pred))
            targets_list.append(int(target))
    print(f"Loaded {len(predictions_list)} predictions from stream file")
else:
    print("No stream file found - using only confusion matrix")

print("Results saved!")

# Generate confusion matrix (raw counts)
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(15, 15)

ax = sns.heatmap(cfs, annot=True, fmt='g', cmap='Blues')
ax.set_title(f'{classificationLevel} {modelArchitecture} Multi-Graph Confusion Matrix')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')

print("\nRaw Confusion Matrix:")
print(cfs)
plt.savefig(f'{imageSavePath}/Confusion Matrix.png')
plt.clf()

# Generate ROC-AUC and PR-AUC curves from streamed probabilities
print("\nComputing ROC-AUC and PR-AUC from streamed probabilities...")

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Load probabilities from stream file (process in chunks if needed)
y_true_list = []
y_probs_list = []

if os.path.exists(probabilities_stream_file):
    print(f"Loading probabilities from: {probabilities_stream_file}")
    with open(probabilities_stream_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == out_feats + 1:  # target + N probabilities
                target = int(parts[0])
                probs = [float(p) for p in parts[1:]]
                y_true_list.append(target)
                y_probs_list.append(probs)

    print(f"Loaded {len(y_true_list)} probability samples")

    if len(y_true_list) > 0:
        # Convert to numpy arrays
        y_true = np.array(y_true_list)
        y_probs = np.array(y_probs_list)

        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=list(range(out_feats)))

        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(out_feats):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve and AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(out_feats)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(out_feats):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= out_feats

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        print(f"\nROC-AUC Scores:")
        print(f"  Macro-Average AUC: {roc_auc['macro']:.4f}")
        print(f"  Micro-Average AUC: {roc_auc['micro']:.4f}")
        for i, jet_name in enumerate(jetNames):
            print(f"  {jet_name}: {roc_auc[i]:.4f}")

        # Plot ROC curves
        plt.figure(figsize=(10, 8))

        # Plot macro and micro average
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})',
                color='deeppink', linestyle=':', linewidth=3)

        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})',
                color='navy', linestyle=':', linewidth=3)

        # Plot per-class ROC curves
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, out_feats))
        for i, (jet_name, color) in enumerate(zip(jetNames, colors)):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{jet_name} (AUC = {roc_auc[i]:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{classificationLevel} {modelArchitecture} - ROC Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{imageSavePath}/ROC-AUC.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Compute precision-recall curves and AUPR
        precision = dict()
        recall = dict()
        pr_auc = dict()

        for i in range(out_feats):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
            pr_auc[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])

        # Compute macro-average PR
        all_recall = np.unique(np.concatenate([recall[i] for i in range(out_feats)]))
        mean_precision = np.zeros_like(all_recall)
        for i in range(out_feats):
            mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
        mean_precision /= out_feats

        precision["macro"] = mean_precision
        recall["macro"] = all_recall
        pr_auc["macro"] = auc(recall["macro"], precision["macro"])

        # Compute micro-average PR
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), y_probs.ravel())
        pr_auc["micro"] = average_precision_score(y_true_bin, y_probs, average="micro")

        print(f"\nPR-AUC (Average Precision) Scores:")
        print(f"  Macro-Average AUPR: {pr_auc['macro']:.4f}")
        print(f"  Micro-Average AUPR: {pr_auc['micro']:.4f}")
        for i, jet_name in enumerate(jetNames):
            print(f"  {jet_name}: {pr_auc[i]:.4f}")

        # Plot PR curves
        plt.figure(figsize=(10, 8))

        # Plot macro and micro average
        plt.plot(recall["micro"], precision["micro"],
                label=f'Micro-average (AUPR = {pr_auc["micro"]:.4f})',
                color='deeppink', linestyle=':', linewidth=3)

        plt.plot(recall["macro"], precision["macro"],
                label=f'Macro-average (AUPR = {pr_auc["macro"]:.4f})',
                color='navy', linestyle=':', linewidth=3)

        # Plot per-class PR curves
        for i, (jet_name, color) in enumerate(zip(jetNames, colors)):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'{jet_name} (AUPR = {pr_auc[i]:.4f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'{classificationLevel} {modelArchitecture} - Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{imageSavePath}/PR-AUC.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Compute class-wise probability statistics
        print("\nComputing class-wise probability statistics...")

        # For each true class, compute statistics of predicted probabilities
        prob_stats = {}

        for class_idx in range(out_feats):
            # Get samples belonging to this true class
            class_mask = (y_true == class_idx)
            class_probs = y_probs[class_mask]  # (n_samples_in_class, n_classes)

            if len(class_probs) > 0:
                # Statistics for each predicted class probability
                stats_dict = {}
                for pred_class_idx in range(out_feats):
                    pred_probs = class_probs[:, pred_class_idx]
                    stats_dict[jetNames[pred_class_idx]] = {
                        'mean': pred_probs.mean(),
                        'std': pred_probs.std(),
                        'min': pred_probs.min(),
                        'max': pred_probs.max(),
                        'median': np.median(pred_probs)
                    }
                prob_stats[jetNames[class_idx]] = stats_dict

        # Print probability statistics for each true class
        print("\nClass-wise Probability Statistics:")
        print("(For each TRUE class, shows probability statistics for PREDICTED classes)")
        print("-" * 80)

        for true_class in jetNames:
            if true_class in prob_stats:
                print(f"\nTRUE CLASS: {true_class}")
                print(f"{'Predicted Class':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
                print("-" * 80)
                for pred_class in jetNames:
                    stats = prob_stats[true_class][pred_class]
                    # Highlight diagonal (correct class) with asterisk
                    marker = "*" if true_class == pred_class else " "
                    print(f"{pred_class:<14}{marker} {stats['mean']:.4f}     {stats['std']:.4f}     "
                          f"{stats['min']:.4f}     {stats['max']:.4f}     {stats['median']:.4f}")

        # Create DataFrame for easy export
        prob_stats_data = []
        for true_class in jetNames:
            if true_class in prob_stats:
                for pred_class in jetNames:
                    stats = prob_stats[true_class][pred_class]
                    prob_stats_data.append({
                        'True_Class': true_class,
                        'Predicted_Class': pred_class,
                        'Mean_Probability': stats['mean'],
                        'Std_Probability': stats['std'],
                        'Min_Probability': stats['min'],
                        'Max_Probability': stats['max'],
                        'Median_Probability': stats['median']
                    })

        prob_stats_df = pd.DataFrame(prob_stats_data)

        # Save to CSV
        prob_stats_file = f'{imageSavePath}/Class_Probability_Statistics.csv'
        prob_stats_df.to_csv(prob_stats_file, index=False)
        print(f"\nProbability statistics saved to: {prob_stats_file}")

        # Visualize probability distributions

        # Create heatmap of mean probabilities
        plt.figure(figsize=(12, 10))

        # Create matrix of mean probabilities
        mean_prob_matrix = np.zeros((out_feats, out_feats))
        for i, true_class in enumerate(jetNames):
            if true_class in prob_stats:
                for j, pred_class in enumerate(jetNames):
                    mean_prob_matrix[i, j] = prob_stats[true_class][pred_class]['mean']

        # Plot heatmap
        sns.heatmap(mean_prob_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=jetNames, yticklabels=jetNames,
                    cbar_kws={'label': 'Mean Probability'}, vmin=0, vmax=1)
        plt.title(f'{classificationLevel} {modelArchitecture} - Mean Predicted Probabilities\n(Rows: True Class, Columns: Predicted Class)',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{imageSavePath}/Mean_Probability_Heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create box plots showing probability distributions for diagonal (correct predictions)
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Correct class probabilities (diagonal)
        correct_probs_data = []
        correct_probs_labels = []
        for class_idx, true_class in enumerate(jetNames):
            class_mask = (y_true == class_idx)
            if class_mask.sum() > 0:
                correct_class_probs = y_probs[class_mask, class_idx]
                correct_probs_data.append(correct_class_probs)
                correct_probs_labels.append(true_class)

        bp1 = axes[0].boxplot(correct_probs_data, labels=correct_probs_labels, patch_artist=True)
        for patch, color in zip(bp1['boxes'], plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(correct_probs_labels)))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[0].set_title('Predicted Probability for Correct Class (Confidence)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Probability', fontsize=11)
        axes[0].set_xlabel('True Class', fontsize=11)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.05])

        # Plot 2: Maximum incorrect class probabilities (confusion indicator)
        max_incorrect_probs_data = []
        for class_idx, true_class in enumerate(jetNames):
            class_mask = (y_true == class_idx)
            if class_mask.sum() > 0:
                class_probs = y_probs[class_mask]
                # Get max probability excluding the correct class
                incorrect_probs = np.delete(class_probs, class_idx, axis=1)
                max_incorrect = incorrect_probs.max(axis=1)
                max_incorrect_probs_data.append(max_incorrect)

        bp2 = axes[1].boxplot(max_incorrect_probs_data, labels=correct_probs_labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(correct_probs_labels)))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1].set_title('Maximum Incorrect Class Probability (Confusion Risk)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Probability', fontsize=11)
        axes[1].set_xlabel('True Class', fontsize=11)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(f'{imageSavePath}/Probability_Distribution_BoxPlots.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Compute and print confidence metrics
        print("\nPrediction confidence metrics (per true class):")
        print(f"{'True Class':<15} {'Mean Correct':<15} {'Mean Max Wrong':<15} {'Confidence Gap':<15}")
        print("-" * 80)

        confidence_metrics = {}
        for class_idx, true_class in enumerate(jetNames):
            class_mask = (y_true == class_idx)
            if class_mask.sum() > 0:
                class_probs = y_probs[class_mask]
                correct_prob_mean = class_probs[:, class_idx].mean()

                incorrect_probs = np.delete(class_probs, class_idx, axis=1)
                max_incorrect_mean = incorrect_probs.max(axis=1).mean()

                confidence_gap = correct_prob_mean - max_incorrect_mean

                confidence_metrics[true_class] = {
                    'mean_correct_prob': correct_prob_mean,
                    'mean_max_wrong_prob': max_incorrect_mean,
                    'confidence_gap': confidence_gap
                }

                print(f"{true_class:<15} {correct_prob_mean:.4f}          {max_incorrect_mean:.4f}          {confidence_gap:.4f}")

        print("\nA higher confidence gap indicates more confident predictions.")

        # Clean up arrays
        del y_true, y_probs, y_true_bin, fpr, tpr, precision, recall
        del correct_probs_data, max_incorrect_probs_data
        gc.collect()

    else:
        print("No probability data available")
        roc_auc = {"macro": 0.0, "micro": 0.0}
        pr_auc = {"macro": 0.0, "micro": 0.0}
        prob_stats = {}
        confidence_metrics = {}
else:
    print(f"Probabilities stream file not found: {probabilities_stream_file}")
    roc_auc = {"macro": 0.0, "micro": 0.0}
    pr_auc = {"macro": 0.0, "micro": 0.0}
    prob_stats = {}
    confidence_metrics = {}

# Background rejection analysis
print("\nComputing background rejection metrics...")

# Background class for your dataset
BACKGROUND_CLASS_NAME = 'ZJetsToNuNu'  # QCD/background jets

if BACKGROUND_CLASS_NAME in jetNames and 'y_true' in locals() and 'y_probs' in locals():
    print(f"Background class: {BACKGROUND_CLASS_NAME}")
    print(f"Using {len(y_true)} samples for background rejection analysis")

    try:
        # Run complete background rejection analysis
        rejection_results, rejection_summary = add_background_rejection_analysis(
            y_true=y_true,
            y_proba=y_probs,
            class_names=jetNames,
            background_class_name=BACKGROUND_CLASS_NAME,
            save_dir=imageSavePath,
            wandb_log=True
        )

        # Save rejection results to metrics directory
        if rejection_summary is not None:
            rejection_summary.to_csv(
                f'metrics/{classificationLevel}-{modelArchitecture}-background_rejection.csv',
                index=False
            )
            print("\nBackground rejection analysis done.")
            print(f"Results saved to {imageSavePath}")

        del rejection_results
        gc.collect()

    except Exception as e:
        print(f"Error in background rejection analysis: {e}")
        import traceback
        traceback.print_exc()
        rejection_summary = None

elif BACKGROUND_CLASS_NAME not in jetNames:
    print(f"WARNING: Background class '{BACKGROUND_CLASS_NAME}' not found!")
    print(f"Available classes: {jetNames}")
    print("Skipping background rejection analysis.")
    rejection_summary = None
else:
    print("Probability data not available for background rejection.")
    rejection_summary = None

# Clean up lists
del y_true_list, y_probs_list
gc.collect()

# Calculate final metrics
def calculateConfusionMetrics(confusion_matrix):
    num_classes = len(confusion_matrix)
    metrics = []

    for i in range(num_classes):
        true_positive = confusion_matrix[i][i]
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive
        true_negative = np.sum(confusion_matrix) - true_positive - false_positive - false_negative

        accuracy = (true_positive + true_negative) / np.sum(confusion_matrix)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

        metrics.append([accuracy, precision, recall, specificity])

    return metrics

metrics = calculateConfusionMetrics(cfs)

classLabels = jetNames
metricsDF = pd.DataFrame(metrics, columns=['Accuracy', 'Precision', 'Recall', 'Specificity'], index=classLabels)

# Add AUC and AUPR columns if available
if 'roc_auc' in locals() and isinstance(roc_auc, dict) and len(roc_auc) > 0:
    # Add per-class AUC and AUPR scores
    auc_scores = [roc_auc.get(i, 0.0) for i in range(len(classLabels))]
    aupr_scores = [pr_auc.get(i, 0.0) for i in range(len(classLabels))]

    metricsDF['ROC-AUC'] = auc_scores
    metricsDF['PR-AUC'] = aupr_scores

# Calculate micro and macro averages
microAvg = metricsDF.mean(axis=0)
macroAvg = metricsDF.mean(axis=0)

# Override macro averages for AUC and AUPR with computed values
if 'roc_auc' in locals() and isinstance(roc_auc, dict):
    macroAvg['ROC-AUC'] = roc_auc.get('macro', 0.0)
    macroAvg['PR-AUC'] = pr_auc.get('macro', 0.0)
    microAvg['ROC-AUC'] = roc_auc.get('micro', 0.0)
    microAvg['PR-AUC'] = pr_auc.get('micro', 0.0)

# Add micro and macro averages to the DataFrame
metricsDF.loc['Micro Avg'] = microAvg
metricsDF.loc['Macro Avg'] = macroAvg

# Print the final metrics table
print("\nFinal results")
print(metricsDF)

# Print summary of key metrics
print("\nKey summary metrics:")
print(f"  Overall Accuracy: {metricsDF.loc['Micro Avg', 'Accuracy']:.4f}")
if 'ROC-AUC' in metricsDF.columns:
    print(f"  Macro ROC-AUC: {metricsDF.loc['Macro Avg', 'ROC-AUC']:.4f}")
    print(f"  Macro PR-AUC: {metricsDF.loc['Macro Avg', 'PR-AUC']:.4f}")

# GradCAM visualization and analysis
print("\nProcessing GradCAM results...")

# Load GradCAM results from stream file
gradcam_importances = []
gradcam_class_labels = []

if os.path.exists(gradcam_stream_file):
    print(f"Loading GradCAM data from: {gradcam_stream_file}")
    with open(gradcam_stream_file, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            # New format: class_idx,delta,kT,mSquare,Z (5 values)
            if len(values) == 5:
                gradcam_class_labels.append(int(values[0]))
                gradcam_importances.append([float(v) for v in values[1:]])
            # Old format (backward compatibility): delta,kT,mSquare,Z (4 values)
            elif len(values) == 4:
                gradcam_class_labels.append(-1)  # Unknown class
                gradcam_importances.append([float(v) for v in values])

    print(f"Loaded {len(gradcam_importances)} GradCAM samples")

    if len(gradcam_importances) > 0:
        # Convert to numpy array
        gradcam_array = np.array(gradcam_importances)
        gradcam_classes = np.array(gradcam_class_labels)

        # Compute overall average importance for each graph type
        avg_importances = gradcam_array.mean(axis=0)

        # Normalize to percentages
        total_importance = avg_importances.sum()
        importance_percentages = (avg_importances / total_importance) * 100

        graph_types = ['Delta', 'kT', 'mSquare', 'Z']

        print("\nOverall GradCAM importance scores:")
        for graph_type, importance, percentage in zip(graph_types, avg_importances, importance_percentages):
            print(f"{graph_type:12s}: {importance:.6f} ({percentage:.2f}%)")

        # Per-class GradCAM analysis
        print("\nPer-class GradCAM importance scores:")

        # Dictionary to store per-class statistics
        per_class_importances = {}
        per_class_percentages = {}

        for class_idx, jet_name in enumerate(jetNames):
            # Get samples belonging to this class
            class_mask = (gradcam_classes == class_idx)
            class_samples = gradcam_array[class_mask]

            if len(class_samples) > 0:
                # Compute average importance for this class
                class_avg_importances = class_samples.mean(axis=0)

                # Normalize to percentages
                class_total_importance = class_avg_importances.sum()
                class_importance_percentages = (class_avg_importances / class_total_importance) * 100

                per_class_importances[jet_name] = class_avg_importances
                per_class_percentages[jet_name] = class_importance_percentages

                print(f"\n{jet_name} ({len(class_samples)} mini-batches):")
                print("-" * 60)
                for graph_type, importance, percentage in zip(graph_types, class_avg_importances, class_importance_percentages):
                    print(f"  {graph_type:12s}: {importance:.6f} ({percentage:.2f}%)")
            else:
                print(f"\n{jet_name}: No GradCAM data available")

        # Create bar plot with gradient-based colors
        plt.figure(figsize=(10, 6))

        # Map importance to colors using gradient
        norm_importances = importance_percentages / importance_percentages.max()
        cmap = plt.cm.get_cmap('YlOrRd')
        bar_colors = [cmap(norm_imp) for norm_imp in norm_importances]

        bars = plt.bar(graph_types, importance_percentages, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=2)

        # Add value labels on bars
        for bar, val in zip(bars, importance_percentages):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.title(f'{classificationLevel} {modelArchitecture} - Graph Type Importance (GradCAM)', fontsize=14, fontweight='bold')
        plt.xlabel('Graph Type', fontsize=12)
        plt.ylabel('Relative Importance (%)', fontsize=12)
        plt.ylim(0, max(importance_percentages) * 1.2)
        plt.grid(axis='y', alpha=0.3, linestyle='--')

        # Add colorbar to bar plot
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=importance_percentages.min(), vmax=importance_percentages.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
        cbar.set_label('Importance (%)', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{imageSavePath}/GradCAM_Graph_Type_Importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create heatmap showing importance across batches (sample first 1000 for visualization)
        plt.figure(figsize=(12, 8))

        # Sample for visualization if too many batches
        if len(gradcam_array) > 1000:
            sample_indices = np.linspace(0, len(gradcam_array)-1, 1000, dtype=int)
            sampled_importances = gradcam_array[sample_indices]
        else:
            sampled_importances = gradcam_array

        # Normalize each batch to show relative importance
        normalized_importances = sampled_importances / sampled_importances.sum(axis=1, keepdims=True)

        # Plot heatmap
        sns.heatmap(normalized_importances.T, cmap='YlOrRd', cbar_kws={'label': 'Normalized Importance'},
                    yticklabels=graph_types, xticklabels=False, annot=False)
        plt.title(f'{classificationLevel} {modelArchitecture} - Graph Type Importance Across Batches', fontsize=14, fontweight='bold')
        plt.xlabel('Batch Index', fontsize=12)
        plt.ylabel('Graph Type', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{imageSavePath}/GradCAM_Heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create summary statistics plot
        plt.figure(figsize=(12, 6))

        # Box plot showing distribution of importance across batches
        box_data = [gradcam_array[:, i] for i in range(4)]
        box_plot = plt.boxplot(box_data, labels=graph_types, patch_artist=True)

        # Color the boxes
        for patch, color in zip(box_plot['boxes'], bar_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.title(f'{classificationLevel} {modelArchitecture} - GradCAM Importance Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Graph Type', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{imageSavePath}/GradCAM_Distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Per-class GradCAM visualizations

        # Create grouped bar chart for per-class comparison
        if len(per_class_percentages) > 0:
            fig, ax = plt.subplots(figsize=(14, 8))

            x = np.arange(len(graph_types))
            width = 0.08  # Width of each bar
            num_classes = len(per_class_percentages)

            # Create bars for each class
            for i, (jet_name, percentages) in enumerate(per_class_percentages.items()):
                offset = width * (i - num_classes/2)
                bars = ax.bar(x + offset, percentages, width, label=jet_name, alpha=0.8)

            ax.set_xlabel('Graph Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Relative Importance (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'{classificationLevel} {modelArchitecture} - Per-Class GradCAM Importance',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(graph_types)
            ax.legend(loc='upper right', ncol=2, fontsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            plt.tight_layout()
            plt.savefig(f'{imageSavePath}/GradCAM_Per_Class_Comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Create heatmap showing per-class importance percentages
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create matrix: rows=classes, cols=graph_types
            heatmap_data = []
            class_labels = []
            for jet_name, percentages in per_class_percentages.items():
                heatmap_data.append(percentages)
                class_labels.append(jet_name)

            heatmap_matrix = np.array(heatmap_data)

            sns.heatmap(heatmap_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                       xticklabels=graph_types, yticklabels=class_labels,
                       cbar_kws={'label': 'Importance (%)'}, vmin=0, vmax=100)

            plt.title(f'{classificationLevel} {modelArchitecture} - Per-Class GradCAM Heatmap',
                     fontsize=14, fontweight='bold')
            plt.xlabel('Graph Type', fontsize=12, fontweight='bold')
            plt.ylabel('Jet Class', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{imageSavePath}/GradCAM_Per_Class_Heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Create individual bar plots for each class (faceted view)
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()

            for idx, (jet_name, percentages) in enumerate(per_class_percentages.items()):
                if idx < len(axes):
                    ax = axes[idx]

                    # Color bars by importance
                    norm_percentages = percentages / percentages.max()
                    colors = [cmap(norm_p) for norm_p in norm_percentages]

                    bars = ax.bar(graph_types, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

                    # Add value labels
                    for bar, val in zip(bars, percentages):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

                    ax.set_title(jet_name, fontsize=11, fontweight='bold')
                    ax.set_ylabel('Importance (%)', fontsize=9)
                    ax.set_ylim(0, 100)
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    ax.tick_params(axis='x', rotation=45, labelsize=8)

            # Hide unused subplots
            for idx in range(len(per_class_percentages), len(axes)):
                axes[idx].axis('off')

            plt.suptitle(f'{classificationLevel} {modelArchitecture} - Per-Class GradCAM Breakdown',
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{imageSavePath}/GradCAM_Per_Class_Individual.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Save per-class results to CSV
            per_class_csv_data = []
            for jet_name in per_class_percentages.keys():
                row = {'Class': jet_name}
                for i, graph_type in enumerate(graph_types):
                    row[f'{graph_type}_Importance'] = per_class_importances[jet_name][i]
                    row[f'{graph_type}_Percentage'] = per_class_percentages[jet_name][i]
                per_class_csv_data.append(row)

            per_class_df = pd.DataFrame(per_class_csv_data)
            per_class_csv_file = f'{imageSavePath}/GradCAM_Per_Class_Results.csv'
            per_class_df.to_csv(per_class_csv_file, index=False)
            print(f"\nPer-class GradCAM results saved to: {per_class_csv_file}")

        # Log to wandb
        gradcam_table = wandb.Table(
            data=[[gt, imp, pct] for gt, imp, pct in zip(graph_types, avg_importances, importance_percentages)],
            columns=["Graph Type", "Importance Score", "Percentage"]
        )

        # Prepare wandb logging dict
        wandb_log_dict = {
            "GradCAM/Graph_Type_Importance": wandb.Image(f"{imageSavePath}/GradCAM_Graph_Type_Importance.png"),
            "GradCAM/Importance_Heatmap": wandb.Image(f"{imageSavePath}/GradCAM_Heatmap.png"),
            "GradCAM/Importance_Distribution": wandb.Image(f"{imageSavePath}/GradCAM_Distribution.png"),
            "GradCAM/Importance_Table": gradcam_table,
            "GradCAM/Delta_Importance": importance_percentages[0],
            "GradCAM/kT_Importance": importance_percentages[1],
            "GradCAM/mSquare_Importance": importance_percentages[2],
            "GradCAM/Z_Importance": importance_percentages[3],
            "GradCAM/Total_Samples_Analyzed": len(gradcam_importances)
        }

        # Add per-class visualizations to wandb if available
        if len(per_class_percentages) > 0:
            wandb_log_dict.update({
                "GradCAM/Per_Class_Comparison": wandb.Image(f"{imageSavePath}/GradCAM_Per_Class_Comparison.png"),
                "GradCAM/Per_Class_Heatmap": wandb.Image(f"{imageSavePath}/GradCAM_Per_Class_Heatmap.png"),
                "GradCAM/Per_Class_Individual": wandb.Image(f"{imageSavePath}/GradCAM_Per_Class_Individual.png")
            })

            # Create per-class table for wandb
            per_class_table_data = []
            for jet_name in per_class_percentages.keys():
                row = [jet_name]
                for i in range(4):
                    row.append(per_class_importances[jet_name][i])
                    row.append(per_class_percentages[jet_name][i])
                per_class_table_data.append(row)

            per_class_table = wandb.Table(
                data=per_class_table_data,
                columns=["Class", "Delta_Imp", "Delta_%", "kT_Imp", "kT_%", "mSquare_Imp", "mSquare_%", "Z_Imp", "Z_%"]
            )
            wandb_log_dict["GradCAM/Per_Class_Table"] = per_class_table

        wandb.log(wandb_log_dict)

        # Save summary to text file
        summary_file = f'{imageSavePath}/GradCAM_Summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"GradCAM analysis summary\n")
            f.write(f"Model: {classificationLevel} {modelArchitecture}\n\n")
            f.write(f"Total samples analyzed: {len(gradcam_importances)}\n")
            f.write(f"Total mini-batches: {len(gradcam_importances)}\n\n")

            f.write("Overall graph type importance:\n")
            for graph_type, importance, percentage in zip(graph_types, avg_importances, importance_percentages):
                f.write(f"{graph_type:12s}: {importance:.6f} ({percentage:.2f}%)\n")
            f.write("\n")

            # Add per-class results to summary
            if len(per_class_percentages) > 0:
                f.write("Per-class graph type importance:\n\n")
                for jet_name in per_class_percentages.keys():
                    f.write(f"{jet_name}:\n")
                    for i, graph_type in enumerate(graph_types):
                        importance = per_class_importances[jet_name][i]
                        percentage = per_class_percentages[jet_name][i]
                        f.write(f"  {graph_type:12s}: {importance:.6f} ({percentage:.2f}%)\n")
                    f.write("\n")

        print(f"GradCAM summary saved to: {summary_file}")

        # Clean up
        del gradcam_array, avg_importances, importance_percentages, sampled_importances, normalized_importances
        gc.collect()
    else:
        print("No GradCAM data available")
else:
    print(f"GradCAM stream file not found: {gradcam_stream_file}")

# Clean up gradcam importances list
del gradcam_importances
gc.collect()

# End of GradCAM analysis

# Log to wandb
wandb_log_dict = {
    "Final_Confusion_Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix.png"),
    "Final_ROC-AUC_Curve": wandb.Image(f"{imageSavePath}/ROC-AUC.png") if os.path.exists(f"{imageSavePath}/ROC-AUC.png") else None,
    "Final_PR-AUC_Curve": wandb.Image(f"{imageSavePath}/PR-AUC.png") if os.path.exists(f"{imageSavePath}/PR-AUC.png") else None,
    "Final_Confusion_Matrix_Table": wandb.Table(dataframe=metricsDF.reset_index()),
    "Final_Total_Samples_Processed": total_processed,
    "Final_Overall_Accuracy": metricsDF.loc['Micro Avg', 'Accuracy']
}

# Add AUC and AUPR metrics if available
if 'ROC-AUC' in metricsDF.columns:
    wandb_log_dict.update({
        "Final_Macro_ROC_AUC": metricsDF.loc['Macro Avg', 'ROC-AUC'],
        "Final_Micro_ROC_AUC": metricsDF.loc['Micro Avg', 'ROC-AUC'],
        "Final_Macro_PR_AUC": metricsDF.loc['Macro Avg', 'PR-AUC'],
        "Final_Micro_PR_AUC": metricsDF.loc['Micro Avg', 'PR-AUC']
    })

    # Add per-class AUC scores
    for i, jet_name in enumerate(jetNames):
        wandb_log_dict[f"ROC_AUC/{jet_name}"] = roc_auc.get(i, 0.0)
        wandb_log_dict[f"PR_AUC/{jet_name}"] = pr_auc.get(i, 0.0)

# Add probability statistics if available
if 'prob_stats' in locals() and len(prob_stats) > 0:
    wandb_log_dict.update({
        "Probability_Stats/Mean_Probability_Heatmap": wandb.Image(f"{imageSavePath}/Mean_Probability_Heatmap.png"),
        "Probability_Stats/Distribution_BoxPlots": wandb.Image(f"{imageSavePath}/Probability_Distribution_BoxPlots.png"),
        "Probability_Stats/Statistics_Table": wandb.Table(dataframe=prob_stats_df)
    })

    # Add confidence metrics
    if 'confidence_metrics' in locals() and len(confidence_metrics) > 0:
        for jet_name, metrics in confidence_metrics.items():
            wandb_log_dict[f"Confidence/{jet_name}/Mean_Correct_Prob"] = metrics['mean_correct_prob']
            wandb_log_dict[f"Confidence/{jet_name}/Mean_Max_Wrong_Prob"] = metrics['mean_max_wrong_prob']
            wandb_log_dict[f"Confidence/{jet_name}/Confidence_Gap"] = metrics['confidence_gap']

wandb.log(wandb_log_dict)

wandb.finish()

print("\nMulti-graph testing done.")
print(f"Results saved to: {imageSavePath}")
print(f"Metrics saved to: metrics/")
print(f"Model used: {modelSaveFile}")
print(f"Total samples processed: {total_processed}")
print(f"Final accuracy: {metricsDF.loc['Micro Avg', 'Accuracy']:.4f}")

print("Multi-graph analysis done.")