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

# Checkpoint management functions
def save_checkpoint(checkpoint_file, processed_files, results):
    """Save checkpoint with processed files and accumulated results."""
    checkpoint_data = {
        'processed_files': processed_files,
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'total_processed': results['total_processed'],
        'class_counts': results['class_counts']  # For metrics calculation
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

            # New format - load with backward compatibility for class_counts
            results = {
                'confusion_matrix': np.array(checkpoint_data['confusion_matrix']),
                'total_processed': checkpoint_data['total_processed'],
                'class_counts': checkpoint_data.get('class_counts', {})  # Default to empty dict if missing
            }
            print(f"Checkpoint loaded: {len(checkpoint_data['processed_files'])} batch sets already processed")
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
else:
    cfs = np.zeros((out_feats, out_feats))
    total_processed = 0
    class_counts = {jet_name: 0 for jet_name in jetNames}

# Get all batch file sets (need this to calculate files_processed_count)
all_batch_files = dataset.get_all_batch_files()
total_files = len(all_batch_files)
files_processed_count = len(processed_files)

# Create streaming output files for predictions (append mode)
predictions_stream_file = f'metrics/{classificationLevel}-{modelArchitecture}-MultiGraph-Predictions-Stream.txt'
os.makedirs('metrics', exist_ok=True)

# Initialize stream file (truncate if starting fresh, keep if resuming)
if files_processed_count == 0 and os.path.exists(predictions_stream_file):
    print(f"Removing old predictions stream file: {predictions_stream_file}")
    os.remove(predictions_stream_file)

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
            for mini_batch_graphs, mini_batch_labels in tqdm(batch_loader,
                                                           desc=f"Testing batch {file_idx + 1}",
                                                           leave=False):
                # Unpack graphs
                graph_delta, graph_kT, graph_mSquare, graph_Z = mini_batch_graphs

                # Move to device
                graph_delta = graph_delta.to(device)
                graph_kT = graph_kT.to(device)
                graph_mSquare = graph_mSquare.to(device)
                graph_Z = graph_Z.to(device)
                mini_batch_labels = mini_batch_labels.to(device)

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

                # Stream results to disk (append mode)
                with open(predictions_stream_file, 'a') as f:
                    for pred, target in zip(predictions.cpu().tolist(), mini_batch_labels.cpu().tolist()):
                        f.write(f"{pred},{target}\n")

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
                del mini_batch_graphs, mini_batch_labels, logits, predictions
                del predictions_cpu, labels_cpu
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
                'class_counts': class_counts
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

# ROC curve generation is skipped here: it requires logits, not just predictions
# ROC curve generation requires full logits, which are not stored here to save memory
# If needed, can be regenerated by re-running inference
print("\nNote: ROC curve generation skipped to save memory.")
print("To generate ROC curves, re-run inference with logits tracking enabled.")

# Create placeholder image
try:
    plt.figure(figsize=(8, 6))
    plt.title(f'{classificationLevel} {modelArchitecture} Multi-Graph Results')
    plt.text(0.5, 0.5, 'ROC curve generation skipped\n(logits not stored to save memory)\n\nSee confusion matrix for detailed metrics',
             horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.axis('off')
    plt.savefig(f'{imageSavePath}/ROC-AUC.png')
    plt.close()
except Exception as e:
    print(f"Error creating placeholder ROC image: {e}")

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

# Calculate micro and macro averages
microAvg = metricsDF.mean(axis=0)
macroAvg = metricsDF.mean(axis=0)

# Add micro and macro averages to the DataFrame
metricsDF.loc['Micro Avg'] = microAvg
metricsDF.loc['Macro Avg'] = macroAvg

# Print the final metrics table
print("\nFinal results")
print(metricsDF)

# Log to wandb
wandb.log({
    "Final_Confusion_Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix.png"),
    "Final_ROC-AUC_Curve": wandb.Image(f"{imageSavePath}/ROC-AUC.png"),
    "Final_Confusion_Matrix_Table": wandb.Table(dataframe=metricsDF.reset_index()),
    "Final_Total_Samples_Processed": total_processed,
    "Final_Overall_Accuracy": metricsDF.loc['Micro Avg', 'Accuracy']
})

wandb.finish()

print("\nMulti-graph testing done.")
print(f"Results saved to: {imageSavePath}")
print(f"Metrics saved to: metrics/")
print(f"Model used: {modelSaveFile}")
print(f"Total samples processed: {total_processed}")
print(f"Final accuracy: {metricsDF.loc['Micro Avg', 'Accuracy']:.4f}")

print("Multi-graph analysis done.")