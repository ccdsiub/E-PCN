import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import seaborn as sns
import os
import dgl
import pickle
import wandb
import matplotlib.pyplot as plt
import argparse
import math
import gc
import threading
import time

from dgllife.utils import RandomSplitter
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

batch_size = 256

parser = argparse.ArgumentParser(description='Dynamic Multi-Graph PCN Training')
parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs (default: 500)')
parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size (default: 512)')
parser.add_argument('--device', type=str, default='cuda',choices=['cuda', 'cpu'], help='Device to use (default: cuda)')
parser.add_argument('--classification_level', type=str, default='Dynamic_All_Interactions-1x1_Conv', help=' (Classification levle default: All)')
parser.add_argument('--model_architecture', type=str, default=f'PCN-{batch_size}-OneCycleLR-BN', help='Model architecture name (default: PCN)')
parser.add_argument('--model_type', type=str, default='DGCNN', help='Model type (default: DGCNN)')
parser.add_argument('--load_model', type=str, default='N', help='Load from save file (default: N)')
parser.add_argument('--convergence_threshold', type=float, default=0.0001, help='Convergence threshold (default: 0.0001)')

args = parser.parse_args()

# Use argparse values
maxEpochs = args.max_epochs
batchSize = args.batch_size
device = args.device
classificationLevel = args.classification_level
modelArchitecture = args.model_architecture
modelType = args.model_type
load = True if args.load_model == 'Y' else False
convergence_threshold = args.convergence_threshold


def log_gpu_memory(stage=""):
    """Simple memory logging - continuous monitoring is handled by ContinuousMemoryMonitor"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage} - GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    else:
        print(f"{stage} - GPU not available")
    
    # Also log CPU memory usage
    import psutil
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024**3

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    torch.cuda.empty_cache()
    gc.collect()
    # Force multiple garbage collection cycles
    for _ in range(3):
        gc.collect()
    
    # Log memory after cleanup
    log_gpu_memory("after_cleanup")

class ContinuousMemoryMonitor:
    """Continuous memory monitoring"""
    def __init__(self, interval=5):
        self.interval = interval  # seconds
        self.running = False
        self.monitor_thread = None
        self.start_time = None
        self.current_stage = "preprocessing"
        self.current_epoch = 0
    
    def set_stage(self, stage, epoch = 0):
        """Set the current processing stage"""
        self.current_stage = stage
        self.current_epoch = epoch

    def start(self):
        """Start continuous monitoring in background thread"""
        if self.running:
            return
            
        self.running = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"Continuous memory monitoring started (every {self.interval} seconds)")
        
    def stop(self):
        """Stop continuous monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("Continuous memory monitoring stopped")
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                self._log_memory()
                time.sleep(self.interval)
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                time.sleep(self.interval)
                
    def _log_memory(self):
        """Log current memory state"""
        # GPU Memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_cached = torch.cuda.memory_reserved() / 1024**3
        else:
            gpu_allocated = gpu_cached = 0
            
        # CPU Memory
        import psutil
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**3
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Log to wandb continuously
        wandb.log({
            "Memory_Continuous/GPU_allocated_GB": gpu_allocated,
            "Memory_Continuous/GPU_cached_GB": gpu_cached,
            "Memory_Continuous/CPU_RAM_GB": cpu_memory,
        })

def log_memory_trends():
    """Log current memory state."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_cached = torch.cuda.max_memory_reserved() / 1024**3
        
        wandb.log({
            "Memory_Trends/GPU_allocated_GB": allocated,
            "Memory_Trends/GPU_cached_GB": cached,
            "Memory_Trends/GPU_peak_allocated_GB": peak_allocated,
            "Memory_Trends/GPU_peak_cached_GB": peak_cached,
            "Memory_Trends/GPU_utilization_percent": (allocated / cached * 100) if cached > 0 else 0,
        })
    
    import psutil
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024**3
    
    wandb.log({
        "Memory_Trends/CPU_RAM_GB": cpu_memory,
        "Memory_Trends/CPU_peak_RAM_GB": process.memory_info().vms / 1024**3,  # Virtual memory size as peak
    })

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
    node_features_cpu = base_graph_cpu.ndata['feat'].detach().to('cpu').clone()
    new_graph_cpu.ndata['feat'] = node_features_cpu
    del node_features_cpu

    for key in base_graph_cpu.ndata.keys():
        if key != 'feat':
            temp_data_cpu = base_graph_cpu.ndata[key].detach().to('cpu').clone()
            new_graph_cpu.ndata[key] = temp_data_cpu
            del temp_data_cpu

    # Sanity checks
    assert new_graph_cpu.num_nodes() == base_graph_cpu.num_nodes(), "Node count mismatch!"
    assert new_graph_cpu.num_edges() == base_graph_cpu.num_edges(), "Edge count mismatch!"

    # Final GPU cleanup after this graph
    torch.cuda.empty_cache()

    return new_graph_cpu


class MultiGraphDataset(dgl.data.DGLDataset):
    def __init__(self, jetNames, k, loadFromDisk=False, device='cuda', use_gpu=True):
        self.jetNames = jetNames
        self.k = k
        self.device = device
        self.use_gpu = use_gpu

        # Initialize lists for all graph types
        self.delta = []
        self.mSquare = []
        self.kT = []
        self.Z = []
        self.sampleCountPerClass = []
        self.labels = []

        for jetType in tqdm(jetNames, total=len(jetNames), desc="Processing jet types"):
            if type(jetType) != list:
                if loadFromDisk:
                    base_path = f'pickleFiles/{jetType}.pkl'
                else:
                    base_path = f'data/{jetType}.pkl'
                
                print(f"Loading {jetType}...")
                with open(base_path, 'rb') as f:
                    base_graphs = pickle.load(f)

                # Process each graph after loading
                print(f"Processing {len(base_graphs)} graphs for {jetType}...")
                
                # Lists to store every graph for the current jet class
                jetType_delta = []
                jetType_kT = []
                jetType_Z = []
                jetType_mSquare = []

                for idx, base_graph in tqdm(enumerate(base_graphs),
                                            total=len(base_graphs),
                                            desc=f"Creating weighted graphs for {jetType}",
                                            leave=False):
                    # Create all 4 weighted graph types for this single base graph
                    graph_delta = create_weighted_graphs(base_graph, 'delta', self.device)
                    graph_kT = create_weighted_graphs(base_graph, 'kT', self.device)
                    graph_Z = create_weighted_graphs(base_graph, 'Z', self.device)
                    graph_mSquare = create_weighted_graphs(base_graph, 'mSquare', self.device)

                    # Graphs are already on CPU, just append them
                    jetType_delta.append(graph_delta)
                    jetType_kT.append(graph_kT)
                    jetType_Z.append(graph_Z)
                    jetType_mSquare.append(graph_mSquare)

                    # Clean up references - including the base_graph 
                    del graph_delta, graph_kT, graph_Z, graph_mSquare
                    del base_graph  # Delete base graph immediately after use
                    
                    # Periodic cleanup during processing (reduced frequency for better performance)
                    if idx % 100_000 == 0:
                        force_memory_cleanup()
                        log_gpu_memory(f"processing_{jetType}_{idx}")
                        
                print(f"Generated all weighted graphs for {jetType}")

                # Add ALL graphs from this jet class to main dataset (CPU only)
                self.delta.extend(jetType_delta)
                self.kT.extend(jetType_kT)
                self.Z.extend(jetType_Z)
                self.mSquare.extend(jetType_mSquare)
                class_count = len(base_graphs)
                self.sampleCountPerClass.append(class_count)
                # Create labels for this class immediately to avoid second pass
                current_label = len(self.sampleCountPerClass) - 1
                self.labels.extend([current_label] * class_count)

                print(f"Added {len(base_graphs)} graphs from {jetType} to dataset")
                
                # Clean up this jet class data
                del jetType_delta, jetType_kT, jetType_Z, jetType_mSquare
                del base_graphs

                # Full GPU and CPU cleanup before next jet class
                force_memory_cleanup()

                print(f"{jetType} done; GPU and CPU cleared.")

                print("-" * 50)         # Visual separator between jet classes
        
        for label, sampleCount in enumerate(self.sampleCountPerClass):
            print(f"Class {label} ({self.jetNames[label]}) has {sampleCount} samples")
        
        print("Dataset creation done.")
        print(f"Total samples: {len(self.labels)}")
        print(f"Samples per class: {self.sampleCountPerClass}")
        
        # Final cleanup
        force_memory_cleanup()
        
        # Extra aggressive cleanup at the end
        for _ in range(2):
            gc.collect()
        
        log_gpu_memory("After final cleanup")
        print("Final GPU and CPU cleanup done.")



    def process(self):
        return
    
    def __getitem__(self, idx):
        # Return CPU graphs - they'll be moved to GPU in collate function
        return {
            'graph_delta': self.delta[idx],
            'graph_kT': self.kT[idx],
            'graph_Z': self.Z[idx],
            'graph_mSquare': self.mSquare[idx],
            'label': self.labels[idx]
        }
    
    def __len__(self):
        return len(self.delta)

# collate function for multiple graphs
def collateFunction(batch):
    graphs_delta = [item['graph_delta'] for item in batch]
    graphs_kT = [item['graph_kT'] for item in batch]
    graphs_mSquare = [item['graph_mSquare'] for item in batch]
    graphs_Z = [item['graph_Z'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Batch on CPU first, then move to GPU
    batched_graph_delta = dgl.batch(graphs_delta).to(device)
    batched_graph_kT = dgl.batch(graphs_kT).to(device)
    batched_graph_mSquare = dgl.batch(graphs_mSquare).to(device)
    batched_graph_Z = dgl.batch(graphs_Z).to(device)

    # Ensure all node features AND edge weights are detached to prevent gradient tracking
    batched_graph_delta.ndata['feat'] = batched_graph_delta.ndata['feat'].detach()
    batched_graph_kT.ndata['feat'] = batched_graph_kT.ndata['feat'].detach()
    batched_graph_mSquare.ndata['feat'] = batched_graph_mSquare.ndata['feat'].detach()
    batched_graph_Z.ndata['feat'] = batched_graph_Z.ndata['feat'].detach()
    
    # Also detach edge weights to prevent gradient tracking
    if 'weight' in batched_graph_delta.edata:
        batched_graph_delta.edata['weight'] = batched_graph_delta.edata['weight'].detach()
    if 'weight' in batched_graph_kT.edata:
        batched_graph_kT.edata['weight'] = batched_graph_kT.edata['weight'].detach()
    if 'weight' in batched_graph_mSquare.edata:
        batched_graph_mSquare.edata['weight'] = batched_graph_mSquare.edata['weight'].detach()
    if 'weight' in batched_graph_Z.edata:
        batched_graph_Z.edata['weight'] = batched_graph_Z.edata['weight'].detach()

    del graphs_delta, graphs_kT, graphs_mSquare, graphs_Z
    
    return (batched_graph_delta, batched_graph_kT, batched_graph_mSquare, batched_graph_Z), torch.tensor(labels, device=device)

# GNN Feature Extractor
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


# Classifier class with 1D convolution
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


# Process all jetTypes
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']
Emitter = ['Emitter-Vector', 'Emitter-Top', 'Emitter-Higgs', 'Emitter-QCD']
allJets = Higgs + Vector + Top + QCD

testingSet = Top + Vector + QCD + Higgs
testingSet = [s + "-Testing" for s in testingSet]

jetNames = testingSet
print(jetNames)

# Generate unique model filename to avoid overwriting
os.makedirs("modelSaveFiles", exist_ok=True)
base_filename = classificationLevel + modelArchitecture
base_model_path = f"modelSaveFiles/{base_filename}.pt"

# Check if the base filename exists and increment version if needed
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

# Extract the versioned filename (without path and extension) for consistent naming
versioned_model_name = os.path.splitext(os.path.basename(modelSaveFile))[0]

# Start wandb logging with versioned name
wandb.init(
    project="All Interaction Features (On-the-fly)", 
    name=versioned_model_name,
    config={
        "epochs": maxEpochs,
        "batch_size": batchSize,
        "model": modelArchitecture,
        "model_type": modelType,
        "device": device,
        "convergence_threshold": convergence_threshold,
        "load_model": load,
        "scheduler": "OneCycleLR",
        "scheduler_max_lr": 3e-3,
        "scheduler_pct_start": 0.12,
        "scheduler_epochs": 100,
        "scheduler_div_factor": 25.0,
        "scheduler_final_div_factor": 1e4,
    }
)

# Define custom metrics to plot against time
wandb.define_metric("Time_Minutes")
wandb.define_metric("Training Loss", step_metric="Time_Minutes")
wandb.define_metric("Validation Loss", step_metric="Time_Minutes")
wandb.define_metric("Training Accuracy", step_metric="Time_Minutes")
wandb.define_metric("Validation Accuracy", step_metric="Time_Minutes")
wandb.define_metric("Gradient Norm", step_metric="Time_Minutes")
wandb.define_metric("Learning Rate", step_metric="Time_Minutes")

# Initialize and start continuous memory monitoring
memory_monitor = ContinuousMemoryMonitor(interval=20)  # Log every 20 seconds 
memory_monitor.start()

# Log initial memory state
log_gpu_memory("initial_state")
print("wandb logging initialized.")
print("Memory monitoring active.")

# Create dataset with k=3
print("Creating dataset...")
memory_monitor.set_stage("preprocessing")
dataset = MultiGraphDataset(jetNames, 3, loadFromDisk=False, device=device, use_gpu=True)
dataset.process()

log_gpu_memory()

if maxEpochs != 0:
    print("Creating data splits...")
    train, val, test = RandomSplitter().train_val_test_split(dataset, frac_train=0.8, frac_test=0.1, 
                                                         frac_val=0.1, random_state=42)
else:
    train = dataset

if maxEpochs != 0:
    trainLoader = DataLoader(train, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True, num_workers=0)
    validationLoader = DataLoader(val, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True, num_workers=0)
    testLoader = DataLoader(test, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True, num_workers=0)
else:
    testLoader = DataLoader(train, batch_size=batchSize, shuffle=True, collate_fn=collateFunction, drop_last=True)

in_feats = 16
hidden_feats = 64
out_feats = len(jetNames) # Number of output classes

# Update wandb config with model details
wandb.config.update({
    "in_feats": in_feats,
    "hidden_feats": hidden_feats,
    "out_feats": out_feats,
    "model_save_file": modelSaveFile,
})

chebFilterSize = 16

if modelType == "DGCNN":
    # Create 4 feature extractors for each graph type
    print("Creating models...")
    model_delta = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_kT = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_mSquare = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    model_Z = GNNFeatureExtractor(in_feats, hidden_feats, chebFilterSize)
    
    # Final classifier that takes stacked features and applies 1D convolution
    classifier = Classifier(hidden_feats * 4, hidden_feats, out_feats)  # input_dim not used in new architecture

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

# Watch only the classifier to reduce memory overhead
wandb.watch(classifier, log='gradients', log_freq=100)

# Define the loss function and optimizer for all models
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW([
    {'params': model_delta.parameters()},
    {'params': model_kT.parameters()},
    {'params': model_mSquare.parameters()},
    {'params': model_Z.parameters()},
    {'params': classifier.parameters()}
], lr=1e-3)

# Calculate steps per epoch for OneCycleLR
steps_per_epoch = len(trainLoader) if maxEpochs != 0 else 1

# OneCycleLR scheduler: short warmup (12 epochs) and a long cosine anneal
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=3e-3,                    # peak LR
    steps_per_epoch=steps_per_epoch,
    epochs=100,                     # cap; handles up to 100 epochs
    pct_start=0.12,                 # 12% warmup (12 epochs)
    anneal_strategy='cos',
    div_factor=25.0,                # start LR = max_lr / 25 = 1.2e-4
    final_div_factor=1e4,           # end LR = max_lr / 1e4 = 3e-7
    verbose=True
)

trainingLossTracker = []
trainingAccuracyTracker = []
validationLossTracker = []
validationAccuracyTracker = []

bestLoss = float('inf')
epochs_without_improvement = 0
epochsTillQuit = 10

def cleanup_tensors(*tensors):
    """Helper function to properly delete tensors and clear cache"""
    for tensor in tensors:
        if tensor is not None and hasattr(tensor, 'data'):
            try:
                # Only set data to None if it's a valid tensor
                if tensor.data is not None:
                    tensor.data = None
            except:
                pass  # Ignore any errors when setting data to None
        del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Log training start
if maxEpochs > 0:
    print("Starting training...")
    trainingStartTime = time.time()

    # Train the model
    for epoch in range(maxEpochs):
        epochStartTime = time.time()
        memory_monitor.set_stage("training", epoch + 1)
        runningLoss = 0
        totalCorrectPredictions = 0
        totalSamples = 0
        valTotalCorrectPredictions = 0
        valTotalSamples = 0
    
        # Set all models to training mode
        for model in all_models:
            model.train()
        
        for batchIndex, (graphs, labels) in tqdm(enumerate(trainLoader), total=len(trainLoader), leave=False):
            # Unpack graphs
            graph_delta, graph_kT, graph_mSquare, graph_Z = graphs
            labels = labels.to(device).long()

            # Clear gradients before forward pass
            optimizer.zero_grad()

            # Get embeddings from each graph type
            hg_delta = model_delta(graph_delta)
            hg_kT = model_kT(graph_kT)
            hg_mSquare = model_mSquare(graph_mSquare)
            hg_Z = model_Z(graph_Z)
            
            # Stack embeddings to create 4×64 matrix instead of 256-dimensional vector
            stacked_features = torch.stack([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)  # (batch_size, 4, 64)
            
            # Get final logits from classifier
            logits = classifier(stacked_features)
            
            # Calculate loss and do backpropagation
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Step the scheduler (OneCycleLR steps per batch)
            scheduler.step()
            
            # Update running loss
            runningLoss += loss.item()

            # Compute accuracy
            with torch.no_grad():
                predictions = logits.argmax(dim=1)
                batchCorrectPredictions = (predictions == labels).sum().item()
                batchTotalSamples = labels.numel()

            totalCorrectPredictions += batchCorrectPredictions
            totalSamples += batchTotalSamples

            # Clean up only the unpacked variables (no heavy cleanup during training cycle)
            del graphs

        # Compute epoch statistics
        epochLoss = runningLoss / len(trainLoader)
        trainingLossTracker.append(epochLoss)
    
        epochAccuracy = totalCorrectPredictions / totalSamples
        trainingAccuracyTracker.append(epochAccuracy)

        # End of training epoch: cleanup
        torch.cuda.empty_cache()
        gc.collect()
        log_gpu_memory(f"After training epoch {epoch+1}")

        memory_monitor.set_stage("validation", epoch + 1)
        # Validation
        for model in all_models:
            model.eval()
        validationLoss = 0.0

        with torch.no_grad():
            for val_batch_idx, (graphs, labels) in tqdm(enumerate(validationLoader), total=len(validationLoader), leave=False):                
                # Unpack graphs
                graph_delta, graph_kT, graph_mSquare, graph_Z = graphs
                labels = labels.to(device).long()

                # Get embeddings and logits
                hg_delta = model_delta(graph_delta)
                hg_kT = model_kT(graph_kT)
                hg_mSquare = model_mSquare(graph_mSquare)
                hg_Z = model_Z(graph_Z)
                
                stacked_features = torch.stack([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)  # (batch_size, 4, 64)
                logits = classifier(stacked_features)
                
                loss = criterion(logits, labels)
                validationLoss += loss.item()
                
                predictions = logits.argmax(dim=1)
                batchCorrectPredictions = (predictions == labels).sum().item()
                batchTotalSamples = labels.numel()
                
                valTotalCorrectPredictions += batchCorrectPredictions
                valTotalSamples += batchTotalSamples
                
                # Clean up only the unpacked variables (no heavy cleanup during validation cycle)
                del graphs, labels
                
        avgValidationLoss = validationLoss / len(validationLoader)
        validationLossTracker.append(avgValidationLoss)
    
        validationAccuracy = valTotalCorrectPredictions / valTotalSamples
        validationAccuracyTracker.append(validationAccuracy)

        # End of validation epoch: cleanup
        torch.cuda.empty_cache()
        gc.collect()
        log_gpu_memory(f"After validation epoch {epoch+1}")

        # Save only when validation loss improved beyond the threshold
        if avgValidationLoss < bestLoss - convergence_threshold:
            bestLoss = avgValidationLoss
            bestStateDict = {
                'model_delta': model_delta.state_dict(),
                'model_kT': model_kT.state_dict(),
                'model_mSquare': model_mSquare.state_dict(),
                'model_Z': model_Z.state_dict(),
                'classifier': classifier.state_dict()
            }
            # Save model checkpoint only when improved
            torch.save({
                'model_delta': model_delta.state_dict(),
                'model_kT': model_kT.state_dict(),
                'model_mSquare': model_mSquare.state_dict(),
                'model_Z': model_Z.state_dict(),
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, modelSaveFile)
            print(f'Saved Models to file {modelSaveFile} at epoch {epoch+1}')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Final cleanup for this epoch
        torch.cuda.empty_cache()
        gc.collect()

        # Log gradient norm
        grad_norm = 0
        for model in all_models:
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item()**2
        grad_norm = grad_norm ** 0.5
        
        # Print training and validation losses
        epochTime = time.time() - epochStartTime
        totalTime = time.time() - trainingStartTime
        epochTimeMinutes = epochTime / 60.0
        totalTimeMinutes = totalTime / 60.0
        totalTimeHours = totalTime / 3600.0
        print(f"Epoch {epoch + 1} - Training Loss={epochLoss:.4f} - Validation Loss={avgValidationLoss:.4f} - Training Accuracy={epochAccuracy:.4f} - Validation Accuracy={validationAccuracy:.4f} - Time={epochTimeMinutes:.2f}min - Total Time={totalTimeHours:.2f}h")
        
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": epochLoss,
            "Validation Loss": avgValidationLoss,
            "Training Accuracy": epochAccuracy,
            "Validation Accuracy": validationAccuracy,
            "Gradient Norm": grad_norm,
            "Learning Rate": optimizer.param_groups[0]['lr'],
            "Time_Minutes": totalTimeMinutes,
        })

        # Switch monitor stage back to training after validation
        memory_monitor.set_stage("training", epoch + 1)
        
        # Check convergence criteria
        if epochs_without_improvement >= epochsTillQuit:
            print(f'Convergence achieved at epoch {epoch + 1}. Stopping training.')
            break

        

if maxEpochs != 0:
    torch.save(bestStateDict, modelSaveFile)

# Create directory for saving plots using versioned name
# Create plots directory first, then model-specific subdirectory
plots_dir = 'plots'
model_plots_dir = f'{plots_dir}/{versioned_model_name}'
try:
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(model_plots_dir, exist_ok=True)
    imageSavePath = model_plots_dir
    print(f"Created plots directory structure: {imageSavePath}")
except Exception as e:
    print(f"Error creating directories: {e}")

if maxEpochs != 0:
    print("Creating training plots...")
    
    # Plot training loss
    plt.figure()
    plt.plot(range(len(trainingLossTracker)), trainingLossTracker)
    plt.title(f'{versioned_model_name} Training Loss Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f'{imageSavePath}/Training Loss.png')
    plt.close()

    # Plot training accuracy
    plt.figure()
    plt.plot(range(len(trainingAccuracyTracker)), trainingAccuracyTracker)
    plt.title(f'{versioned_model_name} Training Accuracy Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(f'{imageSavePath}/Training Accuracy.png')
    plt.close()

    # Plot validation loss
    plt.figure()
    plt.plot(range(len(validationLossTracker)), validationLossTracker)
    plt.title(f'{versioned_model_name} Validation Loss Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f'{imageSavePath}/Validation Loss.png')
    plt.close()

    # Plot validation accuracy
    plt.figure()
    plt.plot(range(len(validationAccuracyTracker)), validationAccuracyTracker)
    plt.title(f'{versioned_model_name} Validation Accuracy Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(f'{imageSavePath}/Validation Accuracy.png')
    plt.close()

# Testing and Evaluation
print("Starting testing and evaluation...")
memory_monitor.set_stage("testing")

logitsTracker = []
predictionsTracker = []
targetsTracker = []

cfs = np.zeros((out_feats, out_feats))

# Set all models to eval mode
for model in all_models:
    model.eval()

import sklearn

if maxEpochs != 0:
    with torch.no_grad():
        for batch_idx, (graphs, labels) in enumerate(tqdm(testLoader, total=len(testLoader), leave=False)):            
            # Unpack graphs - graphs is already a tuple of 4 graphs
            graph_delta, graph_kT, graph_mSquare, graph_Z = graphs
            labels = labels.to(device)

            # Get embeddings and logits
            hg_delta = model_delta(graph_delta)
            hg_kT = model_kT(graph_kT)
            hg_mSquare = model_mSquare(graph_mSquare)
            hg_Z = model_Z(graph_Z)
            
            stacked_features = torch.stack([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)  # (batch_size, 4, 64)
            logits = classifier(stacked_features)
            
            # Convert to numpy immediately to save memory
            logits_np = logits.detach().cpu().numpy()
            targets_np = labels.detach().cpu().numpy()
            logitsTracker.append(logits_np)
            targetsTracker.append(targets_np)

            predictions = logits.argmax(dim=1)
            predictions_np = predictions.detach().cpu().numpy()
            predictionsTracker.append(predictions_np)
            
            # Update confusion matrix
            for idx, pred in enumerate(predictions):
                cfs[pred.item()][labels[idx].item()] += 1

            # Clean up only the unpacked variables (no heavy cleanup during testing cycle)
            del graphs, labels, logits_np, targets_np, predictions_np
else:
    # Also set testing stage for maxEpochs == 0 case
    memory_monitor.set_stage("testing")
    with torch.no_grad():
        for batch_idx, (graphs, labels) in tqdm(enumerate(testLoader), total=len(testLoader), leave=False):
            # Unpack graphs
            graph_delta, graph_kT, graph_mSquare, graph_Z = graphs
            labels = labels.to(device)

            # Get embeddings and logits
            hg_delta = model_delta(graph_delta)
            hg_kT = model_kT(graph_kT)
            hg_mSquare = model_mSquare(graph_mSquare)
            hg_Z = model_Z(graph_Z)
            
            stacked_features = torch.stack([hg_delta, hg_kT, hg_mSquare, hg_Z], dim=1)  # (batch_size, 4, 64)
            logits = classifier(stacked_features)
            
            # Convert to lists immediately
            logitsTracker.extend(logits.detach().cpu().tolist())
            targetsTracker.extend(labels.detach().cpu().tolist())

            predictions = logits.argmax(dim=1)
            predictionsTracker.extend(predictions.detach().cpu().tolist())
            
            # Update confusion matrix
            for idx, pred in enumerate(predictions):
                cfs[pred.item()][labels[idx].item()] += 1
            
            # Clean up only the unpacked variables (no heavy cleanup during testing cycle)
            del graphs, labels
# End of testing phase: cleanup
torch.cuda.empty_cache()
gc.collect()
log_gpu_memory("After complete testing phase")

# Save metrics using versioned name
os.makedirs('metrics', exist_ok=True)
logitsTrackerFile = f'metrics/{versioned_model_name}-Logits.pkl'
targetsTrackerFile = f'metrics/{versioned_model_name}-Targets.pkl'
predictionsTrackerFile = f'metrics/{versioned_model_name}-Predictions.pkl'

with open(logitsTrackerFile, 'wb') as f:
    pickle.dump(logitsTracker, f)

with open(targetsTrackerFile, 'wb') as f:
    pickle.dump(targetsTracker, f)

with open(predictionsTrackerFile, 'wb') as f:
    pickle.dump(predictionsTracker, f)

# Clear large tracking lists after saving
del logitsTracker, predictionsTracker, targetsTracker
torch.cuda.empty_cache()
gc.collect()

# Force multiple garbage collection cycles to break reference cycles
for _ in range(3):
    gc.collect()

print("Creating evaluation plots...")

# Plot confusion matrix
fig = plt.gcf()
fig.set_size_inches(15, 15)

ax = sns.heatmap(cfs/np.sum(cfs), annot=True, cmap='Blues')
ax.set_title(f'{versioned_model_name} Confusion Matrix')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')

print(cfs/np.sum(cfs))
plt.savefig(f'{imageSavePath}/Confusion Matrix.png')
plt.close()

# Calculate metrics from the confusion matrix before it is deleted
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

# Print the metrics table
print(metricsDF)

# Clear confusion matrix to free memory
del cfs
gc.collect()

# ROC-AUC Curve
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt

# Load data back from files to avoid keeping large arrays in memory
with open(logitsTrackerFile, 'rb') as f:
    logitsTracker = pickle.load(f)

with open(targetsTrackerFile, 'rb') as f:
    targetsTracker = pickle.load(f)

if maxEpochs != 0:
    rocLogits = np.concatenate(logitsTracker, axis=0)
    rocTargets = np.concatenate(targetsTracker, axis=0)
else:
    rocLogits = np.array(logitsTracker)
    rocTargets = np.array(targetsTracker)

skplt.metrics.plot_roc_curve(rocTargets, rocLogits, figsize=(8, 6), title=f'{versioned_model_name} ROC-AUC Curve')
plt.savefig(f'{imageSavePath}/ROC-AUC.png')
plt.close()

# Clear ROC data after use
del rocLogits, rocTargets, logitsTracker, targetsTracker
torch.cuda.empty_cache()
gc.collect()

# Force cleanup to break any remaining reference cycles
for _ in range(2):
    gc.collect()
wandb.log({
    "Results/micro_avg_accuracy": microAvg['Accuracy'],
    "Results/micro_avg_precision": microAvg['Precision'],
    "Results/micro_avg_recall": microAvg['Recall'],
    "Results/micro_avg_specificity": microAvg['Specificity'],
    "Confusion Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix.png"),
    "ROC-AUC Curve": wandb.Image(f"{imageSavePath}/ROC-AUC.png"),
    "Confusion Matrix Table": wandb.Table(dataframe=metricsDF.reset_index())
})

wandb.save(modelSaveFile)
print("Training done.")
wandb.finish()

# Stop continuous memory monitoring
memory_monitor.stop()

# Final cleanup
del dataset
if 'train' in locals(): del train
if 'val' in locals(): del val
if 'test' in locals(): del test
if 'trainLoader' in locals(): del trainLoader
if 'validationLoader' in locals(): del validationLoader
if 'testLoader' in locals(): del testLoader

torch.cuda.empty_cache()
gc.collect()

# Final memory summary
log_gpu_memory("final_summary")

print("Memory cleanup done.")