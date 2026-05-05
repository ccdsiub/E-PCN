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
import psutil  
import argparse  

# Memory monitoring function
def print_memory_usage(prefix="", device='cpu'):
    """Print current memory usage for CPU and GPU"""
    # CPU Memory
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**3 
    cpu_percent = process.memory_percent()

    print(f"{prefix} CPU Memory: {cpu_mem:.2f} GB ({cpu_percent:.1f}%)", end="")

    # GPU Memory
    if device == 'cuda' and torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f" | GPU Allocated: {gpu_mem_allocated:.2f} GB, Reserved: {gpu_mem_reserved:.2f} GB")
    else:
        print()

class BatchedGraphDataset(dgl.data.DGLDataset):
    def __init__(self, jetNames, k, batchDir='batches', loadFromDisk=False):
        
        self.jetNames = jetNames
        self.batchDir = batchDir
        
        # Get all batch files for each jet type
        self.batch_files = {}
        self.sampleCountPerClass = []
        
        for jetType in jetNames:
            if type(jetType) != list:
                # Find all batch files for this jet type
                jet_dir = os.path.join(batchDir, jetType)
                if os.path.exists(jet_dir):
                    batch_files = sorted([f for f in os.listdir(jet_dir) if f.endswith('.pkl')])
                    self.batch_files[jetType] = [os.path.join(jet_dir, f) for f in batch_files]
                else:
                    self.batch_files[jetType] = []
                
                # Count total samples for this jet type (quick count without loading)
                print(f'{jetType}: Found {len(self.batch_files[jetType])} batch files')
                # Exact count is computed during processing
                self.sampleCountPerClass.append(len(self.batch_files[jetType]))
            else:
                # Handle list of jet types (if needed)
                combined_files = []
                for item in jetType:
                    jet_dir = os.path.join(batchDir, item)
                    if os.path.exists(jet_dir):
                        batch_files = sorted([f for f in os.listdir(jet_dir) if f.endswith('.pkl')])
                        item_files = [os.path.join(jet_dir, f) for f in batch_files]
                        combined_files.extend(item_files)
                
                self.batch_files[str(jetType)] = combined_files
                self.sampleCountPerClass.append(len(combined_files))
        
        # Create a flat list of all batch files with their labels
        self.all_batch_files = []
        label = 0
        for jetType in jetNames:
            jet_key = jetType if type(jetType) != list else str(jetType)
            for batch_file in self.batch_files[jet_key]:
                self.all_batch_files.append((batch_file, label))
            label += 1
        
        print(f'Total batch files to process: {len(self.all_batch_files)}')

    def process(self):
        return
    
    def get_all_batch_files(self):
        """Get list of all batch files with their labels"""
        return self.all_batch_files
                
    def __getitem__(self, idx):
        # Not used: the batch processing path calls get_all_batch_files() instead
        raise NotImplementedError("Use get_all_batch_files() for memory-efficient processing")

    def __len__(self):
        return len(self.all_batch_files)

# Checkpoint management functions
def save_checkpoint(checkpoint_file, processed_files, results):
    """Save checkpoint with processed files and accumulated results"""
    checkpoint_data = {
        'processed_files': processed_files,
        'logitsTracker': results['logitsTracker'],
        'predictionsTracker': results['predictionsTracker'],
        'targetsTracker': results['targetsTracker'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'total_processed': results['total_processed']
    }

    # Save to temporary file first, then rename (atomic operation)
    temp_checkpoint = checkpoint_file + '.tmp'
    try:
        with open(temp_checkpoint, 'w') as f:
            json.dump(checkpoint_data, f)
        # Atomic rename
        os.replace(temp_checkpoint, checkpoint_file)
        print(f"Checkpoint saved: {len(processed_files)} files processed, {results['total_processed']} samples")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        if os.path.exists(temp_checkpoint):
            os.remove(temp_checkpoint)

def load_checkpoint(checkpoint_file):
    """Load checkpoint and return processed files and results"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            results = {
                'logitsTracker': checkpoint_data['logitsTracker'],
                'predictionsTracker': checkpoint_data['predictionsTracker'],
                'targetsTracker': checkpoint_data['targetsTracker'],
                'confusion_matrix': np.array(checkpoint_data['confusion_matrix']),
                'total_processed': checkpoint_data['total_processed']
            }
            print(f"Checkpoint loaded: {len(checkpoint_data['processed_files'])} files already processed")
            print(f"Total samples recovered: {results['total_processed']}")
            return checkpoint_data['processed_files'], results
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from beginning")
            return [], None
    else:
        print("No checkpoint found, starting from beginning")
        return [], None

# Function to calculate and display current metrics
def calculate_and_display_metrics(cfs, targetsTracker, predictionsTracker, jetNames, total_processed, file_count, total_files):
    """Calculate and display current metrics"""
    print(f"\nResults after {file_count}/{total_files} files ({total_processed} samples)")
    
    # Calculate overall accuracy
    if len(targetsTracker) > 0:
        overall_accuracy = sum(1 for t, p in zip(targetsTracker, predictionsTracker) if t == p) / len(targetsTracker)
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

            accuracy = (true_positive + true_negative) / np.sum(confusion_matrix)
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
    
    # Display confusion matrix (normalized)
    print("\nNormalized Confusion Matrix:")
    normalized_cfs = cfs / (np.sum(cfs) + 1e-8)  # Add small epsilon to avoid division by zero
    cfs_df = pd.DataFrame(normalized_cfs, index=jetNames, columns=jetNames)
    print(cfs_df.round(4))
    
    # Display class distribution
    class_counts = {}
    for i, jet_name in enumerate(jetNames):
        class_counts[jet_name] = targetsTracker.count(i)
    
    print("\nClass Distribution (samples processed so far):")
    for jet_name, count in class_counts.items():
        percentage = (count / len(targetsTracker)) * 100 if len(targetsTracker) > 0 else 0
        print(f"  {jet_name}: {count} samples ({percentage:.1f}%)")

    return metricsDF

# Function to compute all metrics safely (can be called at any point)
def compute_all_metrics(logitsTracker, targetsTracker, predictionsTracker, cfs, jetNames):
    """Compute all metrics including AUC, AUPR - safe to call even if data is incomplete"""
    from scipy.special import softmax
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import label_binarize

    metrics_dict = {}

    try:
        if len(logitsTracker) == 0 or len(targetsTracker) == 0:
            print("No data available for metrics computation")
            return None

        # Convert to numpy
        logitsTracker_np = np.array(logitsTracker)
        targetsTracker_np = np.array(targetsTracker)

        # Apply softmax to get probabilities
        probabilitiesTracker = softmax(logitsTracker_np, axis=1)

        # One-hot encode targets
        classes = sorted(list(set(targetsTracker_np)))
        rocTargets = label_binarize(targetsTracker_np, classes=classes)

        # Calculate AUC
        auc_scores_per_class = roc_auc_score(rocTargets, probabilitiesTracker, average=None)
        auc_micro = roc_auc_score(rocTargets, probabilitiesTracker, average='micro')
        auc_macro = roc_auc_score(rocTargets, probabilitiesTracker, average='macro')

        # Calculate AUPR
        aupr_scores_per_class = average_precision_score(rocTargets, probabilitiesTracker, average=None)
        aupr_micro = average_precision_score(rocTargets, probabilitiesTracker, average='micro')
        aupr_macro = average_precision_score(rocTargets, probabilitiesTracker, average='macro')

        metrics_dict = {
            'probabilitiesTracker': probabilitiesTracker,
            'rocTargets': rocTargets,
            'auc_scores_per_class': auc_scores_per_class,
            'auc_micro': auc_micro,
            'auc_macro': auc_macro,
            'aupr_scores_per_class': aupr_scores_per_class,
            'aupr_micro': aupr_micro,
            'aupr_macro': aupr_macro
        }

        return metrics_dict

    except Exception as e:
        print(f"Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to save intermediate results
def save_intermediate_results(imageSavePath, cfs, jetNames, classificationLevel, modelArchitecture,
                            logitsTracker, targetsTracker, predictionsTracker, file_count, total_files):
    """Save intermediate visualizations"""
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Save intermediate confusion matrix
    try:
        # Create new figure explicitly
        fig, ax = plt.subplots(figsize=(15, 15))

        sns.heatmap(cfs/np.sum(cfs), annot=True, cmap='Blues', ax=ax)
        ax.set_title(f'{classificationLevel} {modelArchitecture} Confusion Matrix (Files: {file_count}/{total_files})')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')

        plt.savefig(f'{imageSavePath}/Confusion Matrix_Intermediate_{file_count}.png')
        plt.close(fig)  # Explicitly close figure
        del fig, ax

        # Also save as the latest
        fig2, ax2 = plt.subplots(figsize=(15, 15))
        sns.heatmap(cfs/np.sum(cfs), annot=True, cmap='Blues', ax=ax2)
        ax2.set_title(f'{classificationLevel} {modelArchitecture} Confusion Matrix (Files: {file_count}/{total_files})')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        plt.savefig(f'{imageSavePath}/Confusion Matrix_Latest.png')
        plt.close(fig2)  # Explicitly close figure
        del fig2, ax2

    except Exception as e:
        print(f"Error saving intermediate confusion matrix: {e}")
        plt.close('all')  # Clean up any lingering figures

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='GNN Jet Classification Testing with Memory-Efficient Processing')

    # Model parameters
    parser.add_argument('--model-name', type=str, default='AllJetsPCN-8192.pt',
                        help='Model file name (default: AllJetsPCN-8192.pt)')
    parser.add_argument('--model-type', type=str, choices=['GCNN', 'DGCNN'], default='DGCNN',
                        help='Model architecture type (default: DGCNN)')
    parser.add_argument('--classification-level', type=str, default='AllJets',
                        help='Classification level (default: AllJets)')
    parser.add_argument('--model-architecture', type=str, default='PCN',
                        help='Model architecture name (default: PCN)')

    # Device and batch parameters
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference (default: 32)')

    # Data parameters
    parser.add_argument('--batch-dir', type=str, default='batches',
                        help='Directory containing batch files (default: batches)')
    parser.add_argument('--k', type=int, default=3,
                        help='Parameter k for dataset (default: 3)')

    # Model architecture parameters
    parser.add_argument('--in-feats', type=int, default=16,
                        help='Input features (default: 16)')
    parser.add_argument('--hidden-feats', type=int, default=64,
                        help='Hidden features (default: 64)')
    parser.add_argument('--cheb-filter-size', type=int, default=16,
                        help='Chebyshev filter size (default: 16)')

    # WandB parameters
    parser.add_argument('--wandb-project', type=str, default='PCN Testing 20M',
                        help='WandB project name (default: PCNTesting 20M)')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='WandB run name (default: auto-generated)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable WandB logging')

    # Other options
    parser.add_argument('--no-load', action='store_true',
                        help='Do not load model weights (start fresh)')
    parser.add_argument('--model-save-dir', type=str, default='modelSaveFiles',
                        help='Directory containing model files (default: modelSaveFiles)')

    return parser.parse_args()

# Parse arguments
args = parse_args()

# process all jetTypes
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']

# For testing, use the original jet names (remove "-Testing" suffix if needed)
testingSet = Top + Vector + QCD + Higgs
jetNames = testingSet
print(f"Jet Types: {jetNames}")

# Create dataset object
dataset = BatchedGraphDataset(jetNames, args.k, batchDir=args.batch_dir, loadFromDisk=False)
dataset.process()

# Testing path (maxEpochs = 0) uses the batched approach
maxEpochs = 0  # Set to 0 for testing
batchSize = args.batch_size

# Device and model configuration from args
device = args.device
classificationLevel = args.classification_level
modelArchitecture = args.model_architecture
modelType = args.model_type
modelSaveFile = os.path.join(args.model_save_dir, args.model_name)
load = not args.no_load  # Load by default unless --no-load is specified

print("\nConfiguration")
print(f"  Model File: {modelSaveFile}")
print(f"  Model Name: {args.model_name}")
print(f"  Model Type: {args.model_type}")
print(f"  Classification Level: {args.classification_level}")
print(f"  Model Architecture: {args.model_architecture}")
print(f"  Device: {args.device}")
print(f"  Batch Size: {batchSize}")
print(f"  Input Features: {args.in_feats}")
print(f"  Hidden Features: {args.hidden_feats}")
print(f"  Chebyshev Filter Size: {args.cheb_filter_size}")
print(f"  Load Model Weights: {load}")


import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.batch import batch

class GNNClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k):
        super(GNNClassifier, self).__init__()
        self.conv1 = dgl.nn.ChebConv(in_feats, hidden_feats, k)
        self.conv2 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.conv3 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        
        self.fc = nn.Linear(hidden_feats, out_feats)
        
    def forward(self, g):
        # Apply graph convolutional layers
        h = F.relu(self.conv1(g, g.ndata['feat']))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
    
        # Store the node embeddings in the node data dictionary
        g.ndata['h'] = h
    
        # Compute graph-level representations by taking global mean pooling
        hg = dgl.mean_nodes(g, 'h')
        
        # Pass the graph-level representation through a fully connected layer
        logits = self.fc(hg)
        
        return logits

class DGCNNClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k):
        super(DGCNNClassifier, self).__init__()
        self.conv1 = dgl.nn.ChebConv(in_feats, hidden_feats, k)
        self.conv2 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        self.conv3 = dgl.nn.ChebConv(hidden_feats, hidden_feats, k)
        
        self.edgeconv1 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        self.edgeconv2 = dgl.nn.EdgeConv(hidden_feats, hidden_feats)
        
        self.fc = nn.Linear(hidden_feats, out_feats)
        
    def forward(self, g):
        # Apply graph convolutional layers
        h = F.relu(self.conv1(g, g.ndata['feat']))
        h = F.relu(self.edgeconv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.edgeconv2(g, h))
        h = F.relu(self.conv3(g, h))
    
        # Store the node embeddings in the node data dictionary
        g.ndata['h'] = h
    
        # Compute graph-level representations by taking global mean pooling
        hg = dgl.mean_nodes(g, 'h')
        
        # Pass the graph-level representation through a fully connected layer
        logits = self.fc(hg)
        
        return logits

# Custom collate function for batching
def collateFunction(batch):
    graphs = [item['graph'] for item in batch]
    labels = [item['label'] for item in batch]
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

# Checkpoint file
checkpoint_file = f"checkpoints/{classificationLevel}-{modelArchitecture}-checkpoint.json"
os.makedirs("checkpoints", exist_ok=True)

# Model architecture parameters from args
in_feats = args.in_feats
hidden_feats = args.hidden_feats
out_feats = len(jetNames)  # Number of output classes
chebFilterSize = args.cheb_filter_size

# Start wandb logging
if not args.no_wandb:
    wandb_run_name = args.wandb_run_name if args.wandb_run_name else f"{classificationLevel}-{modelArchitecture}-Testing"
    wandb.init(
        project=args.wandb_project,
        name=wandb_run_name,
        config={
            "epochs": maxEpochs,
            "batch_size": batchSize,
            "model": modelArchitecture,
            "model_name": args.model_name,
            "in_feats": in_feats,
            "hidden_feats": hidden_feats,
            "out_feats": out_feats,
            "device": device,
            "testing_mode": True,
            "cheb_filter_size": chebFilterSize
        }
    )
    print("WandB logging enabled")
else:
    # Create a dummy wandb object if disabled
    class DummyWandB:
        def log(self, *args, **kwargs):
            pass
        def finish(self, *args, **kwargs):
            pass
        def Image(self, *args, **kwargs):
            return None
        def Table(self, *args, **kwargs):
            return None
    wandb = DummyWandB()
    print("WandB logging disabled")

# Initialize model
if modelType == "GCNN":
    model = GNNClassifier(in_feats, hidden_feats, out_feats, chebFilterSize)
elif modelType == "DGCNN":
    model = DGCNNClassifier(in_feats, hidden_feats, out_feats, chebFilterSize)
else:
    print("Invalid selection. Erroring out!")
    exit()

if load:
    model.load_state_dict(torch.load(modelSaveFile))
    print(f"Loaded model from {modelSaveFile}")

model.to(device)
model.eval()

# Load checkpoint if exists
processed_files, checkpoint_results = load_checkpoint(checkpoint_file)

# Initialize tracking variables for testing
if checkpoint_results:
    logitsTracker = checkpoint_results['logitsTracker']
    predictionsTracker = checkpoint_results['predictionsTracker']
    targetsTracker = checkpoint_results['targetsTracker']
    cfs = checkpoint_results['confusion_matrix']
    total_processed = checkpoint_results['total_processed']
else:
    logitsTracker = []
    predictionsTracker = []
    targetsTracker = []
    cfs = np.zeros((out_feats, out_feats))
    total_processed = 0

# Create results directory early
imageSavePath = f'{classificationLevel} {modelArchitecture}'
try:
    os.makedirs(imageSavePath, exist_ok=True)
except Exception as e:
    print(e)

print("Starting batch-wise testing...")

# Get all batch files
all_batch_files = dataset.get_all_batch_files()
total_files = len(all_batch_files)
files_processed_count = len(processed_files)

# Display initial status if resuming
if files_processed_count > 0:
    print(f"\nResuming from checkpoint. Already processed {files_processed_count}/{total_files} files.")
    metricsDF = calculate_and_display_metrics(cfs, targetsTracker, predictionsTracker, jetNames,
                                            total_processed, files_processed_count, total_files)

# Process each batch file one at a time
# Wrap in try-except for crash recovery
try:
    with torch.no_grad():
        for file_idx, (batch_file, label) in enumerate(tqdm(all_batch_files, desc="Processing batch files")):

            # Skip if already processed
            if batch_file in processed_files:
                continue

            try:
                print(f"\nLoading batch file: {batch_file}")
                print_memory_usage(prefix="[Before Load]", device=device)

                # Load one batch file at a time
                with open(batch_file, 'rb') as f:
                    batch_graphs = pickle.load(f)

                print(f"Loaded {len(batch_graphs)} graphs from {batch_file}")
                print_memory_usage(prefix="[After Load]", device=device)

                # Create labels for this batch
                batch_labels = [label] * len(batch_graphs)

                # Create dataset for this batch
                batch_data = []
                for graph, lbl in zip(batch_graphs, batch_labels):
                    batch_data.append({'graph': graph, 'label': lbl})

                # Create DataLoader for this batch with specified batch size
                batch_loader = DataLoader(batch_data, batch_size=batchSize, shuffle=False,
                                        collate_fn=collateFunction, drop_last=False)

                # Process this batch file in mini-batches
                for mini_batch_graphs, mini_batch_labels in tqdm(batch_loader,
                                                               desc=f"Processing {os.path.basename(batch_file)}",
                                                               leave=False):
                    mini_batch_graphs = mini_batch_graphs.to(device)
                    mini_batch_labels = mini_batch_labels.to(device)

                    # Make predictions
                    logits = model(mini_batch_graphs)
                    predictions = logits.argmax(dim=1)

                    # Move to CPU immediately and convert to Python types
                    logits_cpu = logits.detach().cpu()
                    predictions_cpu = predictions.detach().cpu()
                    labels_cpu = mini_batch_labels.detach().cpu()

                    # Store results (convert to Python lists to free torch tensors)
                    logitsTracker.extend(logits_cpu.tolist())
                    predictionsTracker.extend(predictions_cpu.tolist())
                    targetsTracker.extend(labels_cpu.tolist())

                    # Update confusion matrix (use CPU tensors)
                    for idx, pred in enumerate(predictions_cpu):
                        cfs[pred][labels_cpu[idx]] += 1

                    # Clean up GPU memory immediately
                    del mini_batch_graphs, mini_batch_labels, logits, predictions
                    del logits_cpu, predictions_cpu, labels_cpu

                    # Clear GPU cache periodically (not every iteration for performance)
                    if device == 'cuda':
                        torch.cuda.empty_cache()

                # Update counters
                total_processed += len(batch_graphs)
                processed_files.append(batch_file)
                files_processed_count += 1

                print(f"Completed {batch_file}. Total processed: {total_processed}")
                print_memory_usage(prefix="[After Processing]", device=device)

                # Calculate and display current metrics after each file
                metricsDF = calculate_and_display_metrics(cfs, targetsTracker, predictionsTracker, jetNames,
                                                        total_processed, files_processed_count, total_files)

                # Save intermediate results (every 5 files or last file)
                if files_processed_count % 5 == 0 or files_processed_count == total_files:
                    save_intermediate_results(imageSavePath, cfs, jetNames, classificationLevel, modelArchitecture,
                                            logitsTracker, targetsTracker, predictionsTracker,
                                            files_processed_count, total_files)

                    # Log intermediate results to wandb
                    wandb.log({
                        "Current_Overall_Accuracy": metricsDF.loc['Micro Avg', 'Accuracy'],
                        "Current_Files_Processed": files_processed_count,
                        "Current_Samples_Processed": total_processed,
                        "Progress_Percentage": (files_processed_count / total_files) * 100,
                        "Current_Confusion_Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix_Latest.png") if os.path.exists(f"{imageSavePath}/Confusion Matrix_Latest.png") else None
                    })

                # Clean up memory thoroughly
                del batch_graphs, batch_labels, batch_data, batch_loader

                # Force garbage collection
                gc.collect()

                # Additional GPU cleanup if using CUDA
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                print_memory_usage(prefix="[After Cleanup]", device=device)

                # Save checkpoint after each file
                results = {
                    'logitsTracker': logitsTracker,
                    'predictionsTracker': predictionsTracker,
                    'targetsTracker': targetsTracker,
                    'confusion_matrix': cfs,
                    'total_processed': total_processed
                }
                save_checkpoint(checkpoint_file, processed_files, results)

            except Exception as e:
                print(f"Error processing {batch_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

except KeyboardInterrupt:
    print("\nInterrupted by user.")
    print(f"Progress saved in checkpoint: {checkpoint_file}")
    print(f"Processed {files_processed_count}/{total_files} files ({total_processed} samples)")
    print("Re-running the script will resume from the checkpoint.")
    # Save final checkpoint before exiting
    results = {
        'logitsTracker': logitsTracker,
        'predictionsTracker': predictionsTracker,
        'targetsTracker': targetsTracker,
        'confusion_matrix': cfs,
        'total_processed': total_processed
    }
    save_checkpoint(checkpoint_file, processed_files, results)
    raise

except Exception as e:
    print(f"\nError during processing: {e}")
    import traceback
    traceback.print_exc()
    print(f"\nProgress saved in checkpoint: {checkpoint_file}")
    print(f"Processed {files_processed_count}/{total_files} files ({total_processed} samples)")
    print("Re-running the script will resume from the checkpoint.")
    # Save final checkpoint before exiting
    results = {
        'logitsTracker': logitsTracker,
        'predictionsTracker': predictionsTracker,
        'targetsTracker': targetsTracker,
        'confusion_matrix': cfs,
        'total_processed': total_processed
    }
    save_checkpoint(checkpoint_file, processed_files, results)
    raise

print("\nTesting done.")
print(f"Total files processed: {files_processed_count}/{total_files}")
print(f"Total samples processed: {total_processed}")

# Save results
try:
    os.makedirs(imageSavePath, exist_ok=True)
except Exception as e:
    print(e)

# Save tracking data with error handling
logitsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-Logits.pkl'
targetsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-Targets.pkl'
predictionsTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-Predictions.pkl'

os.makedirs('metrics', exist_ok=True)

print("Saving final results to disk...")
try:
    with open(logitsTrackerFile, 'wb') as f:
        pickle.dump(logitsTracker, f)
    print(f"Logits saved: {logitsTrackerFile}")

    with open(targetsTrackerFile, 'wb') as f:
        pickle.dump(targetsTracker, f)
    print(f"Targets saved: {targetsTrackerFile}")

    with open(predictionsTrackerFile, 'wb') as f:
        pickle.dump(predictionsTracker, f)
    print(f"Predictions saved: {predictionsTrackerFile}")

    print("All results saved.")

    # Only clean up checkpoint after successful save
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file cleaned up")

except Exception as e:
    print(f"ERROR saving results: {e}")
    print("Checkpoint file retained for recovery")
    import traceback
    traceback.print_exc()

# Generate confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("\nGenerating visualizations and metrics...")

try:
    fig, ax = plt.subplots(figsize=(15, 15))

    sns.heatmap(cfs/np.sum(cfs), annot=True, cmap='Blues', ax=ax)
    ax.set_title(f'{classificationLevel} {modelArchitecture} Confusion Matrix')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')

    print(cfs/np.sum(cfs))
    plt.savefig(f'{imageSavePath}/Confusion Matrix.png')
    plt.close(fig)  # Explicitly close to free memory
    del fig, ax
    print("Confusion matrix visualization saved.")
except Exception as e:
    print(f"Error generating confusion matrix visualization: {e}")
    import traceback
    traceback.print_exc()
    plt.close('all')  # Clean up on error

# Generate ROC curve and calculate AUC, AUPR
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, precision_recall_curve
import scikitplot as skplt
from sklearn.preprocessing import label_binarize
from scipy.special import softmax

try:
    # Convert lists to numpy arrays
    logitsTracker_np = np.array(logitsTracker)
    targetsTracker_np = np.array(targetsTracker)

    # Apply softmax to convert logits to probabilities
    probabilitiesTracker = softmax(logitsTracker_np, axis=1)

    # Get unique classes
    classes = sorted(list(set(targetsTracker_np)))
    # Create one-hot encoding for targets
    rocTargets = label_binarize(targetsTracker_np, classes=classes)
    # Use probabilities for AUC/AUPR calculations
    rocLogits = probabilitiesTracker

    # Calculate AUC scores (class-wise, micro, macro)
    auc_scores_per_class = roc_auc_score(rocTargets, rocLogits, average=None)
    auc_micro = roc_auc_score(rocTargets, rocLogits, average='micro')
    auc_macro = roc_auc_score(rocTargets, rocLogits, average='macro')

    # Calculate AUPR scores (class-wise, micro, macro)
    aupr_scores_per_class = average_precision_score(rocTargets, rocLogits, average=None)
    aupr_micro = average_precision_score(rocTargets, rocLogits, average='micro')
    aupr_macro = average_precision_score(rocTargets, rocLogits, average='macro')

    # Print AUC and AUPR results
    print("\nROC-AUC scores:")
    for i, jet_name in enumerate(jetNames):
        print(f"{jet_name}: {auc_scores_per_class[i]:.4f}")
    print(f"Micro avg: {auc_micro:.4f}")
    print(f"Macro avg: {auc_macro:.4f}")

    print("\nAUPR scores:")
    for i, jet_name in enumerate(jetNames):
        print(f"{jet_name}: {aupr_scores_per_class[i]:.4f}")
    print(f"Micro avg: {aupr_micro:.4f}")
    print(f"Macro avg: {aupr_macro:.4f}")

    # Save class-wise probabilities
    probabilitiesTrackerFile = f'metrics/{classificationLevel}-{modelArchitecture}-Probabilities.pkl'
    with open(probabilitiesTrackerFile, 'wb') as f:
        pickle.dump(probabilitiesTracker.copy(), f)  # Use copy to ensure data integrity
    print(f"Class-wise probabilities saved to {probabilitiesTrackerFile}")

    # Create AUC/AUPR DataFrames for logging
    auc_aupr_data = {
        'Jet Type': jetNames + ['Micro Avg', 'Macro Avg'],
        'AUC': list(auc_scores_per_class) + [auc_micro, auc_macro],
        'AUPR': list(aupr_scores_per_class) + [aupr_micro, aupr_macro]
    }
    auc_aupr_df = pd.DataFrame(auc_aupr_data)

    # Plot ROC curve
    roc_fig = skplt.metrics.plot_roc_curve(rocTargets, rocLogits, figsize=(10, 8),
                               title=f'{classificationLevel} {modelArchitecture} ROC-AUC Curve\nMacro AUC: {auc_macro:.4f}, Micro AUC: {auc_micro:.4f}')
    plt.savefig(f'{imageSavePath}/ROC-AUC.png', dpi=300, bbox_inches='tight')
    plt.close(roc_fig.figure)  # Explicitly close the figure
    del roc_fig

    # Plot Precision-Recall curve
    pr_fig, pr_ax = plt.subplots(figsize=(10, 8))
    for i, jet_name in enumerate(jetNames):
        precision, recall, _ = precision_recall_curve(rocTargets[:, i], rocLogits[:, i])
        pr_ax.plot(recall, precision, label=f'{jet_name} (AUPR={aupr_scores_per_class[i]:.3f})')

    pr_ax.set_xlabel('Recall')
    pr_ax.set_ylabel('Precision')
    pr_ax.set_title(f'{classificationLevel} {modelArchitecture} Precision-Recall Curve\nMacro AUPR: {aupr_macro:.4f}, Micro AUPR: {aupr_micro:.4f}')
    pr_ax.legend(loc='best', fontsize=8)
    pr_ax.grid(True, alpha=0.3)
    plt.savefig(f'{imageSavePath}/Precision-Recall.png', dpi=300, bbox_inches='tight')
    plt.close(pr_fig)  # Explicitly close the figure
    del pr_fig, pr_ax

    # Clean up large arrays after plotting
    del rocTargets, rocLogits, probabilitiesTracker

except Exception as e:
    print(f"Error generating ROC/PR curves or calculating AUC/AUPR: {e}")
    import traceback
    traceback.print_exc()

    # Create error figure
    err_fig, err_ax = plt.subplots(figsize=(8, 6))
    err_ax.set_title(f'{classificationLevel} {modelArchitecture} ROC-AUC Curve (Error)')
    err_ax.text(0.5, 0.5, f'Error generating curves: {str(e)}',
             horizontalalignment='center', verticalalignment='center')
    plt.savefig(f'{imageSavePath}/ROC-AUC.png')
    plt.close(err_fig)  # Close error figure
    del err_fig, err_ax

    # Set default values if error occurs
    auc_macro = 0.0
    aupr_macro = 0.0
    auc_aupr_df = None

finally:
    # Always clean up matplotlib memory
    plt.close('all')
    gc.collect()

# Calculate metrics
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

# Calculate Macro Accuracy (average of per-class accuracies)
macro_accuracy = metricsDF['Accuracy'].mean()

# Add micro and macro averages to the DataFrame
metricsDF.loc['Micro Avg'] = microAvg
metricsDF.loc['Macro Avg'] = macroAvg

# Print the metrics table
print("\nFinal classification metrics")
print(metricsDF)
print(f"\nMacro accuracy (average of per-class accuracies): {macro_accuracy:.4f} ({macro_accuracy*100:.2f}%)\n")

# Log to wandb
wandb_log_dict = {
    "Confusion Matrix": wandb.Image(f"{imageSavePath}/Confusion Matrix.png"),
    "ROC-AUC Curve": wandb.Image(f"{imageSavePath}/ROC-AUC.png"),
    "Confusion Matrix Table": wandb.Table(dataframe=metricsDF.reset_index()),
    "Total Samples Processed": total_processed,
    "Macro Accuracy": macro_accuracy,
    "Micro Avg Accuracy": microAvg['Accuracy'],
    "Macro Avg Precision": macroAvg['Precision'],
    "Macro Avg Recall": macroAvg['Recall'],
    "Macro Avg Specificity": macroAvg['Specificity']
}

# Add AUC/AUPR metrics if available
if auc_aupr_df is not None:
    wandb_log_dict.update({
        "Precision-Recall Curve": wandb.Image(f"{imageSavePath}/Precision-Recall.png"),
        "AUC-AUPR Table": wandb.Table(dataframe=auc_aupr_df),
        "Macro AUC": auc_macro,
        "Micro AUC": auc_micro,
        "Macro AUPR": aupr_macro,
        "Micro AUPR": aupr_micro
    })

    # Log per-class AUC and AUPR
    for i, jet_name in enumerate(jetNames):
        wandb_log_dict[f"AUC_{jet_name}"] = auc_scores_per_class[i]
        wandb_log_dict[f"AUPR_{jet_name}"] = aupr_scores_per_class[i]

wandb.log(wandb_log_dict)

# Create and save a comprehensive summary table
summary_data = {
    'Metric': ['Macro Accuracy', 'Micro Avg Accuracy', 'Macro Avg Precision',
               'Macro Avg Recall', 'Macro Avg Specificity'],
    'Value': [macro_accuracy, microAvg['Accuracy'], macroAvg['Precision'],
              macroAvg['Recall'], macroAvg['Specificity']]
}

if auc_aupr_df is not None:
    summary_data['Metric'].extend(['Macro AUC', 'Micro AUC', 'Macro AUPR', 'Micro AUPR'])
    summary_data['Value'].extend([auc_macro, auc_micro, aupr_macro, aupr_micro])

summary_df = pd.DataFrame(summary_data)
print("\nSummary of key metrics")
print(summary_df.to_string(index=False))


# Save summary to CSV
summary_csv_path = f'{imageSavePath}/summary_metrics.csv'
summary_df.to_csv(summary_csv_path, index=False)
print(f"Summary metrics saved to {summary_csv_path}")

wandb.finish()

# Final memory cleanup
print("\nFinal memory cleanup")

print_memory_usage(prefix="[Before Final Cleanup]", device=device)

# Clean up large tracking lists (already saved to disk)
del logitsTracker, predictionsTracker, targetsTracker
del cfs

# Clean up dataframes
try:
    del metricsDF, summary_df, auc_aupr_df
except:
    pass

# Force garbage collection
gc.collect()

# GPU cleanup if applicable
if device == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print_memory_usage(prefix="[After Final Cleanup]", device=device)

print("Analysis done.")