import os
import json
import numpy as np
import pandas as pd
import uproot
import random
from tqdm import tqdm  

# Configuration

BASE_DIR = "data/JetClass_Pythia_100M"
OUTPUT_DIR = "output"

CLASSES = [
    "HToBB", "HToCC", "HToGG",
    "HToWW2Q1L", "HToWW4Q",
    "TTBar", "TTBarLep",
    "WToQQ", "ZJetsToNuNu", "ZToQQ"
]

JETS_PER_FILE = 1000
TOTAL_JETS = 100_000
RANDOM_SEED = 42

# Helper functions

def list_root_files(class_dir):
    return sorted([
        f for f in os.listdir(class_dir)
        if f.endswith(".root")
    ])

def sample_indices(total_entries, k):
    return sorted(np.random.choice(total_entries, size=k, replace=False))

def read_jets_from_file(file_path, indices):
    with uproot.open(file_path) as f:
        tree_name = f.keys()[0].split(";")[0]
        tree = f[tree_name]
        arrays = tree.arrays(library="np")
        sampled = {k: arrays[k][indices] for k in arrays.keys()}
        return sampled

def create_empty_data_structure(first_batch):
    return {k: [] for k in first_batch.keys()}

def append_batch(all_data, batch):
    for k in batch:
        all_data[k].append(batch[k])

def concatenate_all_batches(all_data):
    return {k: np.concatenate(v) for k, v in all_data.items()}

def save_root(data, output_root_path):
    with uproot.recreate(output_root_path) as f:
        f["tree"] = data

def save_tracking_files(indices_map, index_log, output_base):
    clean_indices_map = {k: [int(i) for i in v] for k, v in indices_map.items()}
    with open(output_base + "_indices.json", "w") as f:
        json.dump(clean_indices_map, f, indent=2)

    df = pd.DataFrame(index_log, columns=["New_Index", "Source_File", "Source_Index"])
    df.to_csv(output_base + "_indices.csv", index=False)

def save_preview_file(data, output_base, limit=100):
    preview = {k: v[:limit] for k, v in data.items()}
    df_preview = pd.DataFrame(preview)
    df_preview.to_csv(output_base + ".csv", index=False)
    df_preview.to_json(output_base + ".json", orient="records", indent=2)

# Main class processing

def process_class(class_name):
    print(f"\nProcessing class: {class_name}")

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    class_dir = os.path.join(BASE_DIR, class_name)
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    output_base = os.path.join(output_class_dir, f"{class_name}_100K")
    output_root_file = output_base + ".root"

    root_files = list_root_files(class_dir)

    total_collected = 0
    indices_map = {}
    index_log = []
    collected_batches = None

    for file_name in tqdm(root_files, desc=f"Sampling {class_name}", unit="file"):
        if total_collected >= TOTAL_JETS:
            break

        file_path = os.path.join(class_dir, file_name)
        with uproot.open(file_path) as f:
            tree_name = f.keys()[0].split(";")[0]
            tree = f[tree_name]
            total_jets_in_file = tree.num_entries

        if total_jets_in_file < JETS_PER_FILE:
            print(f"Skipping {file_name}: not enough jets.")
            continue

        selected_indices = sample_indices(total_jets_in_file, JETS_PER_FILE)
        jets_batch = read_jets_from_file(file_path, selected_indices)

        if collected_batches is None:
            collected_batches = create_empty_data_structure(jets_batch)

        append_batch(collected_batches, jets_batch)

        for i, idx in enumerate(selected_indices):
            index_log.append([total_collected + i, file_name, int(idx)])

        indices_map[file_name] = [int(x) for x in selected_indices]

        total_collected += JETS_PER_FILE

    final_data = concatenate_all_batches(collected_batches)

    save_root(final_data, output_root_file)
    save_tracking_files(indices_map, index_log, output_base + "_indices")
    save_preview_file(final_data, output_base)

    print(f"Done: {class_name}")

    return total_collected

# Main entry point

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print("Starting sampling process...")
    print(f"From each class, {TOTAL_JETS} jets will be selected.")
    print(f"Total jets after processing will be {TOTAL_JETS * len(CLASSES)}.")

    summary = {}

    for class_name in CLASSES:
        collected = process_class(class_name)
        summary[class_name] = collected

    print("\nFinal Report:")
    print("{:<15} {:<15}".format("Class", "Jets Collected"))
    print("-" * 30)
    for class_name, jets in summary.items():
        print("{:<15} {:<15}".format(class_name, jets))
    print("-" * 30)
    total_jets = sum(summary.values())
    print("{:<15} {:<15}".format("Total", total_jets))
