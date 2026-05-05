import numpy as np

import pandas as pd

from operator import truth
import pandas as pd
import numpy as np
import awkward as ak

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
import uproot
import torch

from tqdm import tqdm
import timeit
import os

import dill

import scipy.sparse as sp
import dgl
import pickle


# take ROOT file and convert to an awkward array
def fileToAwk(path):
    file = uproot.open(path)
    tree = file['tree']
    
    awk = tree.arrays(tree.keys())
    return awk

input_features = ["part_px", "part_py", "part_pz", "part_energy",
                  "part_deta", "part_dphi", "part_d0val", "part_d0err", 
                  "part_dzval", "part_dzerr", "part_isChargedHadron", "part_isNeutralHadron", 
                  "part_isPhoton", "part_isElectron", "part_isMuon" ] # features used to train the model

 
# Take AWK dict and convert to a point cloud
def awkToPointCloud(awkDict, input_features):
    featureVector = []
    for jet in tqdm(range(len(awkDict)), total=len(awkDict)):
        currJet = awkDict[jet][input_features]
        try:
            pT = np.sqrt(ak.to_numpy(currJet['part_px']) ** 2 + ak.to_numpy(currJet['part_py']) ** 2)
            # Create numpy array to represent the 4-momenta of all particles in a jet
            currJet = np.column_stack((
                ak.to_numpy(currJet['part_px']),
                ak.to_numpy(currJet['part_py']),
                ak.to_numpy(currJet['part_pz']),
                ak.to_numpy(currJet['part_energy']),
                pT,
                ak.to_numpy(currJet['part_deta']),
                ak.to_numpy(currJet['part_dphi']),
                ak.to_numpy(currJet["part_d0val"]),
                ak.to_numpy(currJet["part_d0err"]),
                ak.to_numpy(currJet["part_dzval"]),
                ak.to_numpy(currJet["part_dzerr"]),
                ak.to_numpy(currJet["part_isChargedHadron"]),
                ak.to_numpy(currJet["part_isNeutralHadron"]),
                ak.to_numpy(currJet["part_isPhoton"]),
                ak.to_numpy(currJet["part_isElectron"]),
                ak.to_numpy(currJet["part_isMuon"])
            ))
            featureVector.append(currJet)
        except Exception as e:
            print(f"Error processing jet {jet}: {e}")
            featureVector.append(np.empty((0, len(input_features) + 1)))  # Add an empty array for failed jets
    return featureVector  # Return a list of arrays instead of a single numpy array

from scipy.spatial import cKDTree

#take point cloud and build KNN graph
def buildKNNGraph(points, k):
    
    # Compute k-nearest neighbors
    tree = cKDTree(points)
    dists, indices = tree.query(points, k+1)  # +1 to exclude self
    
    # Build adjacency matrix
    num_points = len(points)
    adj_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in indices[i, 1:]:  # exclude self
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
    return adj_matrix

# take adjacency matrix and turn it into a DGL graph
def adjacencyToDGL(adj_matrix):
    adj_matrix = sp.coo_matrix(adj_matrix)
    g_dgl = dgl.from_scipy(adj_matrix)
        
    return g_dgl


# wrap the functionality of fileToAwk and awkToPointCloud in a function to return a point cloud numpy array
def fileToPointCloudArray(jetType, input_features):
    filepath = f'../data/JetClass/JetRoots/{jetType}/{jetType}_100K.root' # original root file
    savepath = f'../data/JetClass/PointClouds/{jetType}.npy' # save file

    awk = fileToAwk(filepath)
    nparr = awkToPointCloud(awk, input_features)
    
    return nparr

def fileToGraph(jetType, k=3, save=True):
    print(f'Starting processing on {jetType} jets')
    pointCloudArr = fileToPointCloudArray(jetType, input_features)
    
    saveFilePath = f'../data/Multi Level Jet Tagging/{jetType}.pkl'
    
    savedGraphs = []
    for idx, pointCloud in tqdm(enumerate(pointCloudArr), leave=False, total=len(pointCloudArr)):
        try:
            adj_matrix = buildKNNGraph(pointCloud, k)
            graph = adjacencyToDGL(adj_matrix)
            
            graph.ndata['feat'] = torch.tensor(pointCloud, dtype=torch.float32)
            
            savedGraphs.append(graph)
            
            del adj_matrix, graph
        except Exception as e:
            print(e)
            
    
    if save:
        with open(saveFilePath, 'wb') as f:
            pickle.dump(savedGraphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        del pointCloudArr
        
    print(f'Graphs for {jetType} done.')
        
    return savedGraphs

def groupToGraph(jetTypeList, groupName):
    allGraphs = []
    for jetType in jetTypeList:
        allGraphs += fileToGraph(jetType, save=False)
    
    saveFilePath = f'../data/Multi Level Jet Tagging/{groupName}.pkl' 
    return allGraphs

# process all jetTypes
Higgs = ['HToBB', 'HToCC', 'HToGG', 'HToWW2Q1L', 'HToWW4Q']
Vector = ['WToQQ', 'ZToQQ']
Top = ['TTBar', 'TTBarLep']
QCD = ['ZJetsToNuNu']
Emitter = ['Emitter-Vector', 'Emitter-Top', 'Emitter-Higgs', 'Emitter-QCD']
allJets = Higgs + Vector + Top + QCD

for jetType in allJets:
   fileToGraph(jetType)
