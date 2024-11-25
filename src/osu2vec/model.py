import math

from src.osu2vec import parser
from src.osu2vec import data

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
import torch.nn.functional as tf

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class GMM(nn.Module):
    def __init__(self, num_components, num_features):
        super(GMM, self).__init__()
        self.num_components = num_components
        self.num_features = num_features
        self.means = nn.Parameter(torch.randn(num_components, num_features))
        self.log_vars = nn.Parameter(torch.randn(num_components, num_features))
        self.weights = nn.Parameter(torch.randn(num_components))
    
    def forward(self, x):
        x = x.unsqueeze(1).expand(-1, self.num_components, -1) # shape: (batch_size, num_components, num_features)
        means = self.means.unsqueeze(0).expand(x.shape[0], -1, -1)
        log_vars = self.log_vars.unsqueeze(0).expand(x.shape[0], -1, -1)
        weights = self.weights.unsqueeze(0).expand(x.shape[0], -1)
        x = -0.5 * (log_vars + (x - means) ** 2 / (2 * torch.exp(log_vars)))
        x = torch.sum(x, dim=2) + torch.log(weights)
        return x
    
    def sample(self, num_samples):
        samples = torch.randn(num_samples, self.num_features)
        samples = samples.unsqueeze(1).expand(-1, self.num_components, -1)
        means = self.means.unsqueeze(0).expand(num_samples, -1, -1)
        log_vars = self.log_vars.unsqueeze(0).expand(num_samples, -1, -1)
        weights = self.weights.unsqueeze(0).expand(num_samples, -1)
        samples = -0.5 * (log_vars + (samples - means) ** 2 / (2 * torch.exp(log_vars)))
        samples = torch.sum(samples, dim=2) + torch.log(weights)
        samples = torch.argmax(samples, dim=1)
        return samples
    
class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        
        N, D, F = x.shape

        cosine_similarities = torch.zeros(F, dtype=torch.float32)

        for i in range(F):
            feature_vectors = x[:, :, i]
            similarity_matrix = tf.cosine_similarity(feature_vectors.unsqueeze(1), feature_vectors.unsqueeze(0), dim=2)

            upper_triangle_indices = torch.triu_in  dices(N, N, offset=1)
            upper_triangle_similarities = similarity_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]
            cosine_similarities[i] = upper_triangle_similarities.mean()
        
        normalized_similarities = cosine_similarities**2
        log_likelihoods = torch.log(normalized_similarities)/2
        
        
        return log_likelihoods

def loss(x, y):
    criterion = nn.MSELoss()
    return criterion(x, y)

class Osu2Vec(nn.Module):
    def __init__(self, hashing_size: int=512, embedding_size: int = 512, vector_size: int=256, num_layers=6, num_heads=8, feature_size=8, hidden_sizes=[512, 256, 128, 64, 128]):
        super(Osu2Vec, self).__init__()
        self.hashing_size = hashing_size
        self.embedding_size = embedding_size
        self.vector_size = vector_size

        self.hashed_data = np.array([])
        self.binned_data = np.array([])

        self.beatmap_log_likelihoods = np.array([])

        self.positional_encoding = PositionalEncoding(self.embedding_size)
        
        self.transformer_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=num_heads),
                num_layers=num_layers
            ) for _ in range(feature_size)
        ])

        self.relu = nn.ReLU()
        self.linear_layers = nn.ModuleList([
            nn.Linear(self.embedding_size * feature_size, hidden_sizes[0]) if i == 0 else nn.Linear(hidden_sizes[i-1], hidden_sizes[i]) for i in range(len(hidden_sizes))
        ])
        self.last_layer = nn.Linear(hidden_sizes[-1], self.vector_size)

        self.GMM = GMM(8, self.vector_size)

        

    def load(self, beatmaps: list[parser.Beatmap]):
        print("Loading beatmaps...")
        output_hashed = list(self.hashed_data)
        output_binned = list(self.binned_data)
        for beatmap in tqdm(beatmaps):
            output_hashed.append(beatmap.hashed_data)
            output_binned.append(beatmap.binned_data)
        
        self.hashed_data = np.array(output_hashed)
        self.binned_data = np.array(output_binned)

    def load_list(self, beatmaps: list[str], hashing_size=512, binning_size=512, correction=True):
        print("Loading beatmaps into model...")
        output_hashed = list(self.hashed_data)
        output_binned = list(self.binned_data)
        for beatmap in tqdm(beatmaps):
            beatmap = parser.Beatmap(beatmap, hashing_size=hashing_size, binning_size=binning_size, correction=correction)
            output_hashed.append(beatmap.hashed_data)
            output_binned.append(beatmap.binned_data)
        
        self.hashed_data = np.array(output_hashed)
        self.binned_data = np.array(output_binned)

    def load_hashed(self, beatmaps: list[parser.Beatmap]):
        print("Loading beatmaps...")
        output = list(self.hashed_data)
        for beatmap in tqdm(beatmaps):
            output.append(beatmap.hashed_data)
        
        self.hashed_data = np.array(output)

    def load_list_hashed(self, beatmaps: list[str], hashing_size=512, correction=True):
        print("Loading beatmaps into model...")
        output = list(self.hashed_data)
        for beatmap in tqdm(beatmaps):
            output.append(parser.Beatmap(beatmap, hashing_size=hashing_size, correction=correction).hashed_data)
        
        self.hashed_data = np.array(output)

    def load_binned(self, beatmaps: list[parser.Beatmap]):
        print("Loading beatmaps...")
        output = list(self.binned_data)
        for beatmap in tqdm(beatmaps):
            binned_data = beatmap.binned_data
            output.append(binned_data)
        
        self.binned_data = np.array(output)
    
    def load_list_binned(self, beatmaps: list[str], binning_size=512, correction=True):
        print("Loading beatmaps into model...")
        output = list(self.binned_data)
        for beatmap in tqdm(beatmaps):
            binned_data = parser.Beatmap(beatmap, binning_size=binning_size, correction=correction).binned_data
            output.append(binned_data)
        
        self.binned_data = np.array(output)

    def forward(self, src):
         # Convert numpy array to PyTorch tensor if necessary
        if isinstance(src, np.ndarray):
            src = torch.tensor(src, dtype=torch.float32)

        # src shape: (batch_size, seq_len, num_features)
        batch_size, seq_len, num_features = src.shape
        outputs = []

        for i in tqdm(range(num_features)):
            feature_series = src[:, :, i]
            feature_series = feature_series.unsqueeze(2) # shape: (batch_size, seq_len, 1)
            feature_series = self.positional_encoding(feature_series) # shape: (seq_len, batch_size, embedding_size)
            feature_series = feature_series.permute(1, 0, 2) # shape: (seq_len, batch_size, embedding_size)
            transformer_output = self.transformer_encoders[i](feature_series) # shape: (seq_len, batch_size, embedding_size)
            transformer_output = transformer_output.permute(1, 0, 2) # shape: (batch_size, seq_len, embedding_size)
            outputs.append(transformer_output)
        
        concatenated_output = torch.cat(outputs, dim=2) # shape: (batch_size, seq_len, embedding_size * num_features)
        concatenated_output = concatenated_output.mean(dim=1) # shape: (batch_size, embedding_size * num_features)
        
        for i, layer in enumerate(self.linear_layers):
            concatenated_output = layer(concatenated_output)
            concatenated_output = self.relu(concatenated_output)

        output = self.last_layer(concatenated_output)

        GMM_output = self.GMM(output)

        GMM_output[torch.isnan(GMM_output)] = -1e10

        return GMM_output

    def one_pass_loss(self, src):
        GMM_log_likelihoods = self.forward(src)
        print(torch.mean(GMM_log_likelihoods, dim=0))
        return GMM_log_likelihoods, loss(GMM_log_likelihoods, src)

    def save(self, path: str):
        pass