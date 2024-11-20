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

class Osu2Vec(nn.Module):
    def __init__(self, hashing_size: int=512, embedding_size: int = 512, vector_size: int=256, num_layers=6, num_heads=8, feature_size=8):
        super(Osu2Vec, self).__init__()
        self.hashing_size = hashing_size
        self.embedding_size = embedding_size
        self.vector_size = vector_size

        self.hashed_data = np.array([])
        self.binned_data = np.array([])

        self.positional_encoding = PositionalEncoding(self.embedding_size)
        
        self.transformer_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=num_heads),
                num_layers=num_layers
            ) for _ in range(feature_size)
        ])

        self.fc1 = nn.Linear(self.embedding_size * feature_size, self.vector_size)
        self.relu = nn.ReLU()

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
        output = self.fc1(concatenated_output)
        output = self.relu(output)

        return output

    def save(self, path: str):
        pass