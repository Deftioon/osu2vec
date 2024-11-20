import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity



def feature_hashing(data):
    hashed_data = []
    for item in data:
        hashed_item = hashlib.md5(str(item).encode()).hexdigest()
        hashed_data.append(hashed_item)
    return hashed_data

def one_bit_output_hashing(item):

    hashed_item = hashlib.md5(str(item).encode()).hexdigest()

    hashed_item = int(hashed_item[0], 16) % 2
    return hashed_item

def hashing_vectorizer(data, N, correction=False):
    hashed_data = {i: 0 for i in range(N)}
    for idx, item in enumerate(data):
        hashed_item = int(hashlib.md5(str(item).encode()).hexdigest(), 16)
        index = hashed_item % N

        if one_bit_output_hashing(item) != 1 and correction == True:
            hashed_data[index] -= 1
        else:
            hashed_data[index] += 1
    
    hashed_data = [x[1] for x in sorted(hashed_data.items(), key=lambda x: x[0])]
    
    return np.array(hashed_data)

def hash_beatmap(beatmap, N, correction=False):
    hashed_data = {
        "time": hashing_vectorizer(beatmap.dataframe["time"], N, correction),
        "time_diff": hashing_vectorizer(beatmap.dataframe["time_diff"], N, correction),
        "slider_length": hashing_vectorizer(beatmap.dataframe["slider_length"], N, correction),
        "cursor_velocity": hashing_vectorizer(beatmap.dataframe["cursor_velocity"], N, correction),
        "distance": hashing_vectorizer(beatmap.dataframe["distance"], N, correction),
        "angle_cosine": hashing_vectorizer(beatmap.dataframe["angle_cosine"], N, correction),
        "vector_x": hashing_vectorizer(beatmap.dataframe["vector_x"], N, correction),
        "vector_y": hashing_vectorizer(beatmap.dataframe["vector_y"], N, correction),
    }
    hashed_data_array = np.array(list(hashed_data.values())).T
    return hashed_data_array

def percentile_binning(data, label="time", N = 512):
    data = data[["time", "time_diff", "slider_length", "cursor_velocity", "distance", "angle_cosine", "vector_x", "vector_y"]]
    data = data.sort_values(label)
    data["bin"] = pd.qcut(data[label], N, labels=False)
    
    slices = []
    max_size = 0  # variable to store the maximum size of the slices
    
    for i in range(N):
        bin_slice = data[data["bin"] == i]
        bin_slice = bin_slice.drop(columns=["bin"])
        slices.append(np.array(bin_slice))
        max_size = max(max_size, len(bin_slice))  # update the maximum size
    
    # pad all slices to the same size
    for i in range(N):
        bin_slice = slices[i]
        if len(bin_slice) < max_size:
            padding = max_size - len(bin_slice)
            padded_slice = np.pad(bin_slice, ((0, padding), (0, 0)), mode='constant')
            slices[i] = padded_slice
    
    # average each column along each bin
    averaged_slices = []
    for i in range(N):
        bin_slice = slices[i]
        averaged_slice = np.mean(bin_slice, axis=0)
        averaged_slices.append(averaged_slice)
    
    return np.array(averaged_slices)

def beatmap_similarity(beatmap1, beatmap2):
    hashed_data1 = beatmap1.hashed_data
    hashed_data2 = beatmap2.hashed_data
    features = ["time", "time_diff", "slider_length", "cursor_velocity", "distance", "angle_cosine", "vector_x", "vector_y"]
    cosine_similarities = {}

    for column in range(hashed_data1.shape[1]):
        vector1 = hashed_data1[:, column].reshape(1, -1)
        vector2 = hashed_data2[:, column].reshape(1, -1)
        similarity = cosine_similarity(vector1, vector2)
        cosine_similarities[features[column]] = similarity[0][0]

    return cosine_similarities