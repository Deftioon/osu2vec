from src.osu2vec import parser
import hashlib
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


# Placeholder code
def feature_hashing(data):
    hashed_data = []
    for item in data:
        # Convert item to string and hash it
        hashed_item = hashlib.md5(str(item).encode()).hexdigest()
        hashed_data.append(hashed_item)
    return hashed_data

def one_bit_output_hashing(item):

    hashed_item = hashlib.md5(str(item).encode()).hexdigest()

    hashed_item = int(hashed_item[0], 16) % 2
    return hashed_item

def hashing_vectorizer(data, N):
    hashed_data = {i: 0 for i in range(N)}
    for idx, item in enumerate(data):
        hashed_item = int(hashlib.md5(str(item).encode()).hexdigest(), 16)
        index = hashed_item % N

        if one_bit_output_hashing(item) == 1:
            hashed_data[index] += 1
        else:
            hashed_data[index] -= 1
    
    hashed_data = [x[1] for x in sorted(hashed_data.items(), key=lambda x: x[0])]
    
    return np.array(hashed_data)

def hash_beatmap(beatmap, N):
    hashed_data = {
        "time": hashing_vectorizer(beatmap.dataframe["time"], N),
        "time_diff": hashing_vectorizer(beatmap.dataframe["time_diff"], N),
        "slider_length": hashing_vectorizer(beatmap.dataframe["slider_length"], N),
        "cursor_velocity": hashing_vectorizer(beatmap.dataframe["cursor_velocity"], N),
        "distance": hashing_vectorizer(beatmap.dataframe["distance"], N),
        "angle_cosine": hashing_vectorizer(beatmap.dataframe["angle_cosine"], N),
        "vector_x": hashing_vectorizer(beatmap.dataframe["vector_x"], N),
        "vector_y": hashing_vectorizer(beatmap.dataframe["vector_y"], N),
    }
    hashed_data_array = np.array(list(hashed_data.values())).T
    return hashed_data_array

def beatmap_similarity(beatmap1, beatmap2, N):
    hashed_data1 = hash_beatmap(beatmap1, N)
    hashed_data2 = hash_beatmap(beatmap2, N)
    features = ["time", "time_diff", "slider_length", "cursor_velocity", "distance", "angle_cosine", "vector_x", "vector_y"]
    cosine_similarities = {}

    for column in range(hashed_data1.shape[1]):
        vector1 = hashed_data1[:, column].reshape(1, -1)
        vector2 = hashed_data2[:, column].reshape(1, -1)
        similarity = cosine_similarity(vector1, vector2)
        cosine_similarities[features[column]] = similarity[0][0]

    return cosine_similarities