from src.osu2vec import data, parser
from src.osu2vec import model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

N = 512

beatmap1 = parser.Beatmap("data/map4.osu", correction=True)

beatmap2 = parser.Beatmap("data/map7.osu", correction=True)

similarities = data.beatmap_similarity(beatmap1, beatmap2)
average_similarity = np.mean(list(similarities.values()))
print("Average Similarity:", average_similarity)
print("Similarities:", similarities)