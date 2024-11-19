from src.osu2vec import parser
from src.osu2vec import data
import numpy as np

N = 4096

beatmap1 = parser.Beatmap("data/map1.osu")

beatmap1.heatmap()

hashed_data1 = data.hash_beatmap(beatmap1, N)

beatmap2 = parser.Beatmap("data/map2.osu")

hashed_data2 = data.hash_beatmap(beatmap2, N)

cosine_similarities = data.beatmap_similarity(beatmap1, beatmap2, N)
print(cosine_similarities)