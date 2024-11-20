from src.osu2vec import data, parser
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

N = 100

beatmap1 = parser.Beatmap("data/map7.osu")

hashed_data1 = data.hash_beatmap(beatmap1, N, correction=True)