from src.osu2vec import data, parser
from src.osu2vec import model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

N = 512

beatmap1 = parser.Beatmap("data/map1.osu", correction=True)

print(beatmap1.dataframe.head())

osu2vec = model.Osu2Vec()
osu2vec.load_list(["data/map1.osu", "data/map2.osu", "data/map4.osu"], correction=True)
print(osu2vec.linear_layers)
print(osu2vec.binned_data.shape)
print(osu2vec.hashed_data.shape)

log_likelihoods, loss = osu2vec.one_pass_loss(osu2vec.binned_data)
print(loss)