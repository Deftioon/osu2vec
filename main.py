from src.osu2vec import parser


beatmap = parser.Beatmap("data/map1.osu")
print(beatmap.dataframe[:300])
filtered_beatmap = beatmap.dataframe[(beatmap.dataframe['circle'] != 0) | (beatmap.dataframe['slider'] != 0)]
print(filtered_beatmap)

print(beatmap.dataframe)