from src.osu2vec import parser


beatmap = parser.Beatmap("data/map2.osu")
print(beatmap.metadata)
circle_count = beatmap.dataframe['circle'].sum()
slider_count = beatmap.dataframe['slider'].sum()
print(f"Number of circles: {circle_count}")
print(f"Number of sliders: {slider_count}")

print(beatmap.dataframe)
average_angle_cosine = beatmap.dataframe['angle_cosine'].mean()
print(f"Average angle cosine: {average_angle_cosine}")

beatmap.heatmap(save=True, show=True)