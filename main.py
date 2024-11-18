from src.osu2vec import parser
import seaborn as sns


beatmap = parser.Beatmap("data/map1.osu")
circle_count = beatmap.dataframe['circle'].sum()
slider_count = beatmap.dataframe['slider'].sum()
print(f"Number of circles: {circle_count}")
print(f"Number of sliders: {slider_count}")

print(beatmap.dataframe)
average_angle_cosine = beatmap.dataframe['angle_cosine'].mean()
print(f"Average angle cosine: {average_angle_cosine}")

import matplotlib.pyplot as plt

# Normalize the dataframe
normalized_df = beatmap.dataframe.copy()
columns_to_normalize = normalized_df.columns.difference(['angle_cosine'])
normalized_df[columns_to_normalize] = 2 * (normalized_df[columns_to_normalize] - normalized_df[columns_to_normalize].min()) / (normalized_df[columns_to_normalize].max() - normalized_df[columns_to_normalize].min()) - 1

# Plot the dataframe as a heatmap with each data point as a pixel
plt.figure(figsize=(10, 8))
sns.heatmap(normalized_df, cmap='coolwarm', cbar=True)
plt.title('Heatmap of Beatmap Dataframe')
plt.show()