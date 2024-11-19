# Data Submodule
This module allows the processing of `parser.Beatmap` objects, allowing to calculate various metrics and vectorization.

## Methods:
### `data.hash_beatmap(beatmap: parser.Beatmap, N: int) -> np.ndarray`
This method uses [Feature Hashing](https://arxiv.org/pdf/0902.2206) to convert a beatmap of unknown length to a matrix with dimensions $(N, 8)$. The rows are essentially features across a time, while the columns are the hashed features. In order, the columns are the hashed features of `["time", "time_diff", "slider_length", "cursor_velocity", "distance", "angle_cosine", "vector_x", "vector_y"]`

### `data.beatmap_similarity(beatmap1: parser.Beatmap, beatmap2: parser.Beatmap, N: int) -> dict`
This method uses the previous `data.hash_beatmap` method to convert two `parser.Beatmap` objects into vectorized matrices, and then calculates the cosine similarity between each of the features. This is stored in a dictionary, with each key `["time", "time_diff", "slider_length", "cursor_velocity", "distance", "angle_cosine", "vector_x", "vector_y"]` being paired with the corresponding cosine similarity.