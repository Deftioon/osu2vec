# Parser Submodule
This module parses beatmap files into a machine-readable format, which can then be vectorised.

# `parser.Beatmap`
The `Beatmap` class parses a `.osu` file into an intuitive format.

```py
beatmap = parser.Beatmap("maps/map.osu")
```

When instantiating a `Beatmap` object, two parameters are given to the constructor: `file_path`, the location of the file, and the `file_type`. Currently, `.osu` is the default and only supported format.

## Attributes:

### `Beatmap.file_path`
Contains the path to the file

### `Beatmap.file_type`
Contains the file type (`.osu`)

### `Beatmap.metadata`
Contains the metadata of a beatmap, as contained in the `.osu` file. It is presented in a dictionary format, with keys `["Title", "TitleUnicode", "Artist", "ArtistUnicode", "Creator", "Version", "Source", "Tags", "BeatmapID", "BeatmapSetID"]`.

### `Beatmap.difficulty`
Contains the beatmap difficulty information, in a dictionary format, with keys `["HPDrainRate", "CircleSize", "OverallDifficulty", "ApproachRate", "SliderMultiplier", "SliderTickRate"]`

### `Beatmap.stats`
Contains the beatmap stats, which are the difficulty information, timing points, and a list of hit objects. This is all in native `.osu` formatting.

### `Beatmap.dataframe`
Contains a `pandas` dataframe of the hit objects in the map, presented as a timeline, with the first column of the dataframe being the `time` in milliseconds. In this dataframe, each row contains the information of the note type, note position, average cursor velocity, angle formed by the current and previous jump, and the vector from the last hit object to the current.

### `Beatmap.normalized_dataframe`
Contains a `pandas` dataframe of the hit objects in the map, presented in the same format. This is normalized from $-1$ to $1$. Everything is normalized except for the angle, which is stored as the cosine value.

## Methods:

### `Beatmap.heatmap(show=True, save=True)`
Generates a heatmap representation of the beatmap as a timeline. This provides visualization for jumps, streams, and difficulty spikes.

Example:
![Colors Power](<Colors_Power_ni_Omakasero! (Mitsuboshi_Jump!).png>)
