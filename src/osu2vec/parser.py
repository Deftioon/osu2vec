import pandas as pd
import numpy as np

class Beatmap:
    def __init__(self, file_path: str, file_type: str=".osu"):
        with open(file_path, "r", encoding="utf-8") as file:
            self.file = file.read()
        self.file_path = file_path
        self.file_type = file_type
        self.stats = self.split_file()
        self.difficulty = self.stats["difficulty"]

        self.dataframe = pd.DataFrame(columns=["time", "x", "y", "circle", "slider"])

    def split_file(self):
        """
        Splits the file into different sections.

        Output:
            - [Difficulty]
                - HPDrainRate: Decimal, [0, 10]
                - CircleSize: Deciman, [0, 10]
                - OverallDifficulty: Decimal, [0, 10]
                - ApproachRate: Decimal, [0, 10]
                - SliderMultiplier: Decimal, Hundreds of osu! pixels per beat
                - SliderTickRate: Decimal, Amount of slider ticks per beat
            - [TimingPoints]
            - [HitObjects]
        """
        file = self.file.split("\n")
        file = [line for line in file if line.strip()]  # Remove empty lines
        difficulty_label_idx = file.index("[Difficulty]")
        events_label_idx = file.index("[Events]")
        timing_points_label_idx = file.index("[TimingPoints]")
        colours_label_idx = file.index("[Colours]")
        hit_objects_label_idx = file.index("[HitObjects]")

        difficulty_data = file[difficulty_label_idx+1:events_label_idx]
        difficulty_data = [x.split(":") for x in difficulty_data]
        difficulty_data = {x[0]: x[1] for x in difficulty_data}

        hit_objects_data = file[hit_objects_label_idx+1:]
        hit_objects_data = [x.split(",") for x in hit_objects_data]

        timing_points_data = file[timing_points_label_idx+1:colours_label_idx]
        timing_points_data = [x.split(",") for x in timing_points_data]

        output = {
            "difficulty": difficulty_data,
            "timing_points": timing_points_data,
            "hit_objects": hit_objects_data,
        }
        
        return output

    def parse(self):
        split_data = self.split_file()
        timings = split_data["timing_points"]
        hit_objects = split_data["hit_objects"]
        return split_data

    def __repr__(self):
        return f"Beatmap({self.file_path})"

if __name__ == "__main__":
    pass
