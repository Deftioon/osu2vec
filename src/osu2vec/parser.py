import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class HitCircle:
    def __init__(self, x: int, y: int, time: int):
        self.x = x
        self.y = y
        self.time = time

class Slider:
    def __init__(self, x: int, y: int, time: int, points: list, slides: int, length: int, duration: int):
        self.x = x
        self.y = y
        self.time = time
        self.points = points
        self.slides = slides
        self.length = length
        self.duration = duration

class Beatmap:
    def __init__(self, file_path: str, file_type: str=".osu"):
        with open(file_path, "r", encoding="utf-8") as file:
            self.file = file.read()
        self.file_path = file_path
        self.file_type = file_type
        self.metadata = None
        self.stats = self.split_file()
        self.difficulty = self.stats["difficulty"]

        self.dataframe = pd.DataFrame(columns=["time", "x", "y", "circle", "slider", "slider_length", "cursor_velocity", "distance","angle_cosine", "vector_x", "vector_y"])

        self.parse()

        self.normalized_dataframe = self.dataframe.copy()
        columns_to_normalize = self.normalized_dataframe.columns.difference(['angle_cosine'])
        self.normalized_dataframe[columns_to_normalize] = 2 * (self.normalized_dataframe[columns_to_normalize] - self.normalized_dataframe[columns_to_normalize].min()) / (self.normalized_dataframe[columns_to_normalize].max() - self.normalized_dataframe[columns_to_normalize].min()) - 1
        


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

        metadata_label_idx = file.index("[Metadata]")
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

        metadata_data = file[metadata_label_idx+1:difficulty_label_idx]
        metadata_data = [x.split(":") for x in metadata_data]
        metadata_data = {x[0]: x[1] for x in metadata_data}

        self.metadata = metadata_data

        output = {
            "difficulty": difficulty_data,
            "timing_points": timing_points_data,
            "hit_objects": hit_objects_data,
        }
        
        return output

    def parse(self):
        """
        Creates a dataframe from the hit_objects data.
        """
        split_data = self.split_file()
        timings = split_data["timing_points"]
        hit_objects = split_data["hit_objects"]

        for i, hit_object in enumerate(hit_objects):
            time = hit_object[2]
            x = hit_object[0]
            y = hit_object[1]

            if len(hit_object) < 11:
                circle = 1
                slider = 0
                slider_length = 0
            else:
                circle = 0
                slider = 1
                slider_length = hit_object[7]

            if i == 0:
                distance = 0
                time_diff = 0
                angle_cosine = 0
                cursor_velocity = 0
                vector_x = 0
                vector_y = 0
            else:
                prev_x = int(self.dataframe.iloc[-1]["x"])
                prev_y = int(self.dataframe.iloc[-1]["y"])
                prev_time = int(self.dataframe.iloc[-1]["time"])

                time_diff = int(time) - prev_time
                distance = np.sqrt((int(x) - prev_x) ** 2 + (int(y) - prev_y) ** 2)
                cursor_velocity = distance / time_diff if time_diff != 0 else 0
                vector_x = int(x) - prev_x
                vector_y = int(y) - prev_y
                
                if i == 1:
                    angle_cosine = 0
                else:
                    angle_cosine = (int(x) - prev_x) / distance if distance != 0 else 0
                    prev_vector_x = int(self.dataframe.iloc[-1]["vector_x"])
                    prev_vector_y = int(self.dataframe.iloc[-1]["vector_y"])
                    prev_distance = np.sqrt(prev_vector_x ** 2 + prev_vector_y ** 2)
                    
                    if prev_distance != 0 and distance != 0:
                        angle_cosine = (prev_vector_x * vector_x + prev_vector_y * vector_y) / (prev_distance * distance)
                    else:
                        angle_cosine = 0

            new_row = pd.DataFrame([{
                "time": time,
                "time_diff": time_diff,
                "x": x,
                "y": y,
                "circle": circle,
                "slider": slider,
                "slider_length": slider_length,
                "cursor_velocity": cursor_velocity,
                "distance": distance,
                "angle_cosine": angle_cosine,
                "vector_x": vector_x,
                "vector_y": vector_y
            }])
            self.dataframe = pd.concat([self.dataframe, new_row], ignore_index=True)

        # # Create a complete dataframe with all milliseconds from 1 to 500000
        # complete_time_range = pd.DataFrame({"time": np.arange(1, 500001)})

        # # Ensure 'time' column in both dataframes is of the same type (int)
        # self.dataframe["time"] = self.dataframe["time"].astype(int)
        # complete_time_range["time"] = complete_time_range["time"].astype(int)

        # # Merge the complete time range with the existing dataframe
        # self.dataframe = pd.merge(complete_time_range, self.dataframe, on="time", how="left")

        # # Fill NaN values with 0 for x, y, circle, and slider columns
        # self.dataframe[["x", "y", "circle", "slider"]] = self.dataframe[["x", "y", "circle", "slider"]].fillna(0)

        # Convert all columns to integers
        self.dataframe = self.dataframe.astype(float)

    def heatmap(self, show=True, save=True):
        normalized_df = self.dataframe.copy()
        columns_to_normalize = normalized_df.columns.difference(['angle_cosine'])
        normalized_df[columns_to_normalize] = 2 * (normalized_df[columns_to_normalize] - normalized_df[columns_to_normalize].min()) / (normalized_df[columns_to_normalize].max() - normalized_df[columns_to_normalize].min()) - 1
        plt.figure(figsize=(10, 8))
        sns.heatmap(normalized_df, cmap='coolwarm', cbar=True)
        plt.title(f"Heatmap of Beatmap {self.metadata['Title']} ({self.metadata['Version']})")
        
        if save:
            plt.savefig(f"{self.metadata['Title'].replace(' ', '_')} ({self.metadata['Version'].replace(' ', '_')}).png")
        if show:
            plt.show()


    def __repr__(self):
        return f"Beatmap({self.file_path})"

if __name__ == "__main__":
    pass
