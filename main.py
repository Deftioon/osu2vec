from src.osu2vec import parser


beatmap = parser.Beatmap("data/map1.osu")
print(len(beatmap.stats["hit_objects"]))