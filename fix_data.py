import pandas
import csv

animes = pandas.read_csv("to_change.csv", names=["user", "anime"])
csv_writer = csv.writer(open("anime_list.csv", "w", encoding="utf-8", newline=""))
for i, (user, anime) in animes.iterrows():
    split_animes = anime.split(",")
    join_anime = (", ").join(split_animes)
    csv_writer.writerow([user, join_anime])
