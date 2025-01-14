import pandas
import csv
import joblib

file = open("our_staff2.csv", "w", encoding="UTF-8", newline="")
fav_anime_list = pandas.read_csv("our_staff.csv", names=["UserId", "StaffIds"]).sample(
    frac=1
)
unnecessarywords = joblib.load(open("unnecessary.sav", "rb"))
unnecessarywords = set(unnecessarywords.keys())
print(unnecessarywords)
print(len(unnecessarywords))
writer = csv.writer(file)
for index, animes in fav_anime_list.iterrows():
    animeIds = set(animes["StaffIds"].split(","))
    print(animeIds)
    animeIds_filtered = animeIds - unnecessarywords
    print(animeIds_filtered)
    if len(animeIds_filtered) > 3:
        animeIds_filtered = (",").join(str(animeId) for animeId in animeIds_filtered)
        writer.writerow([animes["UserId"], animeIds_filtered])
