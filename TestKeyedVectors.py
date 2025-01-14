import gensim

AnimeKeyedVector = gensim.models.KeyedVectors.load("AnimeKeyedVectors.kv")


similar_anime = AnimeKeyedVector.most_similar(
    positive="101922",
    negative=[],
    topn=25,
)

print(similar_anime)
