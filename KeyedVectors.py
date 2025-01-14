import gensim
import csv

embedding_dims = csv.reader(
    open("staff2Vec_embedding_dims100Final.tsv", "r", encoding="UTF-8"), delimiter="\t"
)
tokens = csv.reader(
    open("staff2Vec_embedding_metadata100Final.tsv", "r", encoding="UTF-8"),
    delimiter="\t",
)

StaffKeyedVectors = gensim.models.KeyedVectors(60)
token_vector = {}
for token, embedding_dim in zip(tokens, embedding_dims):
    if token[0] == "OOV":
        StaffKeyedVectors.add_vector(token[0], list(map(float, [0] * 60)))
    else:
        StaffKeyedVectors.add_vector(token[0], list(map(float, embedding_dim)))
StaffKeyedVectors.save("StaffKeyedVectors60.kv")
