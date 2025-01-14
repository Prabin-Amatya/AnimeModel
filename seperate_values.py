import joblib


def main():
    num = 1
    for i in range(1, 7):
        full_targets = joblib.load(f"targets{i}.sav")
        full_contexts = joblib.load(f"contexts{i}.sav")
        for targets, contexts in zip(full_targets, full_contexts):
            joblib.dump(targets, open(f"./Variables/Targets/target{num}.sav", "wb"))
            joblib.dump(contexts, open(f"./Variables/Contexts/context{num}.sav", "wb"))
            num += 1


main()
