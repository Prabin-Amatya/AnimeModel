import pandas as pd
import tensorflow as tf
import joblib


def main():
    ##getting data from csv files
    fav_anime_filtered = pd.read_csv("our_staff.csv", names=["User_Id", "Staff_Ids"])
    corpus = fav_anime_filtered["Staff_Ids"]

    # tokenizing the data
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters="", lower=False, split=",", oov_token="OOV"
    )
    tokenizer.fit_on_texts(corpus)
    sorted_word_count = {
        k: v
        for k, v in sorted(
            tokenizer.word_counts.items(), key=lambda item: item[1], reverse=True
        )
    }
    joblib.dump(tokenizer, open("tokenizer2.sav", "wb"))
    print(len(sorted_word_count))
    exit()
    filtered_word_count = list(
        filter(lambda item: item[1] < 10, sorted_word_count.items())
    )
    filtered_word_count = {k: v for k, v in filtered_word_count}
    print(len(filtered_word_count))
    joblib.dump(filtered_word_count, open("unnecessary.sav", "wb"))


if __name__ == "__main__":
    main()
