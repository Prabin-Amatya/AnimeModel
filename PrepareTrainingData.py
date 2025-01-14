import pandas as pd
import tensorflow as tf
import numpy
import tqdm
import joblib
import gc
from itertools import repeat
from multiprocessing import Pool


numpy.random.seed(1)
tf.random.set_seed(0)


##getting a sampling distribution
def make_sampling_table(word_frequencies):
    sorted_word_count = {
        k: v
        for k, v in sorted(word_frequencies, key=lambda item: item[1], reverse=True)
    }
    total = []
    for item, count in sorted_word_count.items():
        total.append(pow(count, 0.5))
    total_sum = numpy.sum(total)
    probs = numpy.divide(total, total_sum)
    return list(probs)


tokenizer = joblib.load(open("tokenizer2.sav", "rb"))
print(len(tokenizer.word_index))
sampling_table = make_sampling_table(tokenizer.word_counts.items())

vocab_size = len(tokenizer.word_index)
#  oov so start from index 1
vocab_words = list(tokenizer.index_word.keys())[1:]


##generating positive and negative samples
def generate_training_data(sequences, window_size, num_ns):
    targets, contexts, labels = [], [], []

    for sequence in tqdm.tqdm(sequences):
        # print("sequence", sequence)
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            window_size=window_size,
            negative_samples=0,
            seed=1,
        )
        for target_word, context_word in positive_skip_grams:
            # print("target word", target_word)
            # print("context word", context_word)
            negative_sampling_words = []
            negative_sampling_words = numpy.random.choice(
                vocab_words,
                size=num_ns,
                replace=False,
                p=sampling_table,
            )
            ##filtering target and context words
            negative_sampling_words = [
                x
                for x in negative_sampling_words
                if x != target_word and x != context_word
            ]
            ##sampling till there are 10 negative samples
            while len(negative_sampling_words) != num_ns:
                additional_samples = numpy.random.choice(
                    vocab_words,
                    size=num_ns - len(negative_sampling_words),
                    replace=False,
                    p=sampling_table,
                )
                ##checking target and context word again
                additional_samples = [
                    x
                    for x in additional_samples
                    if x != target_word and x != context_word
                ]
                negative_sampling_words.extend(additional_samples)

            context = tf.concat(
                [
                    tf.constant([context_word], dtype="int64"),
                    tf.constant(negative_sampling_words, dtype="int64"),
                ],
                axis=0,
            )

            # print(context)
            targets.append(target_word)
            contexts.append(context)
    return (targets, contexts)


def main():
    ##getting data from csv files

    pool_size = 4
    num = 1
    # tokenizing the data
    fav_anime_filtered = pd.read_csv(f"our_staff.csv", names=["User_Id", "Staff_Ids"])
    corpus = fav_anime_filtered["Staff_Ids"]
    vectorized_corpus = tokenizer.texts_to_sequences(corpus)
    window_size = 25
    negative_samples = 10
    size = len(vectorized_corpus)
    pool = Pool(pool_size)
    split_vectorized_corpus = [
        vectorized_corpus[i : i + int(size / pool_size)]
        for i in range(0, len(vectorized_corpus), int(size / pool_size))
    ]
    results = pool.starmap(
        generate_training_data,
        zip(split_vectorized_corpus, repeat(window_size), repeat(negative_samples)),
    )
    pool.close()
    pool.join()
    print("done")
    part_targets, part_contexts = zip(*results)

    for target, context in zip(part_targets, part_contexts):
        joblib.dump(target, open(f"./Variables/TargetsStaff/target{num}.sav", "wb"))
        joblib.dump(context, open(f"./Variables/ContextsStaff/context{num}.sav", "wb"))
        num += 1
        print("num")
    print("done")
    del pool
    del part_contexts
    del part_targets
    del results
    print("done")
    gc.collect()


if __name__ == "__main__":
    main()
