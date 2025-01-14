import pandas as pd
import tensorflow as tf
import numpy
import joblib
import io
import gc
import keras
from tensorflow.keras import backend as K

numpy.random.seed(1)
tf.random.set_seed(0)


class Anime2Vec(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dims,
        target_embeddings=None,
        context_embeddings=None,
    ):
        self.target_embeddings = (
            target_embeddings
            if target_embeddings
            else tf.keras.layers.Embedding(
                vocab_size, embedding_dims, name="A2V_embedding"
            )
        )
        self.context_embeddings = (
            context_embeddings
            if context_embeddings
            else tf.keras.layers.Embedding(
                vocab_size, embedding_dims, name="A2V_context"
            )
        )
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        super(Anime2Vec, self).__init__()

    def call(self, pair):
        target, context = pair
        if len(target.shape) == 2:
            target = tf.squeeze(target)
        word_embedding = self.target_embeddings(target)
        context_embedding = self.context_embeddings(context)
        dots = tf.einsum("be, bce -> bc", word_embedding, context_embedding)
        return dots

    def get_config(self):
        base_config = super().get_config()
        config = {
            "A2V_embedding_config": keras.saving.serialize_keras_object(
                self.target_embeddings
            ),
            "A2V_context_config": keras.saving.serialize_keras_object(
                self.context_embeddings
            ),
            "vocab_size": self.vocab_size,
            "embedding_dims": self.embedding_dims,
            # Save weights explicitly
            "target_weights": [
                w.tolist() for w in self.target_embeddings.get_weights()
            ],
            "context_weights": [
                w.tolist() for w in self.context_embeddings.get_weights()
            ],
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        targetlayer_config = config.pop("A2V_embedding_config")
        contextlayer_config = config.pop("A2V_context_config")
        target_weights = [numpy.array(w) for w in config.pop("target_weights")]
        context_weights = [numpy.array(w) for w in config.pop("context_weights")]

        # Deserialize layers
        targetlayer = keras.saving.deserialize_keras_object(targetlayer_config)
        contextlayer = keras.saving.deserialize_keras_object(contextlayer_config)

        # Create the instance
        instance = cls(
            vocab_size=config["vocab_size"],
            embedding_dims=config["embedding_dims"],
            target_embeddings=targetlayer,
            context_embeddings=contextlayer,
        )

        # Set the restored weights
        instance.target_embeddings.set_weights(target_weights)
        instance.context_embeddings.set_weights(context_weights)

        return instance


def main():
    tokenizer = joblib.load(open("tokenizer.sav", "rb"))
    vocab_size = len(tokenizer.index_word) + 1
    vocab = tokenizer.index_word
    ##getting data from csv files
    start = 14
    for i in range(5, 6):

        print("Enter epoch:", i)
        for j in range(start, 26):
            if i == 1 and j == 1:
                print("Created")
                optimizer = keras.optimizers.Adam(learning_rate=0.0001)
                embedding_dims = 100
                anime2vec = Anime2Vec(
                    vocab_size,
                    embedding_dims,
                )
                anime2vec.compile(
                    optimizer=optimizer,
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=["accuracy"],
                )
            else:
                print("Loaded")
                anime2vec = tf.keras.models.load_model(
                    "anime_to_vector_md100Final.keras",
                    custom_objects={"Anime2Vec": Anime2Vec},
                )
            print("Epoch:", i)
            print("Entered file:", j)
            targets = joblib.load(f"./Variables/Targets/target{j}.sav")
            contexts = joblib.load(f"./Variables/Contexts/context{j}.sav")
            labels = [[1] + [0] * 5] * len(contexts)
            print(len(targets))
            print(len(contexts))
            BATCH_SIZE = 128
            BUFFER_SIZE = 2048
            dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))

            dataset = dataset.shuffle(BUFFER_SIZE).batch(
                BATCH_SIZE, drop_remainder=True
            )
            dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

            anime2vec.fit(dataset, epochs=1)

            del targets
            del contexts
            del labels
            del dataset
            gc.collect()
            print("Exit file:", j)

            anime2vec.save("anime_to_vector_md100Final.keras")
            weights = anime2vec.get_layer("A2V_embedding").get_weights()[0]
            out_v = io.open("anime_embedding_dims100Final.tsv", "w", encoding="utf-8")
            out_m = io.open(
                "anime_embedding_metadata100Final.tsv", "w", encoding="utf-8"
            )
            out_i = io.open("anime_embedding_index100Final.tsv", "w", encoding="utf-8")

            for index, anime in vocab.items():
                vector = weights[index]
                out_v.write("\t".join([str(weight) for weight in vector]) + "\n")
                out_m.write(anime + "\n")
                out_i.write(str(index) + "\n")

            out_v.close()
            out_m.close()
            del out_v
            del out_m
            del out_i
            del anime2vec
            gc.collect()
            K.clear_session()
        print("Exit epoch:", i)
        start = 1


if __name__ == "__main__":
    main()
