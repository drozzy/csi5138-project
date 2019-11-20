import tensorflow_datasets as tfds
import tensorflow as tf

DATA_DIR = 'data'
BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 200

## Setup input pipline
def get_datasets():
    """
    Use [TFDS](https://www.tensorflow.org/datasets) to load the IMDB movie reviews dataset with labels for positive or negative sentiments.

    This dataset contains 25000 training examples and 25000 test examples.
    Note: To keep this example small and relatively fast, drop examples with some max length.
    """
    (train_examples, test_examples), info = tfds.load('imdb_reviews/subwords8k',  data_dir=DATA_DIR,
    split=['train', 'test'],
    with_info=True, as_supervised=True)

    def filter_max_length(review, _sentiment):
        return tf.size(review) <= MAX_LENGTH

    train_dataset = train_examples.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
        BATCH_SIZE, padded_shapes=train_examples.output_shapes)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = test_examples.filter(filter_max_length)
    test_dataset = test_dataset.padded_batch(
        BATCH_SIZE, padded_shapes=train_examples.output_shapes)

    return (train_dataset, test_dataset), info