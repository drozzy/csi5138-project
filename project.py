import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from functools import partial

## Setup input pipline
data_path = 'data'

def get_datasets():
    (train_examples, test_examples), info = tfds.load('imdb_reviews/subwords8k',  data_dir=data_path,
    split=['train', 'test'],
    with_info=True, as_supervised=True)
    return train_examples, test_examples, info


def transform_datasets(train_examples, test_examples, encoder, batch_size, max_length, buffer_size):
    """
    Use [TFDS](https://www.tensorflow.org/datasets) to load the IMDB movie reviews dataset with labels for positive or negative sentiments.

    This dataset contains 25000 training examples and 25000 test examples.
    Note: To keep this example small and relatively fast, drop examples with some max length.
    """
   
    def filter_max_length(review, _sentiment):
        return tf.size(review) <= max_length

    train_dataset = train_examples.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size).padded_batch(
        batch_size, padded_shapes=train_examples.output_shapes)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    test_dataset = test_examples.filter(filter_max_length)
    test_dataset = test_dataset.padded_batch(
        batch_size, padded_shapes=train_examples.output_shapes)

    return train_dataset, test_dataset

def path_to(fname):
    return os.path.join(data_path, fname)


## Checkpoint manager
def create_checkpoint_manager(transformer, optimizer, max_to_keep):
    """
    Create the checkpoint path and the checkpoint manager. This will be used to save checkpoints every `n` epochs.
    """
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    return ckpt_manager


## Optimizer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Use the Adam optimizer with a custom learning rate scheduler according
    to the formula in the [paper](https://arxiv.org/abs/1706.03762).
    """
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

## Evaluate

def evaluate(inp_sentence, encoder, transformer):
    # inp sentence is the review  
    inp_sentence = encoder.encode(inp_sentence)  
    encoder_input = tf.expand_dims(inp_sentence, 0)

    predictions = transformer(encoder_input, False)
   
    sent = tf.squeeze(predictions, axis=0)
    if sent >= 0.5:
        sent = 'pos'
    else:
        sent = 'neg'
    return sent

def sentiment(review, encoder, transformer, plot=''):
    sentiment = evaluate(review, encoder, transformer)
  
    print('Input: {}'.format(review))
    print('Predicted sentiment: {}'.format(sentiment))
  
#     if plot:
#         plot_attention_weights(attention_weights, sentence, result, plot)
# def plot_attention_weights(attention, sentence, result, layer):
#   fig = plt.figure(figsize=(16, 8))
  
#   sentence = tokenizer_pt.encode(sentence)
  
#   attention = tf.squeeze(attention[layer], axis=0)
  
#   for head in range(attention.shape[0]):
#     ax = fig.add_subplot(2, 4, head+1)
    
#     # plot the attention weights
#     ax.matshow(attention[head][:-1, :], cmap='viridis')

#     fontdict = {'fontsize': 10}
    
#     ax.set_xticks(range(len(sentence)+2))
#     ax.set_yticks(range(len(result)))
    
#     ax.set_ylim(len(result)-1.5, -0.5)
        
#     ax.set_xticklabels(
#         ['<start>']+[tokenizer_pt.decode([i]) for i in sentence]+['<end>'], 
#         fontdict=fontdict, rotation=90)
    
#     ax.set_yticklabels([tokenizer_en.decode([i]) for i in result 
#                         if i < tokenizer_en.vocab_size], 
#                        fontdict=fontdict)
    
#     ax.set_xlabel('Head {}'.format(head+1))
  
#   plt.tight_layout()
#   plt.show()