import tensorflow as tf
import tensorflow.keras.callbacks as cb
from model import TransformerEncoderClassifier
from data  import get_datasets
from tensorflow.keras.callbacks import Callback
from bertviz import head_view
import torch

import os

def decode_sentence(s, encoder):
    # s here is x[0] from some batch.
    s_str = encoder.decode(s)
    # Get rid of padding
    tokens = [encoder.decode([w]) for w in s if w != 0]
    return s_str, tokens

def visualize(tokens, weights):    
    # weights - is weights[0] from some batch (e.g. from get_prediction)
    # Make sure to include this in Jupyter Notebook:
    """
    %%javascript
    require.config({
      paths: {
          d3: '//cdnjs.cloudflare.com/ajax/libs/d3/3.4.8/d3.min',
          jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
      }
    });
    """
    head_view(torch.tensor(weights.numpy()), tokens)

def get_prediction(inp_sentence, encoder, transformer, adv_k = None, adv_model = None):
    inp_sentence = encoder.encode(inp_sentence) 

    encoder_input = tf.expand_dims(inp_sentence, 0)
    if adv_k is not None:
        predictions, attn_weights,_ = transformer(encoder_input,custom_k=adv_k ,training=False)
    elif adv_model is not None:
        y_logits, w, k = transformer(encoder_input, training=False)
        adv_k = adv_model(k, training=False)
        predictions, attn_weights, adv_k = transformer(encoder_input, custom_k=adv_k ,training=False)
            
    else:
        predictions, attn_weights,_ = transformer(encoder_input ,training=False)
    
    return predictions, attn_weights

def get_sentiment(prediction):
    sent = tf.squeeze(prediction, axis=0)
    return 'pos' if sent >= 0.5 else 'neg'    

def sentiment(inp_sentence, encoder, transformer, adv_k = None, adv_model = None):
    predictions, attn_weights = get_prediction(inp_sentence, encoder, transformer, adv_k, adv_model)
    
    sent = tf.squeeze(predictions, axis=0)
    if sent >= 0.5:
        sent = 'pos'
    else:
        sent = 'neg'
    return sent, attn_weights

def find_adv_k(x, y, transformer, adv_model, adv_optimizer, epochs=1, random_k=False):
    """
    x - (batch, text)
    y - (batch, label)
    """
    D_MODEL = 128

    best_loss_t = None
    best_adv_k = None
    

    y_logits, w, k = transformer(x, training=False)
    if random_k: # Randomize k
        k = tf.random.uniform(tf.shape(k))

    y_logits2, _, _ = transformer(x, training=False, custom_k=k)
    
    
    loss_t = tf.keras.losses.binary_crossentropy(y_true=y, y_pred=y_logits, from_logits=True)
    best_loss_t = loss_t
    original_loss_t = loss_t
    original_k = k
    
    loss_t = tf.keras.losses.binary_crossentropy(y_true=y, y_pred=y_logits2, from_logits=True)
    alpha = 0.5
    
    for epoch in range(1, epochs+1):        
        with tf.GradientTape() as tape:
            adv_k = adv_model(k, training=True)
            if best_adv_k is None:
                best_adv_k = adv_k
            
            adv_y_logits, adv_w, adv_k2  = transformer(x, custom_k=adv_k, training=False)
            cond = tf.reduce_all(tf.equal(adv_k, adv_k2)).numpy()
            bs = tf.shape(k)[0]
            assert cond == True
            loss_t = tf.keras.losses.binary_crossentropy(y_true=y, y_pred=adv_y_logits, from_logits=True)
            summed = tf.math.reduce_sum(tf.math.pow(tf.reshape(k, [bs, -1]) - tf.reshape(adv_k, [bs, -1]), 2), axis=1)
            
            loss_k = - tf.math.log(tf.math.reduce_mean(summed))
            
            loss = alpha*loss_t + (1 - alpha)*loss_k
            
            if loss_t < best_loss_t:
                best_loss_t = loss_t
                best_adv_k  = adv_k
            
        grads = tape.gradient(loss, adv_model.trainable_weights)                
        adv_optimizer.apply_gradients(zip(grads, adv_model.trainable_weights))
  
    return original_loss_t, original_k, best_loss_t, best_adv_k

def study(results_dir="results", models_dir="models", data_dir="data", max_epochs=100, load_checkpoint=False):
    if not tf.io.gfile.exists(results_dir):
        tf.io.gfile.mkdir(results_dir)
    if not tf.io.gfile.exists(models_dir):
        tf.io.gfile.mkdir(models_dir)

    positional_encoding = [True, False]

    for p in positional_encoding:
        experiment(max_epochs, use_positional_encoding=p, load_checkpoint=load_checkpoint, 
            results_dir=results_dir, models_dir=os.path.join(models_dir, f"pos_enc_{p}"), data_dir=data_dir)

def experiment(max_epochs, use_positional_encoding, load_checkpoint, results_dir, models_dir, data_dir):
    print(f"Experiment: Positional Encoding={use_positional_encoding}")

    (train_dataset, test_dataset), info = get_datasets(data_dir)    
    vocab_size = info.features['text'].encoder.vocab_size 
    
    transformer, optimizer, checkpoint, manager = create_model(load_checkpoint, vocab_size, use_positional_encoding, 
        models_dir, run_eagerly=True)

    history = fit_data(max_epochs, transformer, optimizer, checkpoint, manager, train_dataset, test_dataset, results_dir, models_dir)

    return history

class EarlyStop(object):
    """
    early_stop = EarlyStop(patience=5)

    while True:
        accuracy = ...
        early_stop.step(accuracy)
        if early_stop.better:
            print("This value was better than the best so far!")
        if early_stop.stop:
            print(f'Stopping early after {early_stop.steps} steps of no improvement.')
            break

    print(f"Best accuracy was {early_stop.best_value}.")    
    """
    def __init__(self, patience):
        assert patience >= 1              
        self.patience = patience
        self.best_value = None  
        self.steps = 0   
        self.stop = False
        self.better = True     

    def step(self, value):
        if (self.best_value is None) or (value > self.best_value):
            self.better = True
            self.best_value = value
            self.steps = 0
        else:
            self.better = False
            self.steps +=1

        self.stop = (self.steps >= self.patience)

def fit_data(max_epochs, model, optimizer, checkpoint, manager, train_dataset, test_dataset, results_dir="results", 
        models_dir="checkpoints"):
    model_path=os.path.join(models_dir, "train")

    early_stop = EarlyStop(patience=3)

    metrics = tf.keras.metrics.BinaryCrossentropy(from_logits=True)

    initial_test_metrics = evaluate(model, test_dataset)
    report_progress(None, initial_test_metrics)

    for epoch in range(1, max_epochs + 1):
        print(f"Epoch {epoch}")
        metrics.reset_states()

        train_epoch(epoch, model, optimizer, train_dataset, metrics)

        test_metrics = evaluate(model, test_dataset) 
        report_progress(metrics, test_metrics)

        early_stop.step(test_metrics.result().numpy())
        if early_stop.better:
            save_path = manager.save()
            print(f"\t Saved checkpoint better accuracy for epoch {epoch}: {save_path}")
        if early_stop.stop:
            print(f'\t Stopping early after {early_stop.steps} steps of no improvement.')
            break
        
    # Restore best model
    print(f"\t Restoring best model with best accuracy {early_stop.best_value}, from {save_path}.")
    checkpoint.restore(save_path)
        

def train_epoch(epoch, model, optimizer, train_dataset, metrics):

    for batch, (x, y) in enumerate(train_dataset):
        train_batch(epoch, batch, x, y, model, optimizer, metrics)        

def train_batch(epoch, batch, x, y, model, optimizer, metrics):

    with tf.GradientTape() as tape:
        (logits, _weights) = model(x, training=True)
        
        loss = tf.keras.losses.binary_crossentropy(y_true=y, y_pred=logits, from_logits=True)

    grads = tape.gradient(loss, model.trainable_weights)                
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    metrics.update_state(y_true=y, y_pred=logits)
    

def evaluate(model, test_dataset, adv_model=None):
    metrics = tf.keras.metrics.Accuracy()
    for (x, y) in test_dataset:
        if adv_model is not None:
            logits, _, k = model(x, training=False)
            adv_k = adv_model(k, training = False)
            logits, _, k = model(x, custom_k=adv_k, training=False)
        else:
            logits, _, _ = model(x, training=False)
        predicted = tf.cast(tf.round(tf.nn.sigmoid(logits)), dtype=tf.int64)
        metrics.update_state(y_true=y, y_pred=predicted)

    return metrics

def report_progress(train_metrics, test_metrics=None):
    if train_metrics is not None:
        print(f'\t Train Loss: {train_metrics.result().numpy()}')
    if test_metrics is not None:
        print(f'\t Test Accuracy={test_metrics.result().numpy()}')

def shuffle_weights(weights):
    return tf.transpose(tf.random.shuffle(tf.transpose(weights)))    

def create_model(load_checkpoint, vocab_size, use_positional_encoding, 
        models_dir="checkpoints", run_eagerly=False):
    
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    
    dropout_rate = 0.1    

    transformer = TransformerEncoderClassifier(num_layers, d_model,  num_heads,
        dff, vocab_size, pe_input=vocab_size, rate=dropout_rate,
        use_positional_encoding=use_positional_encoding)

    # loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    transformer.run_eagerly = run_eagerly

    optimizer = tf.keras.optimizers.Adam()    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformer)    
    manager = tf.train.CheckpointManager(checkpoint, models_dir, max_to_keep=3)

    if load_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            print("Loaded previously trained model.")            
        else:
            print("No previous model found. Training from scratch.")
    else:
        print("Not loading previously trained model. Training from scratch.")


    return transformer, optimizer, checkpoint, manager


if __name__ == '__main__':
    plac.call(study)