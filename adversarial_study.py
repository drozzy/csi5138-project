import tensorflow as tf
import tensorflow.keras.callbacks as cb
from model import TransformerEncoderClassifier
from data  import get_datasets
from tensorflow.keras.callbacks import Callback
import plac
import os
import model_awg
from experiment import create_model, EarlyStop

def study(results_dir="results_adversarial", models_dir="models", data_dir="data", max_epochs=100):
    if not tf.io.gfile.exists(results_dir):
        tf.io.gfile.mkdir(results_dir)
    if not tf.io.gfile.exists(models_dir):
        tf.io.gfile.mkdir(models_dir)
    
    use_positional_encoding = True

    experiment(max_epochs, results_dir=results_dir, 
        models_dir=os.path.join(models_dir, f"pos_enc_{use_positional_encoding}"), data_dir=data_dir)

def experiment(max_epochs, results_dir, models_dir, data_dir):
    (train_dataset, test_dataset), info = get_datasets(data_dir)    
    vocab_size = info.features['text'].encoder.vocab_size 
    
    awg_models_dir=os.path.join(models_dir, f"awg")

    model, optimizer, checkpoint, manager = model_awg.create_model(models_dir=awg_models_dir, load_checkpoint=True, 
        run_eagerly=True)

    transformer_models_dir=os.path.join(models_dir, f"pos_enc_True")

    transformer, _, _, _ = create_model(load_checkpoint=True, 
        vocab_size=vocab_size, use_positional_encoding=True, permute_attention=False, 
        models_dir=transformer_models_dir, run_eagerly=True)

    transformer_custom_weights, _, _, _ = create_model(load_checkpoint=True, 
        vocab_size=vocab_size, use_positional_encoding=True, permute_attention=False, 
        models_dir=transformer_models_dir, run_eagerly=True, use_custom_attention_weights=True)

    fit_data(max_epochs, model, optimizer, checkpoint, manager, train_dataset, test_dataset, results_dir, models_dir) #, [param_metrics])

def fit_data(max_epochs, model, optimizer, checkpoint, manager, train_dataset, test_dataset,
    transformer, transformer_custom_weights,
     results_dir="results", models_dir="checkpoints"):
    model_path=os.path.join(models_dir, "train")

    # early_stop = EarlyStop(patience=3)

    metrics = tf.keras.metrics.BinaryCrossentropy(from_logits=True)

    # initial_test_metrics = evaluate(model, test_dataset)
    # report_progress(None, initial_test_metrics)

    for epoch in range(1, max_epochs + 1):
        print(f"Epoch {epoch}")
        # metrics.reset_states()

        train_epoch(epoch, model, optimizer, train_dataset, transformer, transformer_custom_weights, metrics)

        # test_metrics = evaluate(model, test_dataset) 
        # report_progress(metrics, test_metrics)

        # early_stop.step(test_metrics.result().numpy())
        # if early_stop.better:
        #     save_path = manager.save()
        #     print(f"\t Saved checkpoint better accuracy for epoch {epoch}: {save_path}")
        # if early_stop.stop:
        #     print(f'\t Stopping early after {early_stop.steps} steps of no improvement.')
        #     break
        
    # Restore best model
    # print(f"\t Restoring best model with best accuracy {early_stop.best_value}, from {save_path}.")
    # checkpoint.restore(save_path)
        

def train_epoch(epoch, model, optimizer, train_dataset, transformer, transformer_custom_weights, metrics):
    for batch, (x, y) in enumerate(train_dataset):
        train_batch(epoch, batch, x, y, model, optimizer, transformer, transformer_custom_weights, metrics)        

def train_batch(epoch, batch, x, y, model, optimizer, transformer_trained_weights, transformer_custom_weights, metrics):

    _, weights = transformer_trained_weights(x, training=False)

    with tf.GradientTape() as tape:
        adv_weights = model(weights, training=True)
        logits, _ = transformer_custom_weights(x, training=False, custom_attention_weights=adv_weights)
        # Adv weight gen loss
        loss = -tf.keras.losses.KLD(weights, adv_weights) + tf.keras.losses.binary_crossentropy(y_true=y, y_pred=logits, from_logits=True)

    grads = tape.gradient(loss, model.trainable_weights)                
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    metrics.update_state(y_true=y, y_pred=logits)
    
if __name__ == '__main__':
    plac.call(study)