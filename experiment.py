import tensorflow as tf
import tensorflow.keras.callbacks as cb
from model import *
from data  import get_datasets
from tensorflow.keras.callbacks import Callback

def study(max_epochs):
    positional_encoding = [True, False]

    for p in positional_encoding:
        experiment(max_epochs, use_positional_encoding=p, load_checkpoint=False)

class ParameterMetrics(Callback):
    def __init__(self, use_positional_encoding):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding

    def on_epoch_end(self, epoch, logs):
        logs['use_positional_encoding'] = int(self.use_positional_encoding)

def experiment(max_epochs, use_positional_encoding, load_checkpoint):
    print(f"Experiment: Positional Encoding={use_positional_encoding}")

    train_dataset, test_dataset, info = get_datasets()    
    vocab_size = info.features['text'].encoder.vocab_size 
    
    transformer = create_model(load_checkpoint, vocab_size, use_positional_encoding)

    param_metrics = ParameterMetrics(use_positional_encoding)
    history = fit_data(max_epochs, transformer, train_dataset, test_dataset, param_metrics)

    return history

def fit_data(max_epochs, model, train_dataset, test_dataset, param_metrics):

    tb = cb.TensorBoard()
    csv = cb.CSVLogger('train.csv', append=True)
    early = cb.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    save = cb.ModelCheckpoint(filepath="checkpoints/train",
             monitor='val_accuracy',
             save_best_only=True,
             save_weights_only=True)

    model_history = model.fit(train_dataset,
        validation_data=test_dataset,  
        validation_freq=1,
        shuffle=False,
        verbose=1,
        callbacks=[param_metrics, save, early, tb, csv],
        epochs=max_epochs)
    return model_history

def create_model(load_checkpoint, vocab_size, use_positional_encoding):
    
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    
    dropout_rate = 0.1    

    transformer = TransformerEncoderClassifier(num_layers, d_model,  num_heads,
    dff, vocab_size, pe_input=vocab_size,  rate=dropout_rate,
    use_positional_encoding=use_positional_encoding)


    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # transformer.run_eagerly = True
    transformer.compile(optimizer=tf.keras.optimizers.Adam(), 
        loss=loss_function, metrics=['accuracy'])

    if load_checkpoint:
        checkpoint = tf.train.latest_checkpoint("./checkpoints")
        if checkpoint is not None:
            print("Loading previously trained model.")
            transformer.load_weights(checkpoint)

    return transformer


if __name__ == '__main__':
    study(max_epochs=100)