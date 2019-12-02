import tensorflow as tf

class Adversarial(tf.keras.Model):
    def __init__(self, d_model):
        super().__init__()
        
        hidden_size = 256

        self.d1 = tf.keras.layers.Dense(hidden_size, input_shape=(d_model,), activation='relu')
        self.d2 = tf.keras.layers.Dense(d_model)

    def call(self, k): # Input is "k"-value: (batch, seq_len, d_model) or just (..., d_model) as the first two are batch dims
        return self.d2(self.d1(k))

def create_model(models_dir, load_checkpoint, run_eagerly=True, d_model=512):

    model = Adversarial(d_model=d_model)
    model.run_eagerly = run_eagerly

    optimizer = tf.keras.optimizers.Adam()    
    # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)    
    # manager = tf.train.CheckpointManager(checkpoint, models_dir, max_to_keep=3)

    # if load_checkpoint:
    #     checkpoint.restore(manager.latest_checkpoint)

    #     if manager.latest_checkpoint:
    #         print("Loaded previously trained model.")            
    #     else:
    #         print("No previous model found. Training from scratch.")
    # else:
    #     print("Not loading previously trained model. Training from scratch.")

    return model, optimizer #, checkpoint, manager