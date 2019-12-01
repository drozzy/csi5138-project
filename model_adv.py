# TODO:
# Create adversarial weight generator model
# Test it 
import tensorflow as tf
class Adversarial(tf.keras.Model):
    def __init__(self, input_shape, out_features):
        super().__init__()
        hidden_size = 256

        print(f'Input shape is: {input_shape}')
        print(f'Output features: {out_features}')
        self.d1 = tf.keras.layers.Dense(hidden_size, input_shape=input_shape)
        self.d2 = tf.keras.layers.Dense(out_features)

    def call(self, x): # Input is a weight: (batch, num_layers, seq_len, seq_len)

        # mask = create_padding_mask(x)
        # mask = tf.squeeze(mask, [1, 2]) # (batch_size, seq_len)
        # sum_mask = 1 - mask
        # sum_mask = tf.expand_dims(sum_mask, 2) # (batch_size, seq_len, 1)
        
        weights = self.d2(self.d1(x))

        # Keep only non-padded entries for the sum ahead - TODO 
        weights = weights #* sum_mask

        return tf.nn.softmax(weights)


def create_model(models_dir, load_checkpoint, run_eagerly=True, num_layers=4, seq_len=100):

    model = AdversarialWeightGenenerator(input_shape=(num_layers, seq_len, seq_len), out_features=seq_len)
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