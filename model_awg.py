# TODO:
# Create adversarial weight generator model
# Test it 

class AdversarialWeightGenenerator(tf.keras.Model):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.d1 = tf.keras.layers.Dense(64, input_shape=(in_features))
        self.d2 = tf.keras.layers.Dense(out_features)

    def call(self, x, mask):

        mask = create_padding_mask(x)
        mask = tf.squeeze(mask, [1, 2]) # (batch_size, seq_len)
        sum_mask = 1 - mask
        sum_mask = tf.expand_dims(sum_mask, 2) # (batch_size, seq_len, 1)
        
        weights = self.d2(self.d1(x))

        # Keep only non-padded entries for the sum ahead - TODO 
        weights = weights * sum_mask
        return weights


def create_model(models_dir, load_checkpoint, run_eagerly):
    model = AdversarialWeightGenenerator()
    model.run_eagerly = run_eagerly

    optimizer = tf.keras.optimizers.Adam()    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)    
    manager = tf.train.CheckpointManager(checkpoint, models_dir, max_to_keep=3)

    if load_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            print("Loaded previously trained model.")            
        else:
            print("No previous model found. Training from scratch.")
    else:
        print("Not loading previously trained model. Training from scratch.")

    return model, optimizer, checkpoint, manager