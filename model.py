import tensorflow as tf
import numpy as np
# from sparsemax2 import sparsemax
from entfox import entmax15
## Positional Encoding 

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

## Masking

def create_padding_mask(seq):
    """ 
    Mask all the pad tokens in the batch of sequence. It ensures that the
    model does not treat padding as the input. The mask indicates where pad value
    `0` is present: it outputs a `1` at those locations, and a `0` otherwise. 
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding
    # to the attention logits.
    seq = tf.expand_dims(seq, 1)  # (batch_size, 1, seq_len)
    return tf.expand_dims(seq, 1) # (batch_size, 1, 1, seq_len)

## Scaled Dot Product Attention

def scaled_dot_product_attention(q, k, v, mask, use_sparsemax=False):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
              to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    if use_sparsemax:
        attention_weights = entmax15(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    else:    
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

## Multi-Head Attention
class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Each multi-head attention block gets three inputs; Q (query), K (key), V
    (value). These are put through linear (Dense) layers and split up into
    multiple heads. 

    The `scaled_dot_product_attention` defined above is applied to each head
    (broadcasted for efficiency). An appropriate mask must be used in the
    attention step.  The attention output for each head is then concatenated
    (using `tf.transpose`, and `tf.reshape`) and put through a final `Dense`
    layer.

    Instead of one single attention head, Q, K, and V are split into multiple
    heads because it allows the model to jointly attend to information at
    different positions from different representational spaces. After the split
    each head has a reduced dimensionality, so the total computation cost is the
    same as a single head attention with full dimensionality.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask, custom_k=None, use_sparsemax=False):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        if custom_k is not None:
            # print("Custom K")
            k = custom_k
        else:
            # print("Non custom k")
            k = self.wk(k)  # (batch_size, seq_len, d_model)

        unsplit_k = k
        
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # print(f'k shape: {tf.shape(k)}')
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, use_sparsemax)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights, unsplit_k

## Point Wise Feed Forward Network

def point_wise_feed_forward_network(d_model, dff):
    """
    Point wise feed forward network consists of two fully-connected layers with a ReLU activation in between.
    """
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer

    Each encoder layer consists of sublayers:

    1. Multi-head attention (with padding mask) 
    2. Point wise feed forward networks. 

    Each of these sublayers has a residual connection around it followed by a
    layer normalization. Residual connections help in avoiding the vanishing
    gradient problem in deep networks.

    The output of each sublayer is `LayerNorm(x + Sublayer(x))`. The normalization
    is done on the `d_model` (last) axis. There are N encoder layers in the
    transformer.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, custom_k=None, use_sparsemax=False):
        attn_output, attn_weights, k = self.mha(x, x, x, mask, custom_k=custom_k, use_sparsemax=use_sparsemax)  # (batch_size, input_seq_len, d_model)
        # attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2, attn_weights, k


class Encoder(tf.keras.layers.Layer):
    """
    Encoder

    The `Encoder` consists of:
    1.   Input Embedding
    2.   Positional Encoding
    3.   N encoder layers

    The input is put through an embedding which is summed with the positional
    encoding. The output of this summation is the input to the encoder layers. The
    output of the encoder is the input to the decoder. 
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1, use_positional_encoding=True):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, custom_k=None, use_sparsemax=False):

        seq_len = tf.shape(x)[1]
        attention_weights = []
        ks = []

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if self.use_positional_encoding:
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        # custom_k - (num_layers, batch, ...)
        for i in range(self.num_layers):
            
            x, layer_weights, k = self.enc_layers[i](x, training, mask, custom_k=custom_k[i] if custom_k is not None else None, use_sparsemax=use_sparsemax) 
            ks.append(k)
            attention_weights.append(layer_weights)
            
        attention_weights = tf.stack(attention_weights)                        # (num_layers, batch, ...)
        attention_weights = tf.transpose(attention_weights, perm=[1, 0, 2, 3, 4]) 
        ks = tf.stack(ks)  # (num_layers, batch, seq_len, d_model)
        ks = tf.transpose(ks, perm=[1, 0, 2, 3]) 
        

        return (x,                   # (batch, input_seq_len, d_model)
               attention_weights,    # (batch, num_layers, seq_len, seq_len)
               ks)                    # (batch, num_layers, seq_len, d_model)

class TransformerEncoderClassifier(tf.keras.Model):
    """ Transformer Encoder Classifier consists of the encoder, and a final
    linear layer. The output of the encoder is the input to the linear layer and
    its output is returned. 
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate=0.1, 
            use_positional_encoding=True):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, pe_input, rate, 
                               use_positional_encoding=use_positional_encoding)
        self.final_layer = tf.keras.layers.Dense(1)
        
    def call(self, inp, training, custom_k=None, use_sparsemax=False):
        # custom_k == (num_layers, batch, seq_len, d_model)
        # enc_padding_mask.shape == (batch_size, 1, 1, seq_len)  - Andriy
        if custom_k is not None:
            custom_k = tf.transpose(custom_k, perm=[1, 0, 2, 3])  # (num_layers, batch, seq_len, d_model)

        enc_padding_mask = create_padding_mask(inp)
        enc_output, attention_weights, k = self.encoder(inp, training, enc_padding_mask, custom_k=custom_k, use_sparsemax=use_sparsemax)  # (batch_size, seq_len, d_model)
        # enc_output.shape == (batch_size, seq_len, d_model)
        if enc_padding_mask is not None:
            enc_padding_mask = tf.squeeze(enc_padding_mask, [1, 2]) # (batch_size, seq_len)
            sum_mask = 1 - enc_padding_mask
            sum_mask = tf.expand_dims(sum_mask, 2) # (batch_size, seq_len, 1)
            
            # Keep only non-padded entries for the sum ahead
            enc_output = enc_output * sum_mask
        enc_output = tf.reduce_sum(enc_output, 1)
        final_output = self.final_layer(enc_output)
        final_output = tf.squeeze(final_output, 1)     
        return (final_output,   # (batch, seq_len, d_model)
            attention_weights,  # (batch, num_layers, seq_len, seq_len)
            k)                  # (batch, num_layers, seq_len, d_model)