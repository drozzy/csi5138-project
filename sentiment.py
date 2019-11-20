import tensorflow as tf
import matplotlib.pyplot as plt

def sentiment(inp_sentence, encoder, transformer):
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