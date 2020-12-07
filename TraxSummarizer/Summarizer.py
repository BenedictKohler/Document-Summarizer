# Imports                                                                                                                                                                                                          import sys                                                                                                                                                                                                         import string                                                                                                                                                                                                      import os                                                                                                                                                                                                          import numpy as np
import textwrap
wrapper = textwrap.TextWrapper(width=70)
import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp
import tensorflow as tf

# Get the training data from tensorflow datasets
train_stream_fn = trax.data.TFDS('cnn_dailymail', keys=('article', 'highlights'), train=True)
eval_stream_fn = trax.data.TFDS('cnn_dailymail', keys=('article', 'highlights'), train=False)

# Special tokens
SEP = 0 # Padding or separator token
EOS = 1 # End of sentence token

# Convert each sentence to a list of numbers
def tokenize(input_str, EOS=1):
    inputs =  next(trax.data.tokenize(iter([input_str]),
        vocab_dir='.', vocab_file='subwords.txt'))
    return list(inputs) + [EOS]

# Convert a list of numbers back into sentences
def detokenize(integers):
    s = trax.data.detokenize(integers, vocab_dir='.', vocab_file='subwords.txt')
    return wrapper.fill(s)

# Format the data in the way required by the model (multiple tokenized lists with separation tokens)
def preprocess(stream) :
    for (article, summary) in stream :
        joint = np.array(list(article) + [EOS, SEP] + list(summary) + [EOS])
        mask = [0] * (len(list(article)) + 2) + [1] * (len(list(summary)) + 1)
        yield joint, joint, np.array(mask)

# This compines previous functions and takes in the data as input before transforming it
input_pipeline = trax.data.Serial(trax.data.Tokenize(vocab_dir='.', vocab_file='subwords.txt'), preprocess,
        trax.data.FilterByLength(2048))

# Pass in required functions to get data
train_stream = input_pipeline(train_stream_fn())
eval_stream = input_pipeline(eval_stream_fn())

# Train stream needs to be an iterator
train_input, train_target, train_mask = next(train_stream)

# This puts summaries of similar lengths together
boundaries =  [128, 256,  512, 1024]
batch_sizes = [16,    8,    4,    2, 1]

train_batch_stream = trax.data.BucketByLength(boundaries, batch_sizes)(train_stream)
eval_batch_stream = trax.data.BucketByLength(boundaries, batch_sizes)(eval_stream)

input_batch, _, mask_batch = next(train_batch_stream)

# Regular tensor creation
def create_tensor(t) :
    return jnp.array(t)

# This function is a way of 'shining a light' on certain words during training, so the model is able to understand more easily
def DotProductAttention(query, key, value, mask) :
    assert query.shape[-1] == key.shape[-1] == value.shape[-1]

    depth = query.shape[-1]

    dots = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / jnp.sqrt(depth) # Part of dot product formula

    # Apply mask
    if mask is not None :
        dots = jnp.where(mask, dots, jnp.full_like(dots, -1e9))

    # Rest of dot product attention formula
    logsumexp = trax.fastmath.logsumexp(dots, axis=-1, keepdims=True)

    dots = jnp.exp(dots - logsumexp)

    attention = jnp.matmul(dots, value)

    return attention

# Function used to apply dot product attention give query, key, and value
def dot_product_self_attention(q, k, v) :
    mask_size = q.shape[-2]
    mask = jnp.tril(jnp.ones((1, mask_size, mask_size), dtype=jnp.bool_), k=0)
    return DotProductAttention(q, k, v, mask)
   
    # Need to define a closure which is passed into layers in the model
def compute_attention_heads_closure(n_heads, d_head) :
    def compute_attention_heads(x) :
        # Data reshaping for the model layers
        batch_size = x.shape[0]
        seqlen = x.shape[1]
        x = jnp.reshape(x, (batch_size, seqlen, n_heads, d_head))
        x = jnp.transpose(x, (0, 2, 1, 3))
        x = jnp.reshape(x, (-1, seqlen, d_head))
        return x
    return compute_attention_heads

# Need to define a closure which is passed into layers in the model
def compute_attention_output_closure(n_heads, d_head) :
    def compute_attention_output(x) :
        # Data reshaping for the model layers
        seqlen = x.shape[1]
        x = jnp.reshape(x, ( -1, n_heads, seqlen, d_head))
        x = jnp.transpose(x, ( 0, 2, 1 , 3))
        return jnp.reshape(x, (-1, seqlen, n_heads * d_head))
    return compute_attention_output

# This creates one of the residual blocks within the model
# Variables passed in refer to the depth of the layers and number of parameters in each layer
def CausalAttention(d_feature, n_heads,
        compute_attention_heads_closure=compute_attention_heads_closure,
        dot_product_self_attention=dot_product_self_attention,
        compute_attention_output_closure=compute_attention_output_closure,
        mode='train'):

    assert d_feature % n_heads == 0
    d_head = d_feature // n_heads

    ComputeAttentionHeads = tl.Fn('AttnHeads', compute_attention_heads_closure(n_heads, d_head), n_out=1)

    # This returns a 3-layer deep neural network which uses attention in order to find most relevant words quicker
    return tl.Serial(
            tl.Branch(
                [tl.Dense(d_feature), ComputeAttentionHeads], # queries
                [tl.Dense(d_feature), ComputeAttentionHeads], # keys
                [tl.Dense(d_feature), ComputeAttentionHeads], # values
                ),
                tl.Fn('DotProductAttn', dot_product_self_attention, n_out=1), # takes QKV
                tl.Fn('AttnOutput', compute_attention_output_closure(n_heads, d_head), n_out=1), # to allow for parallel
                tl.Dense(d_feature)
                )

# This is another residual block that takes a context vector computed in the encoder block and transforms it to summary which then becomes detokenized
def DecoderBlock(d_model, d_ff, n_heads, dropout, mode, ff_activation) :
    causal_attention = CausalAttention(d_model, n_heads=n_heads, mode=mode)

    # Shallow neural netowork with layer normalization and dropout to avoid overfitting
    feed_forward = [
            tl.LayerNorm(),
            tl.Dense(d_ff),
            ff_activation(),
            tl.Dropout(rate=dropout, mode=mode),
            tl.Dense(d_model),
            tl.Dropout(rate=dropout,mode=mode)
    ]
    # This creates the residual network which is used to ensure the model is able to understand complex relationships as well as simple
    # Sometimes when models become too deep they don't learn properly which is why this is needed
    return [
            tl.Residual(
                tl.LayerNorm(),
                causal_attention,
                tl.Dropout(rate=dropout, mode=mode)
                ),
            tl.Residual(
                feed_forward
                ),
            ]

# This sets up the archtecture of our model network and defualt activation layers
def TransformerLM(vocab_size=33300, d_model=512, d_ff=2048, n_layers=6, n_heads=8, dropout=0.1,
        max_len=4096, mode='train', ff_activation=tl.Relu) :
    positional_encoder = [
            tl.Embedding(vocab_size, d_model),
            tl.Dropout(rate=dropout, mode=mode),
            tl.PositionalEncoding(max_len=max_len, mode=mode)]
    decoder_blocks = [
            DecoderBlock(d_model, d_ff, n_heads, dropout, mode, ff_activation) for _ in range(n_layers)]

    # Put the different blocks and functions together to be executed like in a stack
    return tl.Serial(
            tl.ShiftRight(mode=mode),
            positional_encoder,
            decoder_blocks,
            tl.LayerNorm(),
            tl.Dense(vocab_size),
            tl.LogSoftmax(),
            )

from trax.supervised import training

# This sets up the actual training loop where input is passed in and the model is developed and then saved to an output direct where it can be trained later on
def training_loop(TransformerLM, train_gen, eval_gen, output_dir = "./model") :
    output_dir = os.path.expanduser(output_dir)
    lr_schedule = trax.lr.warmup_and_rsqrt_decay(n_warmup_steps=1000, max_value=0.01)

    # This sets up loss function and our adam optimizer used to fit the data efficiently
    train_task = training.TrainTask(
            labeled_data=train_gen,
            loss_layer=tl.CrossEntropyLoss(),
            optimizer=trax.optimizers.Adam(0.01),
            lr_schedule=lr_schedule,
            n_steps_per_checkpoint=10
            )
    # We evaluate on a different dataset to ensure no overfitting
    eval_task = training.EvalTask(
            labeled_data=eval_gen,
            metrics=[tl.CrossEntropyLoss(), tl.Accuracy()]
    )

    loop = training.Loop(TransformerLM(d_model=512, d_ff=2048, n_layers=6, n_heads=8, mode='train'),
            train_task,
            eval_tasks=[eval_task],
            output_dir=output_dir)

    return loop

loop = training_loop(TransformerLM, train_batch_stream, eval_batch_stream) # Set up training loop
loop.run(260) # Number of epochs to run

model = TransformerLM(mode='eval')
model.init_from_file('./model/model.pkl.gz', weights_only=True) # If we have a saved model, use it to further develop weights

# This is used for detokinizing the output sentence
def next_symbol(cur_output_tokens, model):
    # We choose one work at a time until the model computes and EOS tag
    token_length = len(cur_output_tokens)
    padded_length = 2**int(np.ceil(np.log2(token_length + 1)))
    padded = cur_output_tokens + [0] * (padded_length - token_length)
    padded_with_batch = np.array(padded)[None, :]
    output, _ = model((padded_with_batch, padded_with_batch))
    log_probs = output[0, token_length, :]

    return int(np.argmax(log_probs)) # Return the word with highest predicted probability based off of previous words chosen and input sentence

def greedy_decode(input_sentence, model) :
    cur_output_tokens = tokenize(input_sentence) + [0]
    generated_output = []
    cur_output = 0
    EOS = 1
    length = 0

    # While the model hasn't predicted an end of summary tag we detokinize output and coninually choose another word which the model predicts to have the highest probability of occuring
    while cur_output != EOS and length < 60 :
        cur_output = next_symbol(cur_output_tokens, model)
        cur_output_tokens.append(cur_output)
        generated_output.append(cur_output)
        print(detokenize(generated_output))
        length += 1

    return detokenize(generated_output)

article = "It’s the posing craze sweeping the U.S. after being brought to fame by skier Lindsey Vonn, soccer star Omar Cummings, baseball player Albert Pujols - and even Republican politician Rick Perry. But now four students at Riverhead High School on Long Island, New York, have been suspended for dropping to a knee and taking up a prayer pose to mimic Denver Broncos quarterback Tim Tebow. Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll were all suspended for one day because the ‘Tebowing’ craze was blocking the hallway and presenting a safety hazard to students. Scroll down for video. Banned: Jordan Fulcoly, Wayne Drexel, Tyler Carroll and Connor Carroll (all pictured left) were all suspended for one day by Riverhead High School on Long Island, New York, for their tribute to Broncos quarterback Tim Tebow. Issue: Four of the pupils were suspended for one day because they allegedly did not heed to warnings that the 'Tebowing' craze at the school was blocking the hallway and presenting a safety hazard to students."
print(wrapper.fill(article), '\n')
print(greedy_decode(article, model)) # Summarize an article
