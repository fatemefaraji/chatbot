import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define Embedding Layer
class TokenEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)
    
    def call(self, x):
        return self.emb(x)

# Define Reversible Residual Layer
class ReversibleResidualLayer(layers.Layer):
    def __init__(self, f, g):
        super(ReversibleResidualLayer, self).__init__()
        self.f = f
        self.g = g
    
    def call(self, inputs):
        x1, x2 = tf.split(inputs, num_or_size_splits=2, axis=-1)
        y1 = x1 + self.f(tf.concat([x2, x2, x2], axis=-1))  # Corrected concatenation
        y2 = x2 + self.g(y1)
        return tf.concat([y1, y2], axis=-1)

# Define LSH Self-Attention Layer
class LSHSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(LSHSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // (2 * num_heads)

        self.wq = layers.Dense(d_model // 2)
        self.wk = layers.Dense(d_model // 2)
        self.wv = layers.Dense(d_model // 2)
        self.dense = layers.Dense(d_model // 2)

    def call(self, x):  # Input is now x
        batch_size = tf.shape(x)[0]
        v = x[:, :, :self.d_model // 2]
        k = x[:, :, self.d_model // 4: 3 * self.d_model // 4] # Overlapping slices for k
        q = x[:, :, self.d_model // 2:]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)
        return self.dense(output)

# Define Feed-Forward Layer
class FeedForward(layers.Layer):
    def __init__(self, d_model, dff):
        super(FeedForward, self).__init__()
        self.dense1 = layers.Dense(dff, activation='gelu')
        self.dense2 = layers.Dense(d_model // 2)

    def call(self, x):
        return self.dense2(self.dense1(x))

# Define Reformer Block
class ReformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(ReformerBlock, self).__init__()
        self.attn = LSHSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        self.residual = ReversibleResidualLayer(self.attn, self.ffn)

    def call(self, x):
        return self.residual(x)

# Define Reformer Model
class Reformer(keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, dff, num_layers):
        super(Reformer, self).__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.reformer_blocks = [ReformerBlock(d_model, num_heads, dff) for _ in range(num_layers)]
        self.final_layer = layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding(x)
        for layer in self.reformer_blocks:
            x = layer(x)
        return self.final_layer(x)

# Model Parameters
VOCAB_SIZE = 10000
D_MODEL = 512
NUM_HEADS = 8
DFF = 2048
NUM_LAYERS = 6

# Instantiate Model
reformer = Reformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, DFF, NUM_LAYERS)

# Compile Model
reformer.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))


# Create dummy input and call the model to build it
sample_input = np.random.randint(0, VOCAB_SIZE, size=(1, 128))
reformer(sample_input)

# Now you can call summary
reformer.summary()