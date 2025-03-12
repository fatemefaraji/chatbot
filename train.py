import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import numpy as np

MAX_SEQUENCE_LENGTH = 128
BATCH_SIZE = 32
BUFFER_SIZE = 10000
VOCAB_SIZE = 5000
D_MODEL = 512
DROPOUT_RATE = 0.1
DFF = 2048
NUM_HEADS = 8
NUM_LAYERS = 6

class TokenEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)

    def call(self, x):
        return self.emb(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model
        })
        return config

class LSHSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.wq = layers.Dense(d_model // 2)
        self.wk = layers.Dense(d_model // 2)
        self.wv = layers.Dense(d_model // 2)
        self.dense = layers.Dense(d_model // 2)

    def call(self, x):
        q, k, v = tf.split(x, num_or_size_splits=3, axis=-1)
        q, k, v = self.wq(q), self.wk(k), self.wv(v)
        attention_weights = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(tf.shape(k)[-1], tf.float32)), axis=-1)
        return self.dense(tf.matmul(attention_weights, v))

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads
        })
        return config

class FeedForward(layers.Layer):
    def __init__(self, d_model, dff, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dff = dff
        self.dense1 = layers.Dense(dff, activation='gelu')
        self.dense2 = layers.Dense(d_model // 2)

    def call(self, x):
        return self.dense2(self.dense1(x))

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "dff": self.dff
        })
        return config

class ReversibleResidualLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.f = LSHSelfAttention(d_model, num_heads)
        self.g = FeedForward(d_model, dff)

    def call(self, inputs):
        x1, x2 = tf.split(inputs, num_or_size_splits=2, axis=-1)
        y1 = x1 + self.f(tf.concat([x2, x2, x2], axis=-1))
        y2 = x2 + self.g(y1)
        return tf.concat([y1, y2], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff
        })
        return config

class ReformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.residual = ReversibleResidualLayer(d_model, num_heads, dff)

    def call(self, x):
        return self.residual(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff
        })
        return config

class Reformer(keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, dff, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.reformer_blocks = [ReformerBlock(d_model, num_heads, dff) 
                              for _ in range(num_layers)]
        self.final_layer = layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding(x)
        for layer in self.reformer_blocks:
            x = layer(x)
        return self.final_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "num_layers": self.num_layers
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

data_path = "/content/drive/MyDrive/Week 4/data/tokenized_training_data.json"
with open(data_path, "r", encoding="utf-8") as file:
    tokenized_data = json.load(file)

input_sequences = [item['input_ids'] for item in tokenized_data]
target_sequences = [item['output_ids'] for item in tokenized_data]

input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    input_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
target_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    target_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

reformer = Reformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, DFF, NUM_LAYERS)
reformer.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = reformer.fit(
    dataset, 
    epochs=10,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            'reformer_checkpoint_{epoch}.keras',
            save_best_only=True,
            monitor='loss'
        ),
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)


reformer.save('reformer_final_model.keras')

custom_objects = {
    "Reformer": Reformer,
    "ReformerBlock": ReformerBlock,
    "ReversibleResidualLayer": ReversibleResidualLayer,
    "LSHSelfAttention": LSHSelfAttention,
    "FeedForward": FeedForward,
    "TokenEmbedding": TokenEmbedding
}


try:
    loaded_model = keras.models.load_model('reformer_final_model.keras', 
                                         custom_objects=custom_objects)
    print("model successfully saved and loaded!")
except Exception as e:
    print(f"rrror loading model: {e}")