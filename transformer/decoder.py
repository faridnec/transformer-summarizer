import tensorflow as tf
from utils import FullyConnected

class DecoderLayer(tf.keras.layers.Layer):
    """
    The decoder layer is composed by two multi-head attention blocks, 
    one that takes the new input and uses self-attention, and the other 
    one that combines it with the output of the encoder, followed by a
    fully connected block. 
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )

        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
            dropout=dropout_rate
        )

        self.ffn = FullyConnected(
            embedding_dim=embedding_dim,
            fully_connected_dim=fully_connected_dim
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass for the Decoder Layer

        Arguments:
          x (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
          enc_output (tf.Tensor): Tensor of shape(batch_size, input_seq_len, fully_connected_dim)
          training (bool): Boolean, set to true to activate the training mode for dropout layers
          look_ahead_mask (tf.Tensor): Boolean mask for the target_input
          padding_mask (tf.Tensor): Boolean mask for the second multihead attention layer

        Returns:
          out3 (tf.Tensor): Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
          attn_weights_block1 (tf.Tensor): Tensor of shape (batch_size, num_heads, target_seq_len, target_seq_len)
          attn_weights_block2 (tf.Tensor): Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)
        """
        
        mult_attn_out1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask, return_attention_scores=True) 
        
        Q1 = self.layernorm1(x + mult_attn_out1)

        mult_attn_out2, attn_weights_block2 = self.mha2(Q1, enc_output, enc_output, padding_mask, return_attention_scores=True)

        mult_attn_out2 = self.layernorm2(mult_attn_out2 + Q1)

        ffn_output = self.ffn(mult_attn_out2)

        ffn_output = self.dropout_ffn(ffn_output, training=training)

        out3 = self.layernorm3(ffn_output + mult_attn_out2)

        return out3, attn_weights_block1, attn_weights_block2