import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'checkpoint_tar', 'hierdec-mel_16bar.tar',
    'Where pretrained checkpoint is located'
)