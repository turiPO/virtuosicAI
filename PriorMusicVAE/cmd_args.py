import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'checkpoint_tar', 'hierdec-mel_16bar.tar',
    'Where pretrained checkpoint is located'
)

flags.DEFINE_integer(
    'finetune_from_layer', 0,
    'From which layer to fine tune. e.g. -1 works for the last one.'
)

flags.DEFINE_string(
    'finetune_component', '',
    'Which component to fine tune. e.g. "encoder", "decoder". default is all component.'
)