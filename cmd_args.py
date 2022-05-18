import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_steps', 1000,
    'Number of training steps or `None` for infinite.'
)
flags.DEFINE_string(
    'pretrained_path', '',
    'Where pretrained checkpoint is located'
)
flags.DEFINE_string(
    'run_dir', None,
    'Path where checkpoints and summary events will be located during '
    'training and evaluation. Separate subdirectories `train` and `eval` '
    'will be created within this directory.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.'
)