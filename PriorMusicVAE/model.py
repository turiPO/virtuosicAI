from magenta.contrib import training as contrib_training
from magenta.models.music_vae import data
from magenta.models.music_vae import data_hierarchical
from magenta.models.music_vae import lstm_models
from magenta.models.music_vae import configs
from magenta.models.music_vae.base_model import MusicVAE
import note_seq

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import tf_slim

ds = tfp.distributions
BATCH_SIZE = 2


class MusicVAEPrior(MusicVAE):
    def build(self, hparams, output_depth, is_training=True):
        """Builds encoder and decoder.

        Must be called within a graph.

        Args:
        hparams: An HParams object containing model hyperparameters. See
            `get_default_hparams` below for required values.
        output_depth: Size of final output dimension.
        is_training: Whether or not the model will be used for training. - IGNORING
        """
        tf.logging.info('Building MusicVAE model with %s, %s, and hparams:\n%s',
                        self.encoder.__class__.__name__,
                        self.decoder.__class__.__name__, hparams.values())
        self.global_step = tf.train.get_or_create_global_step()
        self._hparams = hparams
        # Fine tune only the encoder in order to generate specific style
        self._encoder.build(hparams, is_training=True)
        self._decoder.build(hparams, output_depth, is_training=False)


prior_hierdec_mel_16bar = configs.Config(
    model=MusicVAEPrior(
        lstm_models.BidirectionalLstmEncoder(),
        lstm_models.HierarchicalLstmDecoder(
            lstm_models.CategoricalLstmDecoder(),
            level_lengths=[16, 16],
            disable_autoregression=True)),
    hparams=configs.merge_hparams(
        lstm_models.get_default_hparams(),
        configs.HParams(
            batch_size=BATCH_SIZE,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[1024, 1024],
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=configs.mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)
