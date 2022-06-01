import tempfile

import tensorflow.compat.v1 as tf

import tf_slim
import os
import tarfile

from magenta.models.music_vae import data
from magenta.models.music_vae import music_vae_train as vae_train

from model import prior_hierdec_mel_16bar
from cmd_args import FLAGS


def split_freeze_train(var_substring):
    model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    tuning = [v for v in model_vars if var_substring in v.name]
    restoring = list(set(model_vars) - set(tuning))
    return restoring, tuning


def set_tuning_vars(vars):
    tf.get_default_graph().clear_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    return [tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, v) for v in vars]


def load_checkpoint(checkpoint_tar, vars2restore):
    return tf_slim.assign_from_checkpoint(checkpoint_tar, vars2restore)


def load_tar_checkpoint(checkpoint_tar, vars2restore):
    with tempfile.TemporaryDirectory() as temp_dir:
        tar = tarfile.open(checkpoint_tar)
        tar.extractall(temp_dir)
        # Assume only a single checkpoint is in the directory.
        for name in tar.getnames():
            if name.endswith('.index'):
                ckpt_name = os.path.join(temp_dir, name[0:-6])
                return load_checkpoint(ckpt_name, vars2restore)


def train(train_dir,
          config,
          dataset_fn,
          checkpoint_tar,
          checkpoints_to_keep=5,
          keep_checkpoint_every_n_hours=1,
          num_steps=None,
          master='',
          num_sync_workers=0,
          num_ps_tasks=0,
          task=0):
    # var_train_pattern = ["latent", ]
    var_train_pattern = ["encoder", ]
    tf.gfile.MakeDirs(train_dir)
    is_chief = (task == 0)
    if is_chief:
        vae_train._trial_summary(
            config.hparams, config.train_examples_path or config.tfds_name,
            train_dir)
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(
                num_ps_tasks, merge_devices=True)):

            model = config.model
            model.build(config.hparams,
                        config.data_converter.output_depth,
                        is_training=True)

            optimizer = model.train(**vae_train._get_input_tensors(dataset_fn(), config))

            # set which vars in the pre-trained model to train
            restoring, tuning = split_freeze_train(var_train_pattern[0])
            set_tuning_vars(tuning)

            hooks = []
            if num_sync_workers:
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer,
                    num_sync_workers)
                hooks.append(optimizer.make_session_run_hook(is_chief))

            grads, var_list = list(zip(*optimizer.compute_gradients(model.loss)))
            global_norm = tf.global_norm(grads)
            tf.summary.scalar('global_norm', global_norm)

            if config.hparams.clip_mode == 'value':
                g = config.hparams.grad_clip
                clipped_grads = [tf.clip_by_value(grad, -g, g) for grad in grads]
            elif config.hparams.clip_mode == 'global_norm':
                clipped_grads = tf.cond(
                    global_norm < config.hparams.grad_norm_clip_to_zero,
                    lambda: tf.clip_by_global_norm(  # pylint:disable=g-long-lambda
                        grads, config.hparams.grad_clip, use_norm=global_norm)[0],
                    lambda: [tf.zeros(tf.shape(g)) for g in grads])
            else:
                raise ValueError(
                    'Unknown clip_mode: {}'.format(config.hparams.clip_mode))
            train_op = optimizer.apply_gradients(
                list(zip(clipped_grads, var_list)),
                global_step=model.global_step,
                name='train_step')

            logging_dict = {'global_step': model.global_step,
                            'loss': model.loss}

            hooks.append(tf.train.LoggingTensorHook(logging_dict, every_n_iter=100))
            if num_steps:
                hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

            # define fine tuning
            vars2restore = tf_slim.get_variables_to_restore(
                include=[v.name for v in restoring])

            pretrained_operation, pretrained_weights = load_tar_checkpoint(checkpoint_tar, vars2restore)

            def init_pre_fn(scaffold, tf_session):
                tf_session.run(pretrained_operation, pretrained_weights)

            scaffold = tf.train.Scaffold(
                init_fn=init_pre_fn,
                saver=tf.train.Saver(
                    max_to_keep=checkpoints_to_keep,
                    keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))
            ###################

            tf_slim.training.train(
                train_op=train_op,
                logdir=train_dir,
                scaffold=scaffold,
                hooks=hooks,
                save_checkpoint_secs=60,
                master=master,
                is_chief=is_chief)


def main_defaults(extra_argv):
    run_dir = os.path.expanduser(FLAGS.run_dir)
    train_dir = os.path.join(run_dir, 'train')

    config_update_map = {}
    if FLAGS.examples_path:
        config_update_map['%s_examples_path' % FLAGS.mode] = os.path.expanduser(FLAGS.examples_path)

    prior_hierdec_mel_16bar_conf = vae_train.configs.update_config(prior_hierdec_mel_16bar, config_update_map)

    def dataset_fn():
        return data.get_dataset(
            prior_hierdec_mel_16bar_conf,
            tf_file_reader=tf.data.TFRecordDataset,
            is_training=FLAGS.mode == 'train',
            cache_dataset=True)

    train(
        train_dir=train_dir,
        config=prior_hierdec_mel_16bar_conf,
        dataset_fn=dataset_fn,
        num_steps=10,
        checkpoint_tar=FLAGS.checkpoint_tar
    )


if __name__ == "__main__":
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(FLAGS.log)
    tf.app.run(main_defaults)
