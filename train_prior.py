import tensorflow.compat.v1 as tf
import tensorflow.contrib.training
import tensorflow.contrib.framework
import tf_slim
import os

from magenta.models.music_vae import data
from magenta.models.music_vae import music_vae_train as vae_train

from model import prior_hierdec_mel_16bar
from cmd_args import FLAGS

PRE_TRAINED_PATH = "hierdec-mel_16bar.tar"


def vars_freezer():
    pass  # todo


def train(train_dir,
          config,
          dataset_fn,
          checkpoints_to_keep=5,
          keep_checkpoint_every_n_hours=1,
          num_steps=None,
          master='',
          num_sync_workers=0,
          num_ps_tasks=0,
          task=0):
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

            ### FINE TUNING ###
            variables_to_restore = tensorflow.contrib.framework.get_variables_to_restore(
                include=[v.name for v in vars_freezer])

            init_assign_op, init_feed_dict = tensorflow.contrib.framework.assign_from_checkpoint(
                config.pretrained_path, variables_to_restore)

            def init_assign_fn(scaffold, sess):
                sess.run(init_assign_op, init_feed_dict)

            scaffold = tf.train.Scaffold(
                init_fn=init_assign_fn,
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


def main_defaults():
    run_dir = os.path.expanduser(FLAGS.run_dir)
    train_dir = os.path.join(run_dir, 'train')

    def dataset_fn():
        return data.get_dataset(
            prior_hierdec_mel_16bar,
            tf_file_reader=tf.data.TFRecordDataset,
            is_training=True,
            cache_dataset=True)

    vae_train.train(
        train_dir=train_dir,
        config=prior_hierdec_mel_16bar,
        dataset_fn=dataset_fn,
        num_steps=10
    )


if __name__ == "__main__":
    tf.logging.set_verbosity(FLAGS.log)
    tf.app.run(main_defaults)
