# VirtuosicAI - fine tune musicVAE
Fine tuning [MusicVAE](https://magenta.tensorflow.org/music-vae) of magenta.

First, install magenta and download the pre-trained model. See [Magenta github](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae) for details.

Crete your dataset by converting midis into single tfrecord:
```
convert_dir_to_note_sequences --input_dir=liszt --output_file=liszt.tfrecord
```

Train:
```powershell
python .\train_prior.py --config=hierdec-mel_16bar --finetune_from_layer=-30 --checkpoint_tar=hierdec-mel_16bar.tar --mode=train --run_dir=model_out --examples_path=priors_example\liszt.tfrecord
```

Generate music with the fine tuned model:
```powershell
music_vae_generate --run_dir=model_out --output_dir=generated --config=hierdec-mel_16bar
``` 