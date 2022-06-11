# VirtuosicAI

From midis folder into model input file:
```
convert_dir_to_note_sequences --input_dir=liszt --output_file=liszt.tfrecord
```

Train prior:
```powershell
python .\train_prior.py --config=hierdec-mel_16bar --checkpoint_tar=hierdec-mel_16bar.tar --mode=train --run_dir=tmp1 --examples_path=priors_example\liszt.tfrecord
```

Generate music:
```powershell
music_vae_generate --run_dir=tmp1 --output_dir=generated --config=hierdec-mel_16bar
``` 