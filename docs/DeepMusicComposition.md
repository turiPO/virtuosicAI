# Deep Music Composition
## Related work
### Passing the Turing test

AI based music composition has been in research since 1980. Classic algorithms, methods from information theory, probabilistic models, chaos theory and genetics algorithms were mainly in use for this task. The first time music generated by DL (deep learning) passed the Turing test was the "Deep Bach" project in 2017, were they composed chorales music based on Bach chorales. [https://arxiv.org/pdf/2108.12290.pdf].

The Deep Bach project mixed classic algorithms as well Bidirectional RNN model in order to "predict" erased arpeggios in Bach chorales (randomly erased arpeggio and ask the model to reconstruct it). In addition, this work was revolutionised in two ways:
1. Similar to image augmentation, the researchers expanded the training dataset by augmenting the music scale of each chorale.
2. Chose the depth of the RNN according the average radius between chord and it's dependents.
[https://arxiv.org/abs/1612.01010]

Meanwhile Magenta project (Acquired by Google) worked on a similar task and publish in 2019. Instead of using recurrent NN (neural network) they used CNN (convolutional NN) to complete partial scores. In order to fix CNN lack of chronologicalization, after the generation the scores run by ML model which check the likelihood of the output based on previous chords and "fix" it to be more chronological. While RNN limited by it's short memory (vanishing/exploding gradient), this CNN based method is more efficient because it can run parallel and proved to be able to capture local structure as well as large scale structure. [https://arxiv.org/pdf/1903.07227.pdf]

Despite the improvement,  chorales composition is an "easy task" because they are very patterned and been written under strict rules. Chorale is a music piece without melody or rhythm which contains four arpeggios in each bar, limited to four voices choir and written under strict harmony rules of the baroque era. Actually, when Bach didn't follow the harmony rules he had been severely criticised in the newspapers.

### More advanced models
The main challenge in the music generation task is to generate longer and better quality music pieces within different genres and thanks to the recent revolutions in NLP and computer vision it's started to happen also in the music field. Most of the music generation breakthroughs of the past years can be divide into two approaches: latent based models and language based model. Latent based models papers are based mainly on VAEs (e.g. MusicVAE, PianoTree, Jukebox), although there are quite successful ones with GANs (e.g. GANSynth, jazzGAN). On the contrasts, music traditionally has been though as a language model, and thus there are state of the art solutions based on transformers (e.g. Music Transformer, Musenet).

The first time VAE has succeeded to catch "long term structure" within 16 bars was MusicVAE, which is able to generate style diverse and good quality melodies. Unlike Vision works with VAE, in order to be able to choose piece length MusicVAE uses Bi-RNN as the decoder and encoder. MusicVAE trained on the Lakh MIDI Dataset and another 1.5 million MIDIs on the web without performance information. In addition, MusicVAE achieves good results when interpolates the latent space between two pieces. [https://arxiv.org/pdf/1803.05428.pdf] 

Transformers based model were first introduced for NLP and were very successful. The main reason for the success was the introduction of the attention layer, which was able to learn the right connection between words in the sentence. Attention layer is ignorant to the order of the words in the sentence, therefore the writers of the paper found a way to encode the position of each word. Unlike RNNs, transformers can be trained with big amount of data, which is more accessible due to the development of the internet and managed to be trained unsupervised for NLP tasks. [https://arxiv.org/pdf/1706.03762]

Magenta team were the first ones to adopt the transformers revolution into music with "Music Transformer". Each "music sentence" is longer than natural language sentence, therefore the paper writers suggested an alternative attention layer algorithm which reduces the space complexity from squared into linear. Music Transformer has been trained on the traditional YAMAHA piano-e-competition dataset as well as all the Youtube piano videos. [https://arxiv.org/pdf/1809.04281.pdf].

Musenet is a transformer based music generation model published by openAI. [https://openai.com/blog/musenet/]

@todo: finish phrase of amny archs

Short pieces:
MusicVAE - https://magenta.tensorflow.org/music-vae
Music Trasformer - magenta
Musenet - GPT2 openai 

Longer pieces:
TransformerVAE
PianoTree
DDPM
Try to find structure with C-RBM.

### Generate a melody when harmony is given
VAE - Generating nontrivial melodies for music as a service 2017
[https://arxiv.org/pdf/1809.07600.pdf]
JazzGAN - Improvising with generative adversarial networks 2018
[https://musicalmetacreation.org/mume2018/proceedings/Trieu.pdf]
BebopNet - jazz improvisiations with LSTM 2020
[https://program.ismir2020.net/static/final_papers/132.pdf]

### Style Transfer
Although the lack of datasets style transfer methods has been used such as tune transfer and instruments addition algorithms, VAE and transfer learning. [https://arxiv.org/pdf/2108.12290.pdf]
The first DNN which succeeded to transfer style of a complete music piece was MIDI-VAE in 2018.   [https://arxiv.org/pdf/1809.07600.pdf]
Transfer learning methods found as very effective for this task, <> find tuned pop generation model into urban music. [????]

Another effective method was by learning the PianoTree VAE model latent vector representation and modify it in a way which changes the style of the music. [https://arxiv.org/pdf/2008.07122.pdf]

GAN based method to mix many genres [https://arxiv.org/pdf/1712.01456.pdf]

## Generation challenges
### Evaluation of music
* Brain EEG(technion jazz paper)
* chord progression histogram
* Train validators (for style transfer)
* Listening tests (like in MuseicVAE)
* creativity
  * Rote Memorization frequencies (RM): Given a specified
  length l, RM measures how frequently the model copies
  note sequences of length l from the corpus. [https://musicalmetacreation.org/mume2018/proceedings/Trieu.pdf]
  * Pitch variation [https://musicalmetacreation.org/mume2018/proceedings/Trieu.pdf]
* 


### Music Features
@todo
A music piece contains a few important elements, which later can be model features:
1. Key - which can be switched between parts of the piece
2. Rhythm
3. Structure - Sonata, perlude, ABABA @todo
4. Melody @todo
5. Texture - @todo
6. Instruments - voice amount, voices range, what kind of instrument etc.
d

# Experiments

## Reconstruction using MuseVAE

Seems by default the model doesn't generate good music. The MIDIs in the website has been cherry picked.

I wil try to solve this problem by creating a prior.






