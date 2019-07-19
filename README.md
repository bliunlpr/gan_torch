# Boosting Noise Robustness of Acoustical Model via Deep Adversarial Training

This is an implementation of our ICASSP2018 Best Student Award paper "Boosting Noise Robustness of Acoustical Model via Deep Adversarial Training" 
on Python 3, PyTorch. We propose an adversarial training method to directly boost noise robustness of acoustic model. The joint 
optimization of generator, discriminator and AM concentrates the strengths of both GAN and AM for speech recognition. 

# Requirements
kaldi, Python 3.5, PyTorch 0.4.0.

# Data
### Chime 4
You can download [Chime 4](http://spandh.dcs.shef.ac.uk/chime_challenge/chime2016/) to run the code.

### Your Own Dataset
You need build train, dev and test directory. Each has ```feats.scp``` ```utt2spk``` ```spk2utt``` and ```text```. 

# Model

The model consists of a generator(G), a discriminator (D) and a classifier(C). The generator G performs the speech enhancement. It transforms the noisy speech signals into the enhanced version. The discriminator D aims to distinguish between the enhanced signals and clean ones. The classifier C classifies senones by features derivated from G. 

<div align="center">
<img src="https://github.com/bliunlpr/gan_torch/blob/master/fig/adn.jpg"  height="400" width="495">
</div>


# Training

### Baseline(cross-entropy loss)
The baseline system is bulit following the Kaldi chime4/s5 1ch recipe. The acoustic model is a DNN with 7 hidden layers. 
After RBM pre-training, the model is trained by minimizing the cross-entropy loss.

```
sh run.sh --stage 2 --seq_training false --model_type lstm --train_type  base_attention | last_state | base_attention ```   
## CNN does not use attention but avgpool.
sh run.sh --stage 2 --seq_training false --model_type cnn 
```

### Deep Adversarial Training
```
sh run.sh --stage 2 --seq_training false --model_type lstm --train_type  base_attention | last_state | base_attention ```   
## CNN does not use attention but avgpool.
sh run.sh --stage 2 --seq_training false --model_type cnn 
```

# Decoding
We use the Kaldi WFST decoder for decoding in all the experiments.
```
sh kaldi/decode.sh  

```

# Results
<div align="center">
<img src="https://github.com/bliunlpr/gan_torch/blob/master/fig/result.png"  >
</div>
