# Boosting Noise Robustness of Acoustical Model via Deep Adversarial Training

This is an implementation of our ICASSP2018 Best Student Award paper "Boosting Noise Robustness of Acoustical Model via Deep Adversarial Training" 
on Python 3, PyTorch. We propose an adversarial training method to directly boost noise robustness of acoustic model. The joint 
optimization of generator, discriminator and AM concentrates the strengths of both GAN and AM for speech recognition. 

# Requirements
Python 3.5, PyTorch 0.4.0.

# Data
## Chime 4
You can download [Chime 4](http://spandh.dcs.shef.ac.uk/chime_challenge/chime2016/) to run the code.

## Your Own Dataset
You need build train, dev and test directory. Each has ```feats.scp``` ```utt2spk``` ```spk2utt``` and ```text```. 

<div align="center">
<img src="https://github.com/bliunlpr/gan_torch/blob/master/fig/adn.jpg"  height="400" width="495">
</div>
