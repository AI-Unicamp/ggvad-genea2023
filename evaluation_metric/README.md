## Fréchet Gesture Distance (FGD)

Scripts to calculate FGD for the GENEA Gesture Generation Challenge 2023.
We followed the FGD implementation in [Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity (ACM TOG, 2020)](https://arxiv.org/abs/2009.02119). It compares distributions of human motion and generated motion to evaluate how the generated motion similar to human motion. Note that FGD only considers main agent motion, not speech and interlocutor context. 

**Disclaimer: Official evaluation of the GENEA Challenge 2023 is subjective human evaluation. We provide this objective metric to help participants evaluate their models faster in the development phase since we found a moderate correlation between FGD and subjective evaluation ratings. Please see [this arXiv paper](https://arxiv.org/pdf/2303.08737.pdf) for more details. Again, a good (low) FGD does not guarantee human preferences.**

The scripts were developed and tested on Ubuntu 22.04, Python 3.6, Pytorch 1.8.0.

### Usage
1. Prepare the pre-trained feature extractor checkpoint (included in this repo; see `output` folder).
2. Convert the generated motion to 3D joint coordinates. You can refer `extract_joint_positions.py` script.
3. Calculate FGD between the sets of natural motion and generated motion
   ```bash
   # make sure you're correctly loading the generated motion
   $ python evaluate_FGD.py
   ``` 

### Training the feature extractor

You can follow the steps below to train the feature extractor.

1. Download GENEA 2023 dataset
2. Convert BVH files to 3D joint coordinates
   ```bash
   $ python extract_joint_positions.py
   ```
3. Train an autoencoder on the train set. You can set `n_frames` in `train_AE.py` to change the number of frames in a sample. 
   ```bash
   $ python train_AE.py
   ```
