# DeepLab v3
## General
This is a PyTorch implementation of [DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611) for semantic image segmentation. Currently, I use Resnet as backbone and train the model using the [Cambridge-driving Labeled Video Database (CamVid) dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid).

## Files

```
.
├── data
│   └── CamVid
│       ├── test
│       ├── test_labels
│       ├── train
│       ├── train_labels
│       ├── val
│       └── val_labels
├── output
│   ├── log.txt
│   ├── loss.png
│   └── weight.pth
├── results
├── dataset.py
├── helper.py
├── loss.py
├── model.py
├── predict.py
├── train.py
└── README.md

```
### Training
```
python train.py
```
Training and validation loss:

![Image description](output/loss.png)
### Testing
```
python test.py
```
Some experimental results:

![Image description](results/0001TP_008820.png)
![Image description](results/0006R0_f01020.png)
![Image description](results/0016E5_04440.png)
![Image description](results/0016E5_08109.png)
![Image description](results/Seq05VD_f03270.png)
![Image description](results/Seq05VD_f04140.png)
