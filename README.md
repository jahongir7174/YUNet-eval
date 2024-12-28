[YUNet](https://link.springer.com/article/10.1007/s11633-023-1423-y) evaluation on WIDERFace dataset

### Installation

```
conda create -n YUNet python=3.11.11
conda activate YUNet
conda install python=3.11.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python
pip install PyYAML
pip install scipy
pip install tqdm
```

### Evaluate

* Configure your dataset path in `main.py` for evaluation
* Run `python main.py`

### Results

| Model                   | AP_easy | AP_medium | AP_hard | #Params | Params Ratio | MFlops (320x320) | FPS(320x320) |
|-------------------------|---------|-----------|---------|---------|--------------|------------------|--------------|
| SCRFD0.5(ICLR2022)      | 0.892   | 0.885     | 0.819   | 631,410 | 8.32x        | 184              | 284          |
| Retinaface0.5(CVPR2020) | 0.907   | 0.883     | 0.742   | 426,608 | 5.62X        | 245              | 235          |
| YuNet_n(Original)       | 0.892   | 0.883     | 0.811   | 75,856  | 1.00x        | 149              | 456          |
| YuNet_n(Ours)           | 0.896   | 0.887     | 0.818   | 72,928  | 1.00x        | 133              | 456          |

### Dataset structure

    ├── WIDERFace 
        ├── WIDER_val
            ├── images
                ├── 0--Parade
                ├── 1--Handshaking

#### Reference

* https://github.com/ShiqiYu/libfacedetection.train
