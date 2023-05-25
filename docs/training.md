
## Training BEDLAM-CLIFF

### Prepare data
To download BEDLAM and AGORA training images and ground truth labels, run the following script.  Since, the data size is huge it will take a while to finish. Please first register on [BEDLAM](https://bedlam.is.tue.mpg.de/) website.
```
bash fetch_training_data.sh
```
Once downloaded, you can uncompress the data in the same directory.

If you have not yet prepare the data for demo run, then you need to follow data preparation instructions in [Demo](#Demo). This will download all the utility files need to run the training.

Finally download the HRNet checkpoint pretrained on COCO images from [here](https://drive.google.com/file/d/15T2XqPjW7Ex0uyC1miGVYUv7ULOxIyJI/view?usp=share_link) and put it in data/ckpt/pretrained.

To use 3DPW as validation dataset during training, you need to download 3DPW images and save them in data/test_images/3DPW/imageFiles

### Train
BEDLAM-CLIFF MODEL
```
python train.py --cfg configs/bedlam_cliff.yaml
```

BEDLAM-HMR MODEL
```
python train.py --cfg configs/bedlam_hmr.yaml
```

### Finetune with 3DPW
If you want to finetune the model with 3DPW training data, you can run the following script.
```
python train.py --cfg configs/bedlam_cliff_3dpw_ft.yaml --resume --ckpt data/ckpt/bedlam_cliff.ckpt 

```
Note that you need to download 3DPW training images from [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) and place them in `data/real_training_images/3DPW/imageFiles`. You can also provide your own image path in train/core/config.py `'3dpw-train-smplx': 'data/real_training_images/3DPW'`.


## Training Hand Model

### Prepare data
Register on [BEDLAM](https://bedlam.is.tue.mpg.de/) website and then run the following script:
```
bash fetch_hand_training_data.sh
```

Download the HRNet pretrained Imagenet checkpoint from [here](https://drive.google.com/drive/folders/1E6j6W7RqGhW1o7UHgiQ9X4g8fVJRU9TX) and place it in data/ckpt/pretrained

Once the data is downloaded, you can train hand model with following command.
```
python train_hands.py --cfg configs/hands.yaml
```

## Training BEDLAM-CLIFF-X

### Prepare data 
Register on [BEDLAM](https://bedlam.is.tue.mpg.de/) website and then run the following script:
```
bash fetch_hand_training_data.sh
```

### BEDLAM-CLIFF-X

For training BEDLAM-CLIFF-X, you can either use the pretrained checkpoint for body and hand model stored in data/ckpt or train them from scratch as described above. Once you have the checkpoints for BEDLAM-CLIFF and hand model, you can train BEDLAM-CLIFF-X

```
python trainx.py --cfg configs/bedlam_cliff_x.yaml --hand_ckpt data/ckpt/hands_with_agora.ckpt --body_ckpt data/ckpt/bedlam_cliff.ckpt
```

## Training CLIFF model with real images

### Prepare data
Register on [BEDLAM](https://bedlam.is.tue.mpg.de/) website and then run the following script:
```
bash fetch_real_training_data.sh
```

Also download the training images from respective dataset websites in data/real_training_images [COCO](https://cocodataset.org/#home) (2014 Train Images), [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/), [MPII](http://human-pose.mpi-inf.mpg.de/),  [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/), [H3.6M](http://vision.imar.ro/human3.6m/description.php). 
After the download, you should have the following structure in your data folder.
```  
${data}  
|-- real_training_images
|   |-- 3DPW
|   |   |-- imageFiles
|   |-- coco
|   |   |-- train2014
|   |-- mpii
|   |   |-- images
|   |-- h36m
|   |   |-- Images
|   |   |   |-- S1 
|   |   |   |-- S2 ..
|   |-- mpi-inf-3dhp
|   |   |-- S1
|   |   |-- S2 ..
```  
Please note that if you don't want to follow the directory structure, you can also modify the path to the image folder in train/core/config.py

### CLIFF
With the following command you could train the CLIFF model with real images.

```
python train_smpl.py --cfg configs/orig_cliff.yaml

```