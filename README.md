<div align="center">

# BEDLAM: Bodies Exhibiting Detailed Lifelike Animated Motion 
## CVPR 2023

## [Project Page](https://bedlam.is.tue.mpg.de) | [Paper](https://bedlam.is.tuebingen.mpg.de/media/upload/BEDLAM_CVPR2023.pdf) | Video (Coming soon)

</div>
</div>

<p align="center">
    <img src="docs/data/myimage.gif">
    <br>
    <sup>Recounstruction results on images from different benchmarks: HBW, SSP-3D, RICH.</sup>
    <br>
</p>


This repository contains the code to train and evaluate BEDLAM-CLIFF, BEDLAM-HMR, BEDLAM-CLIFF-X model from the paper. 

## Install
Create a virtual environment and install all the requirements
```
python3.8 -m venv bedlam_venv
source bedlam_venv/bin/activate
pip install -r requirements.txt
```

## Quick Demo



### Prepare data
If you need to run just the demo, please follow the following steps:

Step 1. Register on [SMPL-X](https://smpl-x.is.tue.mpg.de/) website.

Step 2. Register on [MANO](https://mano.is.tue.mpg.de/) website.

Step 3. Register on [BEDLAM](https://bedlam.is.tue.mpg.de/) website.

Step 4. Run the following script to fetch demo data. The script will need the username and password created in above steps.
```
bash fetch_demo_data.sh
```

### BEDLAM-CLIFF demo

```
 python demo.py --cfg configs/demo_bedlam_cliff.yaml

```
### BEDLAM-CLIFF-X demo
```
python demox.py --cfg configs/demo_bedlam_cliff_x.yaml --display
```


## Evaluation
For instructions on how to run evaluation on different benchmarks please refer to [evaluation.md](docs/evaluation.md)


## Training
For instructions on how to run training please refer to [training.md](docs/training.md)


<!-- ### CLIFF training -->
# Citation
```
@inproceedings{Black:CVPR:2023,
  title = {{BEDLAM}: A Synthetic Dataset of Bodies Exhibiting Detailed Lifelike Animated Motion},
  author = {Black, Michael J. and Patel, Priyanka and Tesch, Joachim and Yang, Jinlong}, 
  booktitle = {Proceedings IEEE/CVF Conf. on Computer Vision and Pattern Recognition ({CVPR})}, 
  month = jun,
  year = {2023},
  month_numeric = {6}
}
```

# References
We benefit from many great resources including but not limited to [SMPL-X](https://smpl-x.is.tue.mpg.de/), [SMPL](https://smpl.is.tue.mpg.de), [PARE](https://gitlab.tuebingen.mpg.de/mkocabas/projects/-/tree/master/pare), [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF), [AGORA](https://agora.is.tue.mpg.de), [PIXIE](https://pixie.is.tue.mpg.de), [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).


