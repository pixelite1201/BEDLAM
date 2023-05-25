

## Evaluation
If you need to run evaluation to reproduce the numbers in the paper, please follow the following steps:

### Prepare data
Step 1. Register on [BEDLAM](https://bedlam.is.tue.mpg.de/) website.

Step 2. Register on [SMPL](https://smpl.is.tue.mpg.de/) website.

Step 3. Run the following script to fetch the data. The script will need the username and password created in above steps.

```
bash fetch_eval_data.sh
```

Step 4. If you have not yet prepare the data for demo run, then you need to follow data preparation instructions in [Demo](#Demo).

Step 5. Download the test images from respective dataset website in data/test_images
[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/), [RICH](https://rich.is.tue.mpg.de), [H3.6M](http://vision.imar.ro/human3.6m/description.php), [SSP](https://github.com/akashsengupta1997/SSP-3D), [HBW](https://shapy.is.tue.mpg.de/), [AGORA](https://agora.is.tue.mpg.de/). 

After the download, you should have the following structure in your data folder.
```  
${data}  
|-- test_images
|   |-- 3DPW
|   |   |-- imageFiles
|   |-- RICH
|   |   |-- test
|   |-- h36m
|   |   |-- Images
|   |-- ssp_3d
|   |   |-- images
|   |   |-- labels.npz
|   |-- HBW_low_resolution
|   |   |-- images
|   |   |   |-- test_small_resolution
|   |--AGORA
|   |   |-- test
```  
Please note that if you don't want to follow the directory structure, you can also modify the path to the image folder in train/core/config.py


### 3DPW/RICH/H3.6M evaluation
The following command will reproduce results of BEDLAM-CLIFF model. You can provide path of different checkpoints with --ckpt flag to reproduce different results e.g. bedlam_cliff_3dpw_ft.ckpt, bedlam_hmr.ckpt, bedlam_hmr_3dpw_ft.ckpt.
```
python train.py --cfg configs/demo_bedlam_cliff.yaml --ckpt data/ckpt/bedlam_cliff.ckpt --test

```

Note if you want to reproduce results for CLIFF trained with real images, you need to run the following script:
```
python train_smpl.py --cfg configs/demo_orig_cliff.yaml --ckpt data/ckpt/cliff.ckpt --test

```
You can provide path of different checkpoints to reproduce different results e.g. cliff_3dpw_ft.ckpt, cliff_no_h36m.ckpt

### HBW evaluation
If you have already download HBW test images in data/test_images/HBW/test_small_resolution then you can run the following script to generate the BEDLAM-CLIFF predictions
```
python demo.py --ckpt data/ckpt/bedlam_cliff.ckpt --eval_dataset hbw --image_folder data/test_images/HBW_low_resolution --output_folder data/test_images/HBW_low_resolution/results
```
This will generate test_hbw_prediction.npz in the output_folder which you can then directly submit to [HBW  evaluation server](https://shapy.is.tue.mpg.de/hbwleaderboard.html) for the results.

### SSP evaluation
Since BEDLAM-CLIFF outputs meshes in SMPL-X format and SSP ground truth are in SMPL format, to evaluate, one needs to find the SMPL shape parameters by fitting SMPL to SMPL-X output of BEDLAM-CLIFF using [SMPL-X to SMPL converter](https://github.com/vchoutas/smplx/tree/main/transfer_model#smpl-x-to-smpl). We provide the BEDLAM-CLIFF results in SMPL format for evaluation. You can run the evaluation on the converted files using following script.

```
python eval_ssp.py ssp_bedlam_cliff_smpl
```

If you want to run the evaluation on SSP-3D test images from scratch, please follow the steps below:
Step 1. After downloading the SSP test images in data/test_images/ssp_3d/images run the following script. This will generate the SMPL-X meshes and parameters in the output_folder
```
python demo.py --eval_dataset ssp --image_folder data/test_images/ssp_3d/images --output_folder data/test_images/ssp_3d/results
```
Step 2. Fit SMPL to SMPL-X meshes from Step 1 and generate SMPL parameter (.pkl) files using [SMPL-X to SMPL converter](https://github.com/vchoutas/smplx/tree/main/transfer_model#smpl-x-to-smpl). Copy all generated pkl files in data/ssp_bedlam_cliff_smpl

Step 3. Run eval script
```
python eval_ssp.py path_to_smplx_pkl_files
```

### AGORA evaluation
If you have already downloaded AGORA test images in data/test_images/AGORA/test then you can run the following script to generate the BEDLAM-CLIFF-X predictions
```
python demox.py --eval_dataset agora --save_result --image_folder data/test_images/AGORA/test --output_folder predictions
```
This will save the results in output folder. You can then zip and submit the result on AGORA evaluation server to get the results. 

### BEDLAM-test evaluation
If you have already downloaded BEDLAM test images in data/test_images/BEDLAM then you can run the following script to generate the BEDLAM-CLIFF-X predictions
```
python demox.py --eval_dataset bedlam --save_result --image_folder demo/test_images/BEDLAM --output_folder predictions
```
This will save the results in output_directory. You can then zip and submit the result on BEDLAM evaluation server to get the results.
