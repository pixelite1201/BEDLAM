# Submission format

For each predicted person in an image, a dictionary should be generated and stored as pickle file in predictions folder with following filename format.

If the image name is Image.png and there are 3 prediction for the corresponding image then the output prediction file name will be Image_personId_0.pkl, Image_personId_1.pkl and Image_personId_2.pkl

## SMPL-X dictionary (uploading joints and vertices):
For each predicted person in the image, a dictionary with following keys needs to be generated. Note that the data type of all the parameters is np.ndarray.

joints : (shape : (24,2), units : pixel coordinate). 2d projected joints location in the image. This is used to match the predition with the ground truth.

verts : (shape : (10475,3), units : meters). 3d vertices in camera coordinates. This is used to calculate the MVE/NMVE error for body, face and hands after aligning the root joint of prediction and ground truth.

allSmplJoints3d : (shape : (127, 3), units : meters). 3d joints in camera coordinates. This is used to calculate the MPJPE/NMJE error for body, face and hands after aligning the root joint of prediction and ground truth.

## Check format
Once you have generated all the prediction (.pkl) files as explained above, create a zip of the folder name predictions containing the files e.g. predictions.zip. The following command will extract the predictions.zip in extract_zip folder and will verify if the shape and type for all the parameters in the individual pickle file is correct.
```
python check_pred_format.py --predZip predictions.zip --extractZipFolder extract_zip

```
You can now upload the predicions.zip file to the leaderboard.