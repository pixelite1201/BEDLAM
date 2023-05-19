#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }


# # SMPL body model for evaluation of 3DPW/RICH/H3.6M
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/body_models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' -O './data/body_models/smpl.zip' --no-check-certificate --continue
unzip data/body_models/smpl.zip -d data/body_models/
mv data/body_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_FEMALE.pkl
mv data/body_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_MALE.pkl
mv data/body_models/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl data/body_models/SMPL_python_v.1.1.0/smpl/models/SMPL_NEUTRAL.pkl


# # # Parsed ground truth labels for 3DPW/RICH/H3.6M evaluation
echo -e "\nYou need to register at https://bedlam.is.tue.mpg.de/"
read -p "Username (BEDLAM):" username
read -p "Password (BEDLAM):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/ckpt
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=checkpoints/bedlam_cliff_3dpw_finetuned.ckpt' -O './data/ckpt/bedlam_cliff_3dpw_ft.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=checkpoints/bedlam_hmr.ckpt' -O './data/ckpt/bedlam_hmr.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=checkpoints/bedlam_hmr_3dpw_finetuned.ckpt' -O './data/ckpt/bedlam_hmr_3dpw_ft.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=checkpoints/cliff.ckpt' -O './data/ckpt/cliff.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=checkpoints/cliff_3dpw_finetuned.ckpt' -O './data/ckpt/cliff_3dpw_ft.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=checkpoints/cliff_noh36m.ckpt' -O './data/ckpt/cliff_no_h36m.ckpt' --no-check-certificate --continue

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=eval_data_parsed.zip' -O './data/eval_data_parsed.zip' --no-check-certificate --continue
unzip data/eval_data_parsed.zip -d data
# HBW bboxes
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=hbw_test_images_bbox.zip' -O './data/hbw_test_images_bbox.zip' --no-check-certificate --continue
unzip data/hbw_test_images_bbox.zip -d data/test_images/
# SSP BEDLAM-CLIFF results in SMPL format
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=ssp_bedlam_cliff_smpl.zip' -O './data/ssp_bedlam_cliff_smpl.zip' --no-check-certificate --continue
unzip data/ssp_bedlam_cliff_smpl.zip -d data
# SSP-3D parsed data
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=ssp_3d_test.npz' -O './data/ssp_3d_test.npz' --no-check-certificate --continue


