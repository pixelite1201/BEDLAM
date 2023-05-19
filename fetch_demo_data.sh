#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL-X model
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/body_models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O './data/body_models/smplx.zip' --no-check-certificate --continue
unzip data/body_models/smplx.zip -d data/body_models/smplx

# # MANO model
echo -e "\nYou need to register at https://mano.is.tue.mpg.de"
read -p "Username (MANO):" username
read -p "Password (MANO):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/body_models
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=mano_v1_2.zip' -O './data/body_models/mano.zip' --no-check-certificate --continue
unzip data/body_models/mano.zip -d data/body_models/mano


# BEDLAM checkpoints
echo -e "\nYou need to register at https://bedlam.is.tue.mpg.de/"
read -p "Username (BEDLAM):" username
read -p "Password (BEDLAM):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data/ckpt
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=checkpoints/bedlam_cliff.ckpt' -O './data/ckpt/bedlam_cliff.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=checkpoints/bedlam_cliff_x.ckpt' -O './data/ckpt/bedlam_cliff_x.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=checkpoints/hands_with_agora.ckpt' -O './data/ckpt/hands_with_agora.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=checkpoints/hands_without_agora.ckpt' -O './data/ckpt/hands_without_agora.ckpt' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=utils.zip' -O './data/utils.zip' --no-check-certificate --continue
unzip data/utils.zip -d data
