import os
import stat
import subprocess
import yaml
import operator
import argparse
from functools import reduce
from train.core.config import get_grid_search_configs

condor_template = """
executable = <<SCRIPTNAME>>
arguments = mask_generation
error = <<PATH>>/<<JOBNAME>>.$(Process).err
output = <<PATH>>/<<JOBNAME>>.$(Process).out
log = <<PATH>>/<<JOBNAME>>.$(Process).log
request_memory = <<MEMORYMBS>>
request_cpus = <<CPU_COUNT>>
request_gpus = <<GPU_COUNT>>
#requirements = (TARGET.CUDACapability<8.0 && TARGET.CUDAGlobalMemoryMb>50000)
# && TARGET.CUDAGlobalMemoryMb>30000)
#&& UtsnameNodename =!= "g033")
#requirements = UtsnameNodename =!= "g087"
#requirements = (CUDADeviceName!=\"NVIDIA A100-SXM-80GB\" && CUDADeviceName!=\"A100-SXM4-40GB\")
requirements = (UtsnameNodename=!="g133" && UtsnameNodename=!="g140" && UtsnameNodename=!="g137" && UtsnameNodename=!="g144" && UtsnameNodename=!="g123" && UtsnameNodename=!="g135" && TARGET.CUDAGlobalMemoryMb>30000)

+MaxRunningPrice = <<MAX_PRICE>>
+RunningPriceExceededAction = "kill"

on_exit_hold = (ExitCode =?= 3)
on_exit_hold_reason = "Checkpointed, will resume"
on_exit_hold_subcode = 1
periodic_release = ( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && (HoldReasonSubCode =?= 1) )
queue <<NJOBS>>
"""
script_template = """
source /etc/profile.d/modules.sh
module load cuda/11.1
source <<VENV>>/bin/activate
cd <<BASE_FOLDER>>
python <<SCRIPT>> --cfg <<PARAMS>> --log_dir <<LOG_DIR>> --resume <<TEST_FLAG>>
"""
#wandb login f1f68133bbd3b59582746eeb674b0335135caa32

######################################################
CPU_COUNT = 16
GPU_COUNT = 1
MAX_MEM_GB = 50
NUM_JOBS = 1
MAX_TIME_H = 36
MAX_PRICE = 2020
BID = 2000
USERNAME = 'ppatel'
######################################################


def get_from_dict(dict, keys):
    return reduce(operator.getitem, keys, dict)


def parse_config(cfg_file):
    cfg = yaml.full_load(open(cfg_file))
    # parse config file to get a list of configs and related hyperparameters
    different_configs, hyperparams = get_grid_search_configs(
        cfg,
        excluded_keys=[],
    )
    return different_configs, hyperparams


def execute_on_cluster(script, params, log_dir_script, base_folder, venv, log_dir, test_flag, job_name):

    st = script_template
    st = st.replace('<<VENV>>', venv)
    st = st.replace('<<SCRIPT>>', script)
    st = st.replace('<<PARAMS>>', params)
    st = st.replace('<<LOG_DIR>>', log_dir_script)
    if test_flag:
        st = st.replace('<<TEST_FLAG>>', '--test')
    else:
        st = st.replace('<<TEST_FLAG>>', '')
    st = st.replace('<<BASE_FOLDER>>', base_folder)
    script_fname = os.path.join(log_dir, job_name+'_run.sh')
    if os.path.exists(script_fname):
        os.chmod(script_fname, stat.S_IRWXU+stat.S_IRWXO)  # make executable
    with open(script_fname, 'w') as fp:
        fp.write(st)
        os.chmod(script_fname, stat.S_IRWXU+stat.S_IRWXO)  # make executable

    cs = condor_template
    cs = cs.replace('<<PATH>>', log_dir)
    cs = cs.replace('<<SCRIPTNAME>>', script_fname)
    cs = cs.replace('<<JOBNAME>>', job_name)
    cs = cs.replace('<<CPU_COUNT>>', str(int(CPU_COUNT)))
    cs = cs.replace('<<GPU_COUNT>>', str(int(GPU_COUNT)))
    cs = cs.replace('<<MEMORYMBS>>', str(int(MAX_MEM_GB * 1024)))
    cs = cs.replace('<<MAX_TIME>>', str(int(MAX_TIME_H * 3600)))
    cs = cs.replace('<<MAX_PRICE>>', str(int(MAX_PRICE)))
    cs = cs.replace('<<NJOBS>>', str(NUM_JOBS))

    condor_fname = os.path.join(log_dir, job_name+'_run.sub')
    if os.path.exists(condor_fname):
        os.chmod(condor_fname, stat.S_IRWXU+stat.S_IRWXO)  # make executable
    with open(condor_fname, 'w') as fp:
        fp.write(cs)
        os.chmod(condor_fname, stat.S_IRWXU+stat.S_IRWXO)  # make executable

    cmd = 'condor_submit_bid %d %s' % (BID, condor_fname,)
    subprocess.call(["ssh", "%s@login.cluster.is.localnet" % (USERNAME,)] + [cmd])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=str, default='/is/ps2/ppatel/HMR_train_smplx_release/train_smpl.py')
    parser.add_argument('--base_folder', type=str, default='/is/ps2/ppatel/HMR_train_smplx_release/')
    parser.add_argument('--venv', type=str, default='/fast/ppatel/venv/bedlam_release')
    parser.add_argument('--logdir', type=str, default='')
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--logdir_script', type=str)
    parser.add_argument('--cluster', type=bool, default=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cont', action='store_true')

    args = parser.parse_args()
    all_cfg, hyperparams = parse_config(args.cfg)
    all_log_path = []
    all_cfg_path = []
    os.makedirs(args.logdir, exist_ok=True)

    for cfg_id, cfg in enumerate(all_cfg):
        output_logfilename = args.logdir_script.split('/')[-1]
        for hp in hyperparams:
            v = get_from_dict(all_cfg[cfg_id], hp.split('/'))
            hp = hp.split('/')[-1]
            output_logfilename += f'_{hp.replace("/", ".").replace("_", "").lower()}-{v}'
        output_logfilepath = os.path.join(args.logdir_script, output_logfilename)
        os.makedirs(output_logfilepath, exist_ok=True)
        all_log_path.append(output_logfilepath)

        cfg_filepath = os.path.join(output_logfilepath, output_logfilename+'.yaml')
        all_cfg_path.append(cfg_filepath)
        f = open(cfg_filepath, 'w')
        yaml.dump(cfg, f, default_flow_style=False)

    for i, log_path in enumerate(all_log_path):
        if args.test:
            jobname = log_path.split('/')[-1]+'_err_h36mp1'
        elif args.cont:
            jobname = log_path.split('/')[-1]+'_cont'
        else:
            jobname = log_path.split('/')[-1]
        execute_on_cluster(args.script, all_cfg_path[i], log_path, args.base_folder, args.venv, args.logdir, args.test, jobname)
