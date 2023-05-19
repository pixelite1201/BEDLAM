import os
import sys
import stat
import subprocess

from loguru import logger

GPUS = {
    'v100-p16': ('\"Tesla V100-PCIE-16GB\"', 'tesla', 16000),
    'v100-p32': ('\"Tesla V100-PCIE-32GB\"', 'tesla', 32000),
    'v100-s32': ('\"Tesla V100-SXM2-32GB\"', 'tesla', 32000),
    'quadro6000': ('\"Quadro RTX 6000\"', 'quadro', 24000),
    'rtx2080ti': ('\"NVIDIA GeForce RTX 2080 Ti\"', 'rtx', 11000),
}

def get_gpus(min_mem=10000, arch=('tesla', 'quadro', 'rtx')):
    gpu_names = []
    for k, (gpu_name, gpu_arch, gpu_mem) in GPUS.items():
        if gpu_mem >= min_mem and gpu_arch in arch:
            gpu_names.append(gpu_name)

    assert len(gpu_names) > 0, 'Suitable GPU model could not be found'

    return gpu_names


def execute_task_on_cluster(
        script,
        exp_name,
        cfg_file,
        num_exp=1,
        exp_opts=None,
        bid_amount=50,
        num_workers=8,
        memory=20000,
        gpu_min_mem=10000,
        gpu_arch=('tesla', 'quadro', 'rtx'),
        num_gpus=1,
):
    gpus = get_gpus(min_mem=gpu_min_mem, arch=gpu_arch)

    gpus = ' || '.join([f'CUDADeviceName=={x}' for x in gpus])

    os.makedirs('run_scripts', exist_ok=True)
    run_script = os.path.join('run_scripts', f'run_{exp_name}.sh')

    os.makedirs(os.path.join('condor_logs', exp_name), exist_ok=True)
    submission = f'executable = {run_script}\n' \
                 'arguments = $(Process) $(Cluster)\n' \
                 f'error = condor_logs/{exp_name}/$(Cluster).$(Process).err\n' \
                 f'output = condor_logs/{exp_name}/$(Cluster).$(Process).out\n' \
                 f'log = condor_logs/{exp_name}/$(Cluster).$(Process).log\n' \
                 f'request_memory = {memory}\n' \
                 f'request_cpus={int(num_workers/2)}\n' \
                 f'request_gpus={num_gpus}\n' \
                 f'requirements={gpus}\n' \
                 f'queue {num_exp}'
                 # f'next_job_start_delay=10\n' \


    with open('submit.sub', 'w') as f:
        f.write(submission)

    logger.info(f'The logs for this experiments can be found under: condor_logs/{exp_name}')
    bash = 'export PYTHONBUFFERED=1\n export PATH=$PATH\n' \
           f'{sys.executable} {script} --cfg {cfg_file} --cfg_id $1'

    if exp_opts is not None:
        bash += ' --opts '
        for opt in exp_opts:
            bash += f'{opt} '
        bash += 'SYSTEM.CLUSTER_NODE $2.$1'
    else:
        bash += ' --opts SYSTEM.CLUSTER_NODE $2.$1'


    with open(run_script, 'w') as f:
        f.write(bash)

    os.chmod(run_script, stat.S_IRWXU)

    cmd = ['condor_submit_bid', f'{bid_amount}', 'submit.sub']
    logger.info('Executing ' + ' '.join(cmd))
    subprocess.call(cmd)
