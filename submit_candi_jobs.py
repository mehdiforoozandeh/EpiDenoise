import subprocess
import time
import copy

# base_hyperparameters = {
#     "data_path": "/project/compbio-lab/encode_data/",
#     "dropout": 0.1,
#     "pool_size": 2,
#     "epochs": 10,
#     "inner_epochs": 1,
#     "mask_percentage": 0.1,
#     "num_loci": 1000,
#     "lr_halflife": 1,
#     "min_avail": 1,
#     "eic":True,
# }

# hyperparameter_space = [
#     {"n_cnn_layers": 3, "conv_kernel_size": 9, "expansion_factor": 2, 
#         "nhead": 4, "n_sab_layers": 8, "context_length": 1600, "pos_enc": "relative", 
#         "batch_size": 50, "learning_rate": 1e-3, "dna":True},

#     {"n_cnn_layers": 4, "conv_kernel_size": 5, "expansion_factor": 2, 
#         "nhead": 8, "n_sab_layers": 4, "context_length": 1200, "pos_enc": "relative", 
#         "batch_size": 50, "learning_rate": 1e-3, "dna":True},

#     {"n_cnn_layers": 5, "conv_kernel_size": 3, "expansion_factor": 2, 
#         "nhead": 8, "n_sab_layers": 4, "context_length": 1600, "pos_enc": "relative", 
#         "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #large
    
#     {"n_cnn_layers": 3, "conv_kernel_size": 3, "expansion_factor": 2, 
#         "nhead": 4, "n_sab_layers": 1, "context_length": 400, "pos_enc": "relative", 
#         "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #small

#     ################################################################################

#     {"n_cnn_layers": 3, "conv_kernel_size": 9, "expansion_factor": 2, 
#         "nhead": 4, "n_sab_layers": 8, "context_length": 1600, "pos_enc": "relative", 
#         "batch_size": 50, "learning_rate": 1e-3, "dna":False},

#     {"n_cnn_layers": 4, "conv_kernel_size": 5, "expansion_factor": 2, 
#         "nhead": 8, "n_sab_layers": 4, "context_length": 1200, "pos_enc": "relative", 
#         "batch_size": 50, "learning_rate": 1e-3, "dna":False},

#     {"n_cnn_layers": 5, "conv_kernel_size": 3, "expansion_factor": 2, 
#         "nhead": 8, "n_sab_layers": 4, "context_length": 1600, "pos_enc": "relative", 
#         "batch_size": 50, "learning_rate": 1e-3, "dna":False},

#     {"n_cnn_layers": 3, "conv_kernel_size": 3, "expansion_factor": 2, 
#         "nhead": 4, "n_sab_layers": 1, "context_length": 400, "pos_enc": "relative", 
#         "batch_size": 50, "learning_rate": 1e-3, "dna":False}
#         ]


hyperparameters_list = [
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 4 --conv_kernel_size 3 --expansion_factor 2 --nhead 8 --n_sab_layers 4 --context_length 1200 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --dna --suffix CL1200_nC4_k3 --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 4 --conv_kernel_size 5 --expansion_factor 2 --nhead 8 --n_sab_layers 4 --context_length 1200 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --dna --suffix CL1200_nC4_k5 --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 7 --expansion_factor 2 --nhead 4 --n_sab_layers 4 --context_length 1600 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --dna --suffix CL1600_nC3_k7 --separate_decoders",

    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1000 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 3 --expansion_factor 3 --nhead 9 --n_sab_layers 2 --context_length 800 --pos_enc relative --batch_size 50 --learning_rate 1e-1 --dna --suffix CL800_nC3_k3_ef3_sepDec_lre-1 --separate_decoders",
    "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1000 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 3 --expansion_factor 3 --nhead 9 --n_sab_layers 2 --context_length 800 --pos_enc relative --batch_size 50 --learning_rate 1e-2 --dna --suffix CL800_nC3_k3_ef3_sepDec_lre-3_fill_y_prompt --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1000 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 3 --expansion_factor 3 --nhead 9 --n_sab_layers 2 --context_length 800 --pos_enc relative --batch_size 50 --learning_rate 1e-4 --dna --suffix CL800_nC3_k3_ef3_sepDec_lre-4 --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1000 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 3 --expansion_factor 3 --nhead 9 --n_sab_layers 2 --context_length 800 --pos_enc relative --batch_size 50 --learning_rate 1e-5 --dna --suffix CL800_nC3_k3_ef3_sepDec_lre-5 --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 3 --expansion_factor 3 --nhead 9 --n_sab_layers 2 --context_length 800 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --dna --suffix CL800_nC3_k3_ef3_shrDec",

    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 5 --inner_epochs 1 --mask_percentage 0.1 --num_loci 3000 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 3 --expansion_factor 3 --nhead 9 --n_sab_layers 4 --context_length 1600 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --dna --suffix CL1600_nC3_k3_ef3_hiLoci --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 20 --inner_epochs 1 --mask_percentage 0.1 --num_loci 750 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 3 --expansion_factor 3 --nhead 9 --n_sab_layers 4 --context_length 1600 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --dna --suffix CL1600_nC3_k3_ef3_hiEpc --separate_decoders",
    
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 5 --expansion_factor 3 --nhead 9 --n_sab_layers 2 --context_length 1200 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --dna --suffix CL1200_nC3_k5_ef3 --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 9 --expansion_factor 3 --nhead 9 --n_sab_layers 2 --context_length 1200 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --dna --suffix CL1200_nC3_k9_ef3 --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 15 --expansion_factor 3 --nhead 9 --n_sab_layers 2 --context_length 1200 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --dna --suffix CL1200_nC3_k15_ef3 --separate_decoders",
    
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 9 --expansion_factor 2 --nhead 4 --n_sab_layers 2 --context_length 1200 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --dna --suffix CL1200_nC3_k9_ef2 --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 15 --expansion_factor 2 --nhead 4 --n_sab_layers 2 --context_length 1200 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --dna --suffix CL1200_nC3_k15_ef2 --separate_decoders",

    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 4 --conv_kernel_size 3 --expansion_factor 2 --nhead 8 --n_sab_layers 4 --context_length 1200 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --suffix CL1200_nC4_k3 --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 4 --conv_kernel_size 5 --expansion_factor 2 --nhead 8 --n_sab_layers 4 --context_length 1200 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --suffix CL1200_nC4_k5 --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 7 --expansion_factor 2 --nhead 4 --n_sab_layers 4 --context_length 1600 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --suffix CL1600_nC3_k7 --separate_decoders",
    # "--data_path /project/compbio-lab/encode_data/ --dropout 0.1 --pool_size 2 --epochs 10 --inner_epochs 1 --mask_percentage 0.1 --num_loci 1500 --lr_halflife 1 --min_avail 1 --hpo --eic --n_cnn_layers 3 --conv_kernel_size 9 --expansion_factor 2 --nhead 4 --n_sab_layers 4 --context_length 1600 --pos_enc relative --batch_size 50 --learning_rate 1e-3 --suffix CL1600_nC3_k9 --separate_decoders",
]

# SLURM job template
job_template = """#!/bin/bash
#SBATCH -J candi-HPO-{job_id}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --time={time_limit}-00:00
#SBATCH --mem={memory}
#SBATCH --output=log_HPO_CANDI_{job_id}.out
#SBATCH --partition=compbio-lab-long
#SBATCH --nodelist=cs-venus-03

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sslgpu
srun python CANDI.py {script_args}
"""

# Parameters for the SLURM job script
cpus_per_task = 2
num_gpus = 1
time_limit = 4  # Time limit in days
memory = '50G'
n_simultaneous_jobs = 2  # Number of jobs to run simultaneously

# Keep track of jobs
jobs_to_submit = copy.deepcopy(hyperparameters_list)
running_jobs = {}
completed_jobs = []
max_jobs = n_simultaneous_jobs

def submit_job(job_id, script_args):
    # Fill in the job script template
    job_script_content = job_template.format(
        job_id=job_id,
        cpus_per_task=cpus_per_task,
        num_gpus=num_gpus,
        time_limit=time_limit,
        memory=memory,
        script_args=script_args
    )
    # Write the job script to a file
    job_script_filename = f"candi_job_{job_id}.sh"
    with open(job_script_filename, 'w') as f:
        f.write(job_script_content)
    # Submit the job using sbatch
    submit_command = ['sbatch', job_script_filename]
    result = subprocess.run(submit_command, capture_output=True, text=True)
    if result.returncode == 0:
        # Extract the SLURM job ID
        output = result.stdout.strip()
        slurm_job_id = output.split()[-1]
        print(f"Submitted job {job_id} with SLURM ID {slurm_job_id}")
        return slurm_job_id
    else:
        print(f"Failed to submit job {job_id}")
        print(result.stderr)
        return None

def check_running_jobs(running_jobs):
    # Get list of running job IDs from squeue
    check_command = ['squeue', '--noheader', '--format=%A']
    result = subprocess.run(check_command, capture_output=True, text=True)
    if result.returncode == 0:
        running_slurm_ids = result.stdout.strip().split()
        updated_running_jobs = running_jobs.copy()
        for job_id, slurm_job_id in running_jobs.items():
            if slurm_job_id not in running_slurm_ids:
                print(f"Job {job_id} with SLURM ID {slurm_job_id} has finished.")
                updated_running_jobs.pop(job_id)
                completed_jobs.append(job_id)
        return updated_running_jobs
    else:
        print("Failed to retrieve running jobs.")
        print(result.stderr)
        return running_jobs

# Main loop to submit and monitor jobs
job_counter = 0
while jobs_to_submit or running_jobs:
    # Submit jobs if we have capacity
    while jobs_to_submit and len(running_jobs) < max_jobs:
        script_args = jobs_to_submit.pop(0)
        job_id = job_counter
        slurm_job_id = submit_job(job_id, script_args)
        if slurm_job_id:
            running_jobs[job_id] = slurm_job_id
            job_counter += 1
        else:
            # If submission failed, consider adding the job back to the queue or handle the error
            pass
    # Wait before checking again
    time.sleep(60)  # Wait for 1 minute
    # Check status of running jobs
    running_jobs = check_running_jobs(running_jobs)

print("All jobs have been completed.")