import torch
import multiprocessing as mp
import os
import time
import psutil
from multiprocessing import Lock
from CANDI import *
import traceback


# Function to check available GPUs
def get_available_resources():
    available_cpus = list(range(psutil.cpu_count(logical=False)))
    available_gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_free = torch.cuda.memory_allocated(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory
            mem_used_ratio = mem_free / total_mem
            # Assume GPU is free if less than 5% of memory is being used
            if mem_used_ratio < 0.05:
                available_gpus.append(i)
    return available_cpus, available_gpus

# Worker function to train the model on a specific GPU
def train_model_on_resources(hyper_parameters, cpu_id, gpu_id, result_queue):
    try:
        # Attempt to set CPU affinity, but only use CPUs 0 and 1
        p = psutil.Process()
        try:
            p.cpu_affinity([cpu_id % 2])  # This will use either 0 or 1
            print(f"Set CPU affinity to {cpu_id % 2}")
        except Exception as e:
            print(f"Warning: Failed to set CPU affinity: {str(e)}. Proceeding without setting CPU affinity.")

        torch.set_num_threads(1)

        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"Training on CPU {cpu_id % 2}, GPU {gpu_id}")
        print(hyper_parameters)

        hyper_parameters['num_workers'] = 0

        model, metrics = Train_CANDI(
            hyper_parameters,
            eic=hyper_parameters["eic"],
            DNA=hyper_parameters["dna"],
            device=device,
            HPO=True,
            suffix=f"CL{hyper_parameters['context_length']}_nC{hyper_parameters['n_cnn_layers']}_nSAB{hyper_parameters['n_sab_layers']}"
        )

        result = {
            "cpu_id": cpu_id % 2,
            "gpu_id": gpu_id,
            "hyper_parameters": hyper_parameters,
            "metrics": metrics
        }

        result_queue.put(result)
    except Exception as e:
        error_msg = f"Error on CPU {cpu_id % 2}, GPU {gpu_id}: {str(e)}\n{traceback.format_exc()}"
        result_queue.put({"error": error_msg})

# Function to manage GPU assignment and training
def distribute_models_across_resources(hyperparameters_list):
    available_cpus = [0, 1]  # We now know only 0 and 1 are eligible
    available_gpus = get_available_resources()[1]
    print(f"Available CPUs: {available_cpus}")
    print(f"Available GPUs: {available_gpus}")

    result_queue = mp.Queue()
    processes = []
    completed_models = []
    errors = []

    # Function to start a new process
    def start_new_process(hyper_parameters, cpu_id, gpu_id):
        p = mp.Process(target=train_model_on_resources, args=(hyper_parameters, cpu_id, gpu_id, result_queue))
        processes.append(p)
        p.start()

    # Start initial batch of processes
    for i, hyper_parameters in enumerate(hyperparameters_list):
        if i < len(available_gpus):
            start_new_process(hyper_parameters, i % 2, available_gpus[i])
        else:
            break

    # Process results and start new processes as resources become available
    remaining_configs = hyperparameters_list[len(processes):]
    while processes or remaining_configs:
        if not result_queue.empty():
            result = result_queue.get()
            if "error" in result:
                errors.append(result["error"])
                print(result["error"])
            else:
                completed_models.append(result)
                print(f"Model training completed on CPU {result['cpu_id']}, GPU {result['gpu_id']}")

            # Start a new process if there are remaining configurations
            if remaining_configs:
                freed_gpu = result['gpu_id']
                start_new_process(remaining_configs.pop(0), freed_gpu % 2, freed_gpu)

        # Clean up finished processes
        processes = [p for p in processes if p.is_alive()]

        time.sleep(1)  # Avoid busy waiting

    if errors:
        raise Exception("Some processes encountered errors. Check the error messages above.")

    return completed_models


if __name__ == "__main__":
    # Example list of hyperparameter dictionaries
    base_hyperparameters = {
        "data_path": "/project/compbio-lab/encode_data/",
        "dropout": 0.1,
        "pool_size": 2,
        "epochs": 20,
        "inner_epochs": 1,
        "mask_percentage": 0.1,
        "num_loci": 10,
        "lr_halflife": 1,
        "min_avail": 1,
        "eic":True,
    }
    hyperparameter_space = [
        {"n_cnn_layers": 3, "conv_kernel_size": 9, "expansion_factor": 2, 
            "nhead": 4, "n_sab_layers": 8, "context_length": 1600, "pos_enc": "relative", 
            "batch_size": 50, "learning_rate": 1e-3, "dna":True},

        {"n_cnn_layers": 4, "conv_kernel_size": 5, "expansion_factor": 2, 
            "nhead": 8, "n_sab_layers": 4, "context_length": 1200, "pos_enc": "relative", 
            "batch_size": 50, "learning_rate": 1e-3, "dna":True},

        {"n_cnn_layers": 5, "conv_kernel_size": 3, "expansion_factor": 2, 
            "nhead": 8, "n_sab_layers": 4, "context_length": 1600, "pos_enc": "relative", 
            "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #large
        
        {"n_cnn_layers": 3, "conv_kernel_size": 3, "expansion_factor": 2, 
            "nhead": 4, "n_sab_layers": 1, "context_length": 400, "pos_enc": "relative", 
            "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #small

        ################################################################################

        {"n_cnn_layers": 3, "conv_kernel_size": 9, "expansion_factor": 2, 
            "nhead": 4, "n_sab_layers": 8, "context_length": 1600, "pos_enc": "relative", 
            "batch_size": 50, "learning_rate": 1e-3, "dna":False},

        {"n_cnn_layers": 4, "conv_kernel_size": 5, "expansion_factor": 2, 
            "nhead": 8, "n_sab_layers": 4, "context_length": 1200, "pos_enc": "relative", 
            "batch_size": 50, "learning_rate": 1e-3, "dna":False},

        {"n_cnn_layers": 5, "conv_kernel_size": 3, "expansion_factor": 2, 
            "nhead": 8, "n_sab_layers": 4, "context_length": 1600, "pos_enc": "relative", 
            "batch_size": 50, "learning_rate": 1e-3, "dna":False},

        {"n_cnn_layers": 3, "conv_kernel_size": 3, "expansion_factor": 2, 
            "nhead": 4, "n_sab_layers": 1, "context_length": 400, "pos_enc": "relative", 
            "batch_size": 50, "learning_rate": 1e-3, "dna":False}
            ]
    random.shuffle(hyperparameter_space)

    hyperparameters_list = []
    for s in hyperparameter_space:
        merged_dict = base_hyperparameters | s
        hyperparameters_list.append(merged_dict)
    print(hyperparameters_list)
    # Make sure to set the correct start method for multiprocessing
    mp.set_start_method('spawn', force=True)

    try:
        results = distribute_models_across_resources(hyperparameters_list)

        with open("hpo_results.txt", "w") as file:
            # Print results
            for result in results:
                print(f"Results for hyperparameters: {result['hyper_parameters']}")
                print(f"Metrics: {result['metrics']}")
                file.write(str(result['hyper_parameters']))
                file.write(str(result['metrics']))
                file.write("\n\n\n")

        print("All configurations tested successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
