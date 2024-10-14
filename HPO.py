from CANDI import *
import torch
import multiprocessing
import os
import time
from multiprocessing import Lock


# Function to check available GPUs
def get_available_gpus():
    available_gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_free = torch.cuda.memory_allocated(i)
            total_mem = torch.cuda.get_device_properties(i).total_memory
            mem_used_ratio = mem_free / total_mem
            # Assume GPU is free if less than 5% of memory is being used
            if mem_used_ratio < 0.05:
                available_gpus.append(i)
    return available_gpus

# Worker function to train the model on a specific GPU
def train_model_on_gpu(hyper_parameters, gpu_id, result_queue, process_index):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Training on GPU {gpu_id} on CPU {process_index}")
    
    # Set CPU affinity for this process
    os.sched_setaffinity(0, {process_index})
    
    # Call the Train_CANDI function to train the model on this GPU
    model, metrics = Train_CANDI(
        hyper_parameters, eic=hyper_parameters["eic"], DNA=hyper_parameters["dna"], device=device, HPO=True, 
        suffix=f"CL{hyper_parameters['context_length']}_nC{hyper_parameters['n_cnn_layers']}_nSAB{hyper_parameters['n_sab_layers']}")
    
    # After training, save the results in a dictionary
    result = {
        "gpu_id": gpu_id,
        "hyper_parameters": hyper_parameters,
        "metrics": metrics
    }
    
    # Put the result into the queue
    result_queue.put(result)

# Function to manage GPU assignment and training
def distribute_models_across_gpus(hyperparameters_list):
    available_gpus = get_available_gpus()
    print(f"Available GPUs: {available_gpus}")
    
    # Queue to store results
    result_queue = multiprocessing.Queue()

    # List to keep track of active processes
    active_processes = []

    # Lock for safe GPU allocation
    lock = Lock()

    # Set to track used hyperparameters to avoid re-running the same
    used_hyperparameters = set()

    # Function to start training on available GPUs
    def start_training():
        with lock:
            for hyper_parameters in hyperparameters_list:
                # Avoid re-running the same hyperparameters
                if str(hyper_parameters) in used_hyperparameters:
                    continue
                used_hyperparameters.add(str(hyper_parameters))
                
                if available_gpus:
                    # Pop a GPU from the available list
                    gpu_id = available_gpus.pop(0)
                    process_index = len(active_processes)  # Unique index for this process
                    # Create a new process for training on this GPU
                    p = multiprocessing.Process(target=train_model_on_gpu, args=(hyper_parameters, gpu_id, result_queue, process_index))
                    active_processes.append(p)
                    p.start()
                    time.sleep(0.1)  # Small sleep to avoid rapid allocation (optional)
                else:
                    # If no GPU is available, break the loop
                    break

    # Start initial training processes
    start_training()

    # Collect results and assign new tasks when GPUs are free
    completed_models = []
    while len(completed_models) < len(hyperparameters_list):
        for p in active_processes:
            if not p.is_alive():
                # Collect the result
                result = result_queue.get()
                print(f"Model training completed on GPU {result['gpu_id']}")
                completed_models.append(result)

                with lock:
                    # Release the GPU back to the pool
                    available_gpus.append(result['gpu_id'])

                # Remove process from the list safely
                if p in active_processes:
                    active_processes.remove(p)

                # Start new training if GPUs are available
                start_training()

    # Wait for all remaining processes to finish
    for p in active_processes:
        p.join()

    return completed_models

# from CANDI import *
# import torch
# import multiprocessing
# import os
# import time
# from multiprocessing import Lock


# # Function to check available GPUs
# def get_available_gpus():
#     available_gpus = []
#     if torch.cuda.is_available():
#         for i in range(torch.cuda.device_count()):
#             mem_free = torch.cuda.memory_allocated(i)
#             total_mem = torch.cuda.get_device_properties(i).total_memory
#             mem_used_ratio = mem_free / total_mem
#             # Assume GPU is free if less than 5% of memory is being used
#             if mem_used_ratio < 0.05:
#                 available_gpus.append(i)
#     return available_gpus

# # Worker function to train the model on a specific GPU
# def train_model_on_gpu(hyper_parameters, gpu_id, result_queue):
#     device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
#     print(f"Training on GPU {gpu_id}")
#     print(hyper_parameters)
    
#     # Call the Train_CANDI function to train the model on this GPU
#     model, metrics = Train_CANDI(
#         hyper_parameters, eic=hyper_parameters["eic"], DNA=hyper_parameters["dna"], device=device, HPO=True, 
#         suffix=f"CL{hyper_parameters['context_length']}_nC{hyper_parameters['n_cnn_layers']}_nSAB{hyper_parameters['n_sab_layers']}")
    
#     # After training, save the results in a dictionary
#     result = {
#         "gpu_id": gpu_id,
#         "hyper_parameters": hyper_parameters,
#         "metrics": metrics
#     }
    
#     # Put the result into the queue
#     result_queue.put(result)
# # Function to manage GPU assignment and training
# def distribute_models_across_gpus(hyperparameters_list):
#     available_gpus = get_available_gpus()
#     print(f"Available GPUs: {available_gpus}")
    
#     # Queue to store results
#     result_queue = multiprocessing.Queue()

#     # List to keep track of active processes
#     active_processes = []

#     # Lock for safe GPU allocation
#     lock = Lock()

#     # Set to track used hyperparameters to avoid re-running the same
#     used_hyperparameters = set()

#     # Function to start training on available GPUs
#     def start_training():
#         with lock:
#             for hyper_parameters in hyperparameters_list:
#                 # Avoid re-running the same hyperparameters
#                 if str(hyper_parameters) in used_hyperparameters:
#                     continue
#                 used_hyperparameters.add(str(hyper_parameters))
                
#                 if available_gpus:
#                     # Pop a GPU from the available list
#                     gpu_id = available_gpus.pop(0)
#                     # Create a new process for training on this GPU
#                     p = multiprocessing.Process(target=train_model_on_gpu, args=(hyper_parameters, gpu_id, result_queue))
#                     active_processes.append(p)
#                     p.start()
#                     time.sleep(0.1)  # Small sleep to avoid rapid allocation (optional)
#                 else:
#                     # If no GPU is available, break the loop
#                     break

#     # Start initial training processes
#     start_training()

#     # Collect results and assign new tasks when GPUs are free
#     completed_models = []
#     while len(completed_models) < len(hyperparameters_list):
#         for p in active_processes:
#             if not p.is_alive():
#                 # Collect the result
#                 result = result_queue.get()
#                 print(f"Model training completed on GPU {result['gpu_id']}")
#                 completed_models.append(result)

#                 with lock:
#                     # Release the GPU back to the pool
#                     available_gpus.append(result['gpu_id'])

#                 # Remove process from the list safely
#                 if p in active_processes:
#                     active_processes.remove(p)

#                 # Start new training if GPUs are available
#                 start_training()

#     # Wait for all remaining processes to finish
#     for p in active_processes:
#         p.join()

#     return completed_models

if __name__ == "__main__":
    # Example list of hyperparameter dictionaries
    base_hyperparameters = {
        "data_path": "/project/compbio-lab/encode_data/",
        "dropout": 0.1,
        "pool_size": 2,
        "epochs": 20,
        "inner_epochs": 1,
        "mask_percentage": 0.1,
        "num_loci": 1000,
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
            "nhead": 16, "n_sab_layers": 2, "context_length": 800, "pos_enc": "relative", 
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
            "nhead": 16, "n_sab_layers": 2, "context_length": 800, "pos_enc": "relative", 
            "batch_size": 50, "learning_rate": 1e-3, "dna":False},

        {"n_cnn_layers": 5, "conv_kernel_size": 3, "expansion_factor": 2, 
            "nhead": 8, "n_sab_layers": 4, "context_length": 1600, "pos_enc": "relative", 
            "batch_size": 50, "learning_rate": 1e-3, "dna":False},

        {"n_cnn_layers": 3, "conv_kernel_size": 3, "expansion_factor": 2, 
            "nhead": 4, "n_sab_layers": 1, "context_length": 400, "pos_enc": "relative", 
            "batch_size": 50, "learning_rate": 1e-3, "dna":False}
            ]

    hyperparameters_list = []
    for s in hyperparameter_space:
        merged_dict = base_hyperparameters | s
        hyperparameters_list.append(merged_dict)
    print(hyperparameters_list)
    # Make sure to set the correct start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Distribute and train models across GPUs
    results = distribute_models_across_gpus(hyperparameters_list)

    with open("hpo_results.txt", "w") as file:
        # Print results
        for result in results:
            print(f"Results for hyperparameters: {result['hyper_parameters']}")
            print(f"Metrics: {result['metrics']}")
            file.write(str(result['hyper_parameters']))
            file.write(str(result['metrics']))
            file.write("\n\n\n")

    
