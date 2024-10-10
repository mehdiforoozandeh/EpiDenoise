from CANDI import *
import torch
import multiprocessing
import os
import time

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
def train_model_on_gpu(hyper_parameters, gpu_id, result_queue):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Training on GPU {gpu_id}")
    
    # Call the Train_CANDI function to train the model on this GPU
    model, metrics = Train_CANDI(hyper_parameters, eic=hyper_parameters["eic"], DNA=hyper_parameters["dna"], device=device)
    
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
    
    def start_training():
        for hyper_parameters in hyperparameters_list:
            if available_gpus:
                # Pop a GPU from the available list
                gpu_id = available_gpus.pop(0)
                # Create a new process for training on this GPU
                p = multiprocessing.Process(target=train_model_on_gpu, args=(hyper_parameters, gpu_id, result_queue))
                active_processes.append(p)
                p.start()
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

if __name__ == "__main__":
    # Example list of hyperparameter dictionaries
    base_hyperparameters = {
        "data_path": "/project/compbio-lab/encode_data/",
        "dropout": 0.1,
        "pool_size": 2,
        "epochs": 1,
        "inner_epochs": 1,
        "mask_percentage": 0.1,
        "num_loci": 10,
        "lr_halflife": 1,
        "min_avail": 1,
        "eic":True,
    }
    hyperparameter_space = [
        {"n_cnn_layers": 3, "conv_kernel_size": 5, "expansion_factor": 2, 
        "nhead": 4, "n_sab_layers": 2, "context_length": 400, "pos_enc": "relative", 
        "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #Base
        
        {"n_cnn_layers": 2, "conv_kernel_size": 5, "expansion_factor": 2, 
        "nhead": 2, "n_sab_layers": 3, "context_length": 800, "pos_enc": "relative", 
        "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #Larger context length, fewer heads
        
        {"n_cnn_layers": 2, "conv_kernel_size": 3, "expansion_factor": 2, 
        "nhead": 4, "n_sab_layers": 2, "context_length": 1600, "pos_enc": "relative", 
        "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #Max context length with fewer SAB layers
        
        {"n_cnn_layers": 2, "conv_kernel_size": 5, "expansion_factor": 3, 
        "nhead": 3, "n_sab_layers": 4, "context_length": 800, "pos_enc": "relative", 
        "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #Max expansion factor, moderate context:
        
        {"n_cnn_layers": 4, "conv_kernel_size": 5, "expansion_factor": 2, 
        "nhead": 2, "n_sab_layers": 1, "context_length": 1200, "pos_enc": "relative", 
        "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #More CNN layers, reduced SAB and heads:
        
        {"n_cnn_layers": 3, "conv_kernel_size": 3, "expansion_factor": 2, 
        "nhead": 4, "n_sab_layers": 3, "context_length": 600, "pos_enc": "relative", 
        "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #Moderate all settings:
        
        {"n_cnn_layers": 1, "conv_kernel_size": 7, "expansion_factor": 2, 
        "nhead": 8, "n_sab_layers": 6, "context_length": 400, "pos_enc": "relative", 
        "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #Fewer CNN layers, max heads and SAB layers
        
        {"n_cnn_layers": 2, "conv_kernel_size": 5, "expansion_factor": 2, 
        "nhead": 4, "n_sab_layers": 6, "context_length": 200, "pos_enc": "relative", 
        "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #Small context, max SAB layers
        
        {"n_cnn_layers": 2, "conv_kernel_size": 3, "expansion_factor": 3, 
        "nhead": 3, "n_sab_layers": 2, "context_length": 1000, "pos_enc": "relative", 
        "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #Max expansion factor, fewer heads

        {"n_cnn_layers": 2, "conv_kernel_size": 3, "expansion_factor": 3, 
        "nhead": 3, "n_sab_layers": 2, "context_length": 1000, "pos_enc": "relative", 
        "batch_size": 50, "learning_rate": 1e-3, "dna":True}, #Max expansion factor, fewer heads
        
        {"n_cnn_layers": 5, "conv_kernel_size": 3, "expansion_factor": 2, 
        "nhead": 8, "n_sab_layers": 4, "context_length": 2000, "pos_enc": "relative", 
        "batch_size": 50, "learning_rate": 1e-3, "dna":True} #Largest
        ]
    
    hyperparameters_list = []
    for s in hyperparameter_space:
        merged_dict = base_hyperparameters | s
        hyperparameters_list.append(merged_dict)

    # Make sure to set the correct start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Distribute and train models across GPUs
    results = distribute_models_across_gpus(hyperparameters_list)

    # Print results
    for result in results:
        print(f"Results for hyperparameters: {result['hyper_parameters']}")
        print(f"Metrics: {result['metrics']}")
