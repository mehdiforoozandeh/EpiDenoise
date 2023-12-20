import torch, math, random, time, json, os, pickle
from datetime import datetime
from torch import nn
import torch.optim as optim
from data import ENCODE_IMPUTATION_DATASET
import torch.nn.functional as F
import pandas as pd
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

class RelativePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(RelativePositionalEncoder, self).__init__()
        self.max_position = max_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_position * 2 + 1, emb_dim))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, seq_len_q, seq_len_k):
        range_vec_q = torch.arange(seq_len_q)
        range_vec_k = torch.arange(seq_len_k)
        relative_matrix = range_vec_k[None, :] - range_vec_q[:, None]
        clipped_relative_matrix = torch.clamp(relative_matrix, -self.max_position, self.max_position)
        relative_position_matrix = clipped_relative_matrix + self.max_position
        embeddings = self.embeddings_table[relative_position_matrix]

        return embeddings

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

#________________________________________________________________________________________________________________________#
class MaskedLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MaskedLinear, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        # Xavier initialization
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x, mask):
        masked_weight = self.weights * mask
        output = torch.matmul(x, masked_weight) + self.bias
        return output

#________________________________________________________________________________________________________________________#

class EpiDenoise(nn.Module): 
    def __init__(self, input_dim, nhead, d_model, nlayers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise, self).__init__()
        
        self.masked_linear = MaskedLinear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=context_length)  # or RelativePositionEncoding(input_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(d_model, output_dim)
        
    def forward(self, src, pmask, fmask):
        src = self.masked_linear(src, fmask)
        src = torch.permute(src, (1, 0, 2)) # to L, N, F
        src = self.pos_encoder(src)

        src = self.transformer_encoder(src)#, src_key_padding_mask=pmask) 
        src = self.decoder(src)
        src = torch.permute(src, (1, 0, 2))
        return src

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def reshape_tensor(tensor, context_length_factor):
    # Get the original size of the tensor
    samples, seq_length, features = tensor.size()

    # Calculate the new sequence length and number of samples
    new_seq_length = int(seq_length * context_length_factor)
    new_samples = int(samples / context_length_factor)

    # Check if the new sequence length is valid
    if seq_length % new_seq_length != 0:
        raise ValueError("The context_length_factor does not evenly divide the sequence length")

    # Reshape the tensor
    reshaped_tensor = tensor.view(new_samples, new_seq_length, features)

    return reshaped_tensor

# Function to mask a certain percentage of the data
def mask_data(data, mask_value=-1, chunk=False, n_chunks=1, mask_percentage=0.15):
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)
    seq_len = data.size(1)
    
    if chunk:
        # Calculate the size of each chunk
        chunk_size = int(mask_percentage * seq_len / n_chunks)
    else: 
        chunk_size = 1
        n_chunks =  int(mask_percentage * seq_len)

    # Initialize an empty list to store the start indices
    start_indices = []
    while len(start_indices) < n_chunks:
        # Generate a random start index
        start = torch.randint(0, seq_len - chunk_size, (1,))
        # Check if the chunk overlaps with any existing chunks
        if not any(start <= idx + chunk_size and start + chunk_size >= idx for idx in start_indices):
            # If not, add the start index to the list
            start_indices.append(start.item())

    # Loop over the start indices
    for start in start_indices:
        # Calculate the end index for the current chunk
        end = start + chunk_size
        # Set the mask values for the current chunk to True
        mask[:, start:end, :] = True

    # Create a copy of the data tensor
    masked_data = data.clone()
    # Set the masked data values to the mask_value
    masked_data[mask] = mask_value
    # Return the masked data and the mask

    return masked_data, mask#[:,:,0]

def sequence_pad(data, max_length, pad_value=-1):
    # Get the original dimensions of the data
    original_size = data.size()
    
    # Create a tensor filled with the pad value with the desired size
    padded_data = torch.full((original_size[0], max_length, original_size[2]), pad_value)
    
    # Copy the original data into the padded data tensor
    padded_data[:, :original_size[1], :] = data
    
    # Create a boolean mask indicating whether each value is padded or not
    pad_mask = padded_data == pad_value
    
    return padded_data, pad_mask

def train_model(
    model, dataset, criterion, optimizer, d_model, scheduler, arcsinh_transform=True,
    num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, 
    context_length=2000, batch_size=100, start_ds=0):
    
    log_strs = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)

    # model.to(device)
    log_strs.append(str(device))
    log_strs.append(f"# model_parameters: {count_parameters(model)}")
    logfile = open("models/log.txt", "w")
    logfile.write("\n".join(log_strs))
    logfile.close()

    ds=0
    # Define your batch size
    for ds_path in dataset.preprocessed_datasets:
        ds+=1
        
        if ds < start_ds:
            continue
        
        print('-_-' * 10)
        x, missing_mask, missing_f_pattern = dataset.get_dataset_pt(ds_path)
        num_features = x.shape[2]

        if arcsinh_transform:
            x = torch.arcsinh_(x)
        
        for epoch in range(0, num_epochs):
            print('-' * 10)
            print(f'Epoch {epoch+1}/{num_epochs}')

            # zero grads before going over all batches and all patterns of missing data
            optimizer.zero_grad()
            epoch_loss = []
            t0 = datetime.now()

            
            p = 0
            for pattern, indices in missing_f_pattern.items():
                p += 1

                pattern_batch = x[indices]
                missing_mask_patten_batch = missing_mask[indices]
                fmask = torch.ones(num_features, d_model)

                for i in pattern:
                    fmask[i,:] = 0

                fmask = fmask.to(device)

                # print(pattern_batch.shape, (fmask.sum(dim=1) > 0).sum().item(), len(pattern))

                if context_length < pattern_batch.shape[1]:
                    context_length_factor = context_length / pattern_batch.shape[1]

                    pattern_batch = reshape_tensor(pattern_batch, context_length_factor)
                    missing_mask_patten_batch = reshape_tensor(missing_mask_patten_batch, context_length_factor)

                # Break down x into smaller batches
                for i in range(0, len(pattern_batch), batch_size):
                    torch.cuda.empty_cache()
                    
                    x_batch = pattern_batch[i:i+batch_size]
                    missing_mask_batch = missing_mask_patten_batch[i:i+batch_size]

                    # Masking a subset of the input data
                    masked_x_batch, cloze_mask = mask_data(x_batch, mask_value=-1, chunk=chunk, n_chunks=n_chunks, mask_percentage=mask_percentage)
                    pmask = cloze_mask[:,:,0].squeeze()

                    cloze_mask = cloze_mask & ~missing_mask_batch
                    x_batch = x_batch.to(device)
                    masked_x_batch = masked_x_batch.to(device)
                    pmask = pmask.to(device)
                    cloze_mask = cloze_mask.to(device)

                    outputs = model(masked_x_batch, pmask, fmask)
                    loss = criterion(outputs[cloze_mask], x_batch[cloze_mask])

                    mean_pred, std_pred = outputs[cloze_mask].mean().item(), outputs[cloze_mask].std().item()
                    mean_target, std_target = x_batch[cloze_mask].mean().item(), x_batch[cloze_mask].std().item()

                    if torch.isnan(loss).sum() > 0:
                        skipmessage = "Encountered nan loss! Skipping batch..."
                        log_strs.append(skipmessage)
                        print(skipmessage)
                        del x_batch
                        del pmask
                        del masked_x_batch
                        del outputs
                        torch.cuda.empty_cache()
                        continue

                    del x_batch
                    del pmask
                    del masked_x_batch
                    del outputs
                    epoch_loss.append(loss.item())

                    # Clear GPU memory again
                    torch.cuda.empty_cache()

                    loss.backward()  
                
                if p%8 == 0:
                    logfile = open("models/log.txt", "w")

                    logstr = [
                        f"DataSet #{ds}/{len(dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', f'Missing Pattern {p}/{len(missing_f_pattern)}', 
                        f"Loss: {loss.item():.4f}", 
                        f"Mean_P: {mean_pred:.3f}", f"Mean_T: {mean_target:.3f}", 
                        f"Std_P: {std_pred:.2f}", f"Std_T: {std_target:.2f}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)
                
            # update parameters over all batches and all patterns of missing data
            optimizer.step()
            scheduler.step()

            t1 = datetime.now()
            logfile = open("models/log.txt", "w")

            logstr = [
                f"DataSet #{ds}/{len(dataset.preprocessed_datasets)}", 
                f'Epoch {epoch+1}/{num_epochs}', 
                f"Epoch Loss Mean: {np.mean(epoch_loss)}", 
                f"Epoch Loss std: {np.std(epoch_loss)}",
                f"Epoch took: {t1 - t0}"
                ]
            logstr = " | ".join(logstr)

            log_strs.append(logstr)
            logfile.write("\n".join(log_strs))
            logfile.close()
            print(logstr)

        # Save the model after each dataset
        try:
            torch.save(model.state_dict(), f'models/model_checkpoint_ds_{ds}.pth')
        except:
            pass

    return model

def train_epidenoise(hyper_parameters, checkpoint_path=None, start_ds=0):

    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    epochs = hyper_parameters["epochs"]
    mask_percentage = hyper_parameters["mask_percentage"]
    chunk = hyper_parameters["chunk"]
    context_length = hyper_parameters["context_length"]

    # one nucleosome is around 150bp -> 6bins
    # each chuck ~ 1 nucleosome

    n_chunks = (mask_percentage * context_length) // 4 # change it to 6 later

    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise(
        input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
        output_dim=output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.97)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()

    start_time = time.time()
    model = train_model(
        model, dataset, criterion, optimizer, scheduler=scheduler, d_model=d_model, num_epochs=epochs, 
        mask_percentage=mask_percentage, chunk=chunk, n_chunks=n_chunks,
        context_length=context_length, batch_size=batch_size, start_ds=start_ds)
    end_time = time.time()

    # Save the trained model
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))
    # os.system(f"mv models/hyper_parameters.pkl models/hyper_parameters_{model_name.replace( '.pt', '.pkl' )}")

    # Write a description text file
    description = {
        "hyper_parameters": hyper_parameters,
        "model_architecture": str(model),
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "number_of_model_parameters": count_parameters(model),
        "training_duration": int(end_time - start_time)
    }
    with open(os.path.join(model_dir, model_name.replace(".pt", ".txt")), 'w') as f:
        f.write(json.dumps(description, indent=4))

    return model

def load_epidenoise(model_path, hyper_parameters):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    context_length = hyper_parameters["context_length"]
    
    # Assuming model is an instance of the correct class
    model = EpiDenoise(
        input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
        output_dim=output_dim, dropout=dropout, context_length=context_length)

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
        
    return model

def predict(model, data, fmask, pmask):
    model.eval()  # set the model to evaluation mode
    
    with torch.no_grad():
        input_data = input_data.to(device)
        predictions = model(input_data, fmask, pmask)
        
    return predictions

# Calling the main function
if __name__ == "__main__":
    # EPIDENOISE-LARGE
    hyper_parameters_large = {
            "data_path": "/project/compbio-lab/EIC/training_data/",
            "input_dim": 35,
            "dropout": 0.1,
            "nhead": 8,
            "d_model": 128,
            "nlayers": 4,
            "epochs": 20,
            "mask_percentage": 0.15,
            "chunk": True,
            "context_length": 400,
            "batch_size": 80,
            "learning_rate": 0.01
        }

    # EPIDENOISE-SMALL
    hyper_parameters_small = {
        "data_path": "/project/compbio-lab/EIC/training_data/",
        "input_dim": 35,
        "dropout": 0.1,
        "nhead": 4,
        "d_model": 64,
        "nlayers": 2,
        "epochs": 30,
        "mask_percentage": 0.15,
        "chunk": True,
        "context_length": 400,
        "batch_size": 50,
        "learning_rate": 0.005
    }

    train_epidenoise(
        hyper_parameters_small, 
        checkpoint_path=None, 
        start_ds=0)