class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=8000):
        super(RelativePositionEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.rel_pos_emb = nn.Embedding(self.max_len*2, self.d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.rel_pos_emb(pos + self.max_len)
        return x + pos_emb

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attention_weights = self.softmax(self.attention(x))
        return (attention_weights * x).sum(dim=1)

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8000):
        super(_PositionalEncoding, self).__init__()

        self.d_model = d_model
        d_model_pad = d_model if d_model % 2 == 0 else d_model + 1  # Ensure d_model is even

        # Create a long enough `pe` matrix
        pe = torch.zeros(max_len, d_model_pad)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model_pad, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model_pad))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register `pe` as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe.squeeze(1)
        pe = pe[:,:self.d_model]

        return x + pe.unsqueeze(0)

class WeightedMSELoss(nn.Module): 
    # gives more weight to predicting larger signal values rather than depletions
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target):
        weights = target.clone().detach()  # Create a copy of target for weights
        max_val = weights.max()
        if max_val != 0:
            weights = weights / max_val  # Normalize weights to be between 0 and 1
            return torch.sum(weights * ((input - target) ** 2))
        else:
            return torch.sum((input - target) ** 2)

class DoubleMaskMultiHeadedAttention(torch.nn.Module):
    
    def __init__(self, heads, d_model, dropout=0.1):
        super(DoubleMaskMultiHeadedAttention, self).__init__()
        
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = MaskedLinear(d_model, d_model)
        self.key = MaskedLinear(d_model, d_model)
        self.value = MaskedLinear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, pmask, fmask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """

        # fmask should be of size d_model * d_model 
        # for each feature index i, if i-th feature is missing fmask[i,:]=0 ; otherwise, fmask[i,:]=1

        # Element-wise multiplication with the weight matrices
        # print("1", torch.sum(self.query.weight.data == 0).item(), self.query.weight.data.sum().item())

        # self.query.weight.data *= fmask
        # self.key.weight.data *= fmask
        # self.value.weight.data *= fmask

        # print("2", torch.sum(self.query.weight.data == 0).item(), self.query.weight.data.sum().item())

        # Element-wise multiplication of mask with the bias terms
        # bias_fmask = fmask.diag()
        # self.query.bias.data *= bias_fmask
        # self.key.bias.data *= bias_fmask
        # self.value.bias.data *= bias_fmask

        # (batch_size, max_len, d_model)
        query = self.query(query, fmask)
        key = self.key(key, fmask)        
        value = self.value(value, fmask)   
        
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        scores = scores.masked_fill(pmask == 0, -1e9)    

        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)           
        weights = self.dropout(weights)

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model)
        return self.output_linear(context)

class FeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

class DoubleMaskEncoderLayer(torch.nn.Module):
    def __init__(
        self, 
        d_model=35,
        heads=5, 
        feed_forward_hidden=35 * 4, 
        dropout=0.1
        ):
        super(DoubleMaskEncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = DoubleMaskMultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, pmask, fmask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, pmask, fmask))
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded


class MaskedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(MaskedConv1d, self).__init__()
        padding = (kernel_size - 1) // 2 #same
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x, mask):
        not_mask = mask.clone()
        not_mask = ~mask
        x = x.permute(0,2,1)
        not_mask = not_mask.permute(0,2,1)
        x = x * not_mask
        x = self.conv(x)
        x = x.permute(0,2,1)
        return x

class MaskPostConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(MaskPostConv1d, self).__init__()
        padding = (kernel_size - 1) // 2 #same
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x, mask):
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0,2,1)
        not_mask = mask.clone()
        not_mask = ~mask
        x = x * not_mask
        return x

class DualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DualConv1d, self).__init__()
        padding = (kernel_size - 1) // 2 #same
        self.data_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, mask):
        mask = mask.clone()

        x = x.permute(0,2,1)
        mask = mask.permute(0,2,1)

        x = self.data_conv(x)
        mask = self.mask_conv(mask.float())

        x = x * mask
        x = x.permute(0,2,1)
        return x

class TripleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, seq_len, stride=1):
        super(TripleConv1d, self).__init__()
        padding = (kernel_size - 1) // 2 #same

        # in_channel: num_features, out_channel: num_filters
        # output shape: (batch_size, num_filters, seq_len)
        self.data_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding) 

        # in_channel: num_features, out_channel: num_filters
        # output shape: (batch_size, num_filters, seq_len)
        self.position_mask_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        # in_channel: seq_len, out_channel: num_features
        # output shape: (batch_size, 1, num_features)
        self.feature_mask_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        

    def forward(self, x, mask):
        mask = mask.clone()
        # x = x.permute(0,2,1)
        mask = mask.permute(0,2,1)

        # x = self.data_conv(x)
        # mask = self.mask_conv(mask.float()).permute(0,2,1)  # transpose the mask convolutions back to original shape

        x = x * mask
        x = x.permute(0,2,1)
        return x


#________________________________________________________________________________________________________________________#
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim):
        super(TransformerEncoder, self).__init__()

        self.pos_encoder = PositionalEncoding(input_dim, max_len=500)  # or RelativePositionEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)
        
    def forward(self, src, src_mask):
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

class MaskedConvEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim, num_filters, kernel_size=5):
        super(MaskedConvEncoder, self).__init__()
        self.masked_conv = MaskedConv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1)
        self.pos_encoder = PositionalEncoding(input_dim)  # or RelativePositionEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)
        
    def forward(self, src, src_mask):
        src = self.masked_conv(src, src_mask)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

class MaskPostConvEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim, num_filters, kernel_size=5):
        super(MaskPostConvEncoder, self).__init__()
        self.masked_conv = MaskPostConv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1)
        self.pos_encoder = PositionalEncoding(input_dim)  # or RelativePositionEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)
        
    def forward(self, src, src_mask):
        src = self.masked_conv(src, src_mask)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

class DualConvEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim, num_filters, kernel_size=5):
        super(DualConvEncoder, self).__init__()
        
        self.dualconv = DualConv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, stride=1)
        self.pos_encoder = PositionalEncoding(input_dim, max_len=500)  # or RelativePositionEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, src, src_mask):
        src = self.dualconv(src, src_mask)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

class TripleConvEncoder(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, output_dim, num_filters, seq_len, kernel_size=5):
        super(DualConvEncoder_T, self).__init__()
        
        self.dualconv = DualConv1d_T(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size, seq_len=seq_len, stride=1)
        self.pos_encoder = PositionalEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, src, src_mask):
        src = self.dualconv(src, src_mask)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src)
        src = self.decoder(src)
        return src

def mask_missing(data, missing_features_ind, mask_value=-1):
    mask = torch.zeros_like(data, dtype=torch.bool)

    # Loop over the missing feature ids
    for id in missing_features_ind:
        # Set the mask values for the current chunk to True
        mask[:, :, id] = True

    # Create a copy of the data tensor
    masked_data = data.clone()
    # Set the masked data values to the mask_value
    masked_data[mask] = mask_value
    # Return the masked data and the mask
    return masked_data, mask


def __train_model(model, dataset, criterion, optimizer, num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, context_length=2000, batch_size=100, start_epoch=0):
    log_strs = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)

    # model.to(device)
    log_strs.append(str(device))
    logfile = open("models/log.txt", "w")
    logfile.write("\n".join(log_strs))
    logfile.close()

    # Define your batch size
    for epoch in range(start_epoch, num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch+1}/{num_epochs}')

        bb=0
        for bios, f in dataset.biosamples.items():
            bb+=1
            print('-' * 10)
            x, missing_mask, missing_f_i = dataset.get_biosample(f)

            # fmask is used to mask QKV of transformer
            num_features = x.shape[2]
            fmask = torch.ones(num_features, num_features)

            for i in missing_f_i:
                fmask[i,:] = 0
            
            fmask = fmask.to(device)
            # Break down x into smaller batches
            for i in range(0, len(x), batch_size):
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                
                x_batch = x[i:i+batch_size]
                missing_mask_batch = missing_mask[i:i+batch_size]

                if context_length < 8000:
                    rand_start = random.randint(0, 8000 - (context_length+1))
                    rand_end = rand_start + context_length

                    x_batch, missing_mask_batch = x_batch[:, rand_start:rand_end, :], missing_mask_batch[:, rand_start:rand_end, :]

                # print("missing_mask_batch   ", missing_mask_batch.shape, missing_mask_batch.sum(), len(missing_f_i))

                # Masking a subset of the input data
                masked_x_batch, cloze_mask = mask_data(x_batch, mask_value=-1, chunk=chunk, n_chunks=n_chunks, mask_percentage=mask_percentage)
                pmask = cloze_mask[:,:,0].unsqueeze(1).unsqueeze(1)
                # print("pmask1    ", pmask.shape, pmask.sum())

                # print("cloze_mask1    ", cloze_mask.shape, cloze_mask.sum())
                cloze_mask = cloze_mask & ~missing_mask_batch
                # print("cloze_mask2    ", cloze_mask.shape, cloze_mask.sum())

                # Convert the boolean values to float and switch the masked and non-masked values
                pmask = 1 - pmask.float()
                # print("pmask2    ", pmask.shape, pmask.sum())
                

                # print("x_batch  ", x_batch[cloze_mask].shape, x_batch[cloze_mask].mean().item(), x_batch[cloze_mask].min().item(), x_batch[cloze_mask].max().item())
                # print("masked_x_batch   ", masked_x_batch[cloze_mask].shape, masked_x_batch[cloze_mask].mean().item(), masked_x_batch[cloze_mask].min().item(), masked_x_batch[cloze_mask].max().item())

                x_batch = x_batch.to(device)
                masked_x_batch = masked_x_batch.to(device)
                pmask = pmask.to(device)
                cloze_mask = cloze_mask.to(device)

                outputs = model(masked_x_batch, pmask, fmask)
                loss = criterion(outputs[cloze_mask], x_batch[cloze_mask])


                sum_pred, sum_target = outputs[cloze_mask].sum().item(), x_batch[cloze_mask].sum().item()

                if torch.isnan(loss).sum() > 0:
                    skipmessage = "Encountered nan loss! Skipping batch..."
                    log_strs.append(skipmessage)
                    print(skipmessage)
                    continue

                del x_batch
                del pmask
                del masked_x_batch
                del outputs

                # Clear GPU memory again
                torch.cuda.empty_cache()

                if (((i//batch_size))+1) % 10 == 0 or i==0:
                    logfile = open("models/log.txt", "w")

                    logstr = f'Epoch {epoch+1}/{num_epochs} | Bios {bb}/{len(dataset.biosamples)}| Batch {((i//batch_size))+1}/{(len(x)//batch_size)+1}\
                        | Loss: {loss.item():.4f} | S_P: {sum_pred:.1f} | S_T: {sum_target:.1f}'

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                loss.backward()                    
                optimizer.step()
        
        # Save the model after each epoch
        torch.save(model.state_dict(), f'models/model_checkpoint_epoch_{epoch+1}.pth')

    return model

def _train_model(model, dataset, criterion, optimizer, d_model, num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, context_length=2000, batch_size=100, start_bios=0):
    log_strs = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)

    # model.to(device)
    log_strs.append(str(device))
    logfile = open("models/log.txt", "w")
    logfile.write("\n".join(log_strs))
    logfile.close()

    bb=0
    # Define your batch size
    for bios, f in dataset.biosamples.items():
        bb+=1
        if bb < start_bios:
            continue

        print('-' * 10)
        x, missing_mask, missing_f_i = dataset.get_biosample(f)

        # fmask is used to mask QKV of transformer
        num_features = x.shape[2]
        fmask = torch.ones(num_features, d_model)

        for i in missing_f_i:
            fmask[i,:] = 0
        
        fmask = fmask.to(device)
        for epoch in range(0, num_epochs):
            print('-' * 10)
            print(f'Epoch {epoch+1}/{num_epochs}')
            optimizer.zero_grad()
            # Break down x into smaller batches
            for i in range(0, len(x), batch_size):
                torch.cuda.empty_cache()
                
                x_batch = x[i:i+batch_size]
                missing_mask_batch = missing_mask[i:i+batch_size]

                if context_length < 8000:
                    rand_start = random.randint(0, 8000 - (context_length+1))
                    rand_end = rand_start + context_length

                    x_batch, missing_mask_batch = x_batch[:, rand_start:rand_end, :], missing_mask_batch[:, rand_start:rand_end, :]

                # print("missing_mask_batch   ", missing_mask_batch.shape, missing_mask_batch.sum(), len(missing_f_i))

                x_batch = torch.arcsinh_(x_batch)

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

                # Clear GPU memory again
                torch.cuda.empty_cache()

                if (((i//batch_size))+1) % 10 == 0 or i==0:
                    logfile = open("models/log.txt", "w")

                    logstr = [
                        f'Epoch {epoch+1}/{num_epochs}', f"Bios {bb}/{len(dataset.biosamples)}", 
                        f"Batch {((i//batch_size))+1}/{(len(x)//batch_size)+1}",
                        f"Loss: {loss.item():.4f}", 
                        f"Mean_P: {mean_pred:.3f}", f"Mean_T: {mean_target:.3f}", 
                        f"Std_P: {std_pred:.2f}", f"Std_T: {std_target:.2f}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                loss.backward()     

            optimizer.step()

        # Save the model after each epoch
        try:
            torch.save(model.state_dict(), f'models/model_checkpoint_bios_{bb}.pth')
        except:
            pass

    return model


def eDICE_eval():
    e = Evaluation(
        model_path= "models/EpiDenoise_20231210014829_params154531.pt", 
        hyper_parameters_path= "models/hyper_parameters_EpiDenoise_20231210014829_params154531.pkl", 
        traindata_path="/project/compbio-lab/EIC/training_data/", 
        evaldata_path="/project/compbio-lab/EIC/validation_data/", 
        is_arcsin=True
    )

    preds_dir = "/project/compbio-lab/EIC/mehdi_preds/scratch/"
    obs_dir1 = "/project/compbio-lab/EIC/validation_data/"
    obs_dir2 = "/project/compbio-lab/EIC/blind_data/"

    results = []

    for pf in os.listdir(preds_dir):
        name = pf.replace(".pkl","")
        assay = name[3:]
        ct = name[:3]
        print(ct, assay)

        with open(preds_dir + pf, 'rb') as pf_file:
            pred = pickle.load(pf_file)
            pred = np.sinh(pred)
        
        if pf.replace(".pkl", ".bigwig") in os.listdir(obs_dir1):
            target = torch.load(obs_dir1 + f"/{ct}_chr21_25.pt")
            target = target[:, int(assay.replace("M", "")) - 1].numpy()

        elif pf.replace(".pkl", ".bigwig") in os.listdir(obs_dir2):
            target = torch.load(obs_dir2 + f"/{ct}_chr21_25.pt")
            target = target[:, int(assay.replace("M", "")) - 1].numpy()

        print(pf, target.sum(), pred.sum())
        metrics = {
                'celltype': ct,
                'feature': assay,

                'MSE-GW': e.mse(target, pred),
                'Pearson-GW': e.pearson(target, pred),
                'Spearman-GW': e.spearman(target, pred),

                'MSE-1obs': e.mse1obs(target, pred),
                'Pearson_1obs': e.pearson1_obs(target, pred),
                'Spearman_1obs': e.spearman1_obs(target, pred),

                'MSE-1imp': e.mse1imp(target, pred),
                'Pearson_1imp': e.pearson1_imp(target, pred),
                'Spearman_1imp': e.spearman1_imp(target, pred),

                'MSE-gene': e.mse_gene(target, pred),
                'Pearson_gene': e.pearson_gene(target, pred),
                'Spearman_gene': e.spearman_gene(target, pred),

                'MSE-prom': e.mse_prom(target, pred),
                'Pearson_prom': e.pearson_prom(target, pred),
                'Spearman_prom': e.spearman_prom(target, pred),
            }
        

        results.append(metrics)

    results = pd.DataFrame(results)
    results.to_csv("eDICE_results.csv", index=False)