
class PROCESS_EIC_DATA(object):
    def __init__(self, path, max_len=8000, resolution=25, stratified=False):
        self.path = path
        self.stratified = stratified
        self.resolution = resolution
        self.max_len = max_len * self.resolution #converts max_len from #bins to #bp
        self.util = COORD(resolution=self.resolution, Meuleman_file="_", outdir=self.path)
        self.genomesize = sum(list(self.util.chr_sizes.values()))
        self.all_assays = ['M{:02d}'.format(i) for i in range(1, 36)]

        self.biosamples = {}
        for f in os.listdir(self.path):
            if ".bigwig" in f: 
                if f[:3] not in self.biosamples.keys():
                    self.biosamples[f[:3]] = {}
                    
                self.biosamples[f[:3]][f[3:6]] = pyBigWig.open(self.path + "/" + f)

    def pkl_generate_m_samples(self, m, multi_p=True, n_p=20): # m per biosample           
        if self.stratified:
            self.util.get_foreground()
            df = self.util.foreground
            df = df[df["chrom"].isin(self.util.chr_sizes.keys())]
            m_regions = []
            used_regions = {chr: [] for chr in df['chrom'].unique()}

            # Sort the DataFrame by chromosome and start position
            df = df.sort_values(['chrom', 'start'])

            # Select m/2 regions from the DataFrame
            for _ in range(m // 2):
                while True:
                    # Select a random row from the DataFrame
                    row = df.sample(1).iloc[0]

                    # Generate a start position that is divisible by self.resolution and within the region
                    rand_start = random.randint(row['start'] // self.resolution, (row['end']) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len


                    # Check if the region overlaps with any existing region in the same chromosome
                    if rand_start >= 0 and rand_end <= self.util.chr_sizes[row['chrom']]:
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[row['chrom']]):
                            m_regions.append([row['chrom'], rand_start, rand_end])
                            used_regions[row['chrom']].append((rand_start, rand_end))
                            break
                        
            # Select m/2 regions that are not necessarily in the DataFrame 
            for chr, size in self.util.chr_sizes.items():
                m_c = int((m // 2) * (size / self.genomesize))  # Calculate the number of instances from each chromosome proportional to its size
                for _ in range(m_c):
                    while True:
                        # Generate a random start position that is divisible by self.resolution
                        rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                        rand_end = rand_start + self.max_len

                        # Check if the region overlaps with any existing region
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                            m_regions.append([chr, rand_start, rand_end])
                            used_regions[chr].append((rand_start, rand_end))
                            break

        else:
            m_regions = []
            used_regions = {chr: [] for chr in self.util.chr_sizes.keys()}

            for chr, size in self.util.chr_sizes.items():
                m_c = int(m * (size / self.genomesize))

                for _ in range(m_c):
                    while True:
                        # Generate a random start position that is divisible by self.resolution
                        rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                        rand_end = rand_start + self.max_len

                        # Check if the region overlaps with any existing region in the same chromosome
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                            m_regions.append([chr, rand_start, rand_end])
                            used_regions[chr].append((rand_start, rand_end))
                            break

        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in self.biosamples[bios].keys():
                bios_data[assay] = []

                bw = self.biosamples[bios][assay]
                bw_query_dicts = []
                for i in range(len(m_regions)):
                    r = m_regions[i]
                    bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})

                if multi_p:
                    with mp.Pool(n_p) as p:
                        m_signals = p.map(get_bin_value, bw_query_dicts)
                    
                    for i in range(len(m_signals)):
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], m_signals[i]))
                else:
                    for i in range(len(bw_query_dicts)):
                        signals = get_bin_value(bw_query_dicts[i])
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], signals))
                    
            file_path = f"{self.path}/{bios}_m{m}_{self.resolution}bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")

    def generate_m_samples(self, m, n_datasets=50, multi_p=True, n_p=10):
        if self.stratified:
            self.util.get_foreground()
            df = self.util.foreground
            df = df[df["chrom"].isin(self.util.chr_sizes.keys())]
            m_regions = []
            used_regions = {chr: [] for chr in df['chrom'].unique()}

            # Sort the DataFrame by chromosome and start position
            df = df.sort_values(['chrom', 'start'])

            # Select m/2 regions from the DataFrame
            while len(m_regions) < (m // 2):
                while True:
                    # Select a random row from the DataFrame
                    row = df.sample(1).iloc[0]

                    # Generate a start position that is divisible by self.resolution and within the region
                    rand_start = random.randint(row['start'] // self.resolution, (row['end']) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region in the same chromosome
                    if rand_start >= 0 and rand_end <= self.util.chr_sizes[row['chrom']]:
                        if not any(start <= rand_end and end >= rand_start for start, end in used_regions[row['chrom']]):
                            m_regions.append([row['chrom'], rand_start, rand_end])
                            used_regions[row['chrom']].append((rand_start, rand_end))
                            break
                        
            # Select m/2 regions that are not necessarily in the DataFrame 
            for chr, size in self.util.chr_sizes.items():
                m_c = int((m // 2) * (size / self.genomesize))  # Calculate the number of instances from each chromosome proportional to its size
                mii = 0
                while mii < m_c:
                    # Generate a random start position that is divisible by self.resolution
                    rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region
                    if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                        m_regions.append([chr, rand_start, rand_end])
                        used_regions[chr].append((rand_start, rand_end))
                        mii += 1 
                        break

        else:
            m_regions = []
            used_regions = {chr: [] for chr in self.util.chr_sizes.keys()}

            for chr, size in self.util.chr_sizes.items():
                m_c = int(m * (size / self.genomesize))
                mii = 0

                while mii < m_c:
                    # Generate a random start position that is divisible by self.resolution
                    rand_start = random.randint(0, (size - self.max_len) // self.resolution) * self.resolution
                    rand_end = rand_start + self.max_len

                    # Check if the region overlaps with any existing region in the same chromosome
                    if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                        m_regions.append([chr, rand_start, rand_end])
                        used_regions[chr].append((rand_start, rand_end))
                        mii += 1 

        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        ds_number = 0  
        print("m2:   ", len(m_regions))
        samples_per_ds = len(m_regions) // n_datasets
        for ds_i in range(0, len(m_regions), samples_per_ds):
            ds_number += 1

            ds_i_regions = m_regions[ds_i : (ds_i + samples_per_ds)]
            ds_i_regions.sort(key=lambda x: x[1]) # sorted based on start coord
            
            all_samples_tensor = []

            for bios in self.biosamples.keys():
                print("     ct:   ", bios)
                bios_data = {}

                for assay in self.all_assays:
                    bios_data[assay] = []

                    if assay in self.biosamples[bios].keys(): # if available
                        print("         assay:   ", assay)
                        bw = self.biosamples[bios][assay]
                        bw_query_dicts = []

                        for r in ds_i_regions:
                            bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})
                        
                        if multi_p:
                            with mp.Pool(n_p) as p:
                                outs = p.map(get_bin_value_dict, bw_query_dicts)
                        else:
                            outs = []
                            for ii in range(len(bw_query_dicts)):
                                outs.append(get_bin_value_dict(bw_query_dicts[ii]))

                        outs.sort(key=lambda x: x['start']) # assert is sorted based on start coord
                        m_signals = [o["signals"] for o in outs]
                        
                        for sample in m_signals:
                            bios_data[assay].append(sample)

                    else: # if missing
                        for r in ds_i_regions:
                            bios_data[assay].append([-1 for _ in range(self.max_len // self.resolution)])
                
                # Convert bios_data to a numpy array
                bios_data_array = np.array([bios_data[assay] for assay in self.all_assays], dtype=np.float32)

                # Add bios_data_array to all_samples
                all_samples_tensor.append(bios_data_array)

            # Convert all_samples to a numpy array
            all_samples_tensor = np.array(all_samples_tensor)

            # Convert all_samples_array to a PyTorch tensor
            all_samples_tensor = torch.from_numpy(all_samples_tensor)

            # Ensure the tensor is of type float
            all_samples_tensor = all_samples_tensor.float()

            all_samples_tensor = torch.permute(all_samples_tensor, (2, 0, 3, 1))
            # Get the shape of the current tensor
            shape = all_samples_tensor.shape

            # Calculate the new dimensions
            new_shape = [shape[0]*shape[1]] + list(shape[2:])

            # Reshape the tensor
            all_samples_tensor = all_samples_tensor.reshape(new_shape)
            
            file_path = f"{self.path}/mixed_dataset{ds_number}_{m//n_datasets}samples_{self.resolution}bp.pt"
            torch.save(all_samples_tensor, file_path)
            print(f"saved DS # {ds_number}, with shape {all_samples_tensor.shape}")

    def load_m_regions(self, file_path):
        # Open the gzip file
        with gzip.open(file_path, 'rb') as f:
            # Load the data using pickle
            bios_data = pickle.load(f)

        # Initialize an empty list to store the m_regions
        m_regions = []

        # Iterate over each biosample and assay
        for sample in bios_data[list(bios_data.keys())[0]]:

            # Append the regions to the m_regions list
            if sample[0] not in m_regions:
                m_regions.append(sample[0])
            
        return m_regions
    
    def generate_m_samples_from_predefined_regions(self, m_regions, multi_p=True, n_p=100):
        m = len(m_regions)
        if multi_p:
            bw_obj = False
            # rewrite biosample-assay dirs instead of obj
            self.biosamples = {}
            for f in os.listdir(self.path):
                if ".bigwig" in f: 
                    if f[:3] not in self.biosamples.keys():
                        self.biosamples[f[:3]] = {}
                        
                    self.biosamples[f[:3]][f[3:6]] = self.path + "/" + f
        else:
            bw_obj = True

        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in self.biosamples[bios].keys():
                bios_data[assay] = []

                bw = self.biosamples[bios][assay]
                bw_query_dicts = []
                for i in range(len(m_regions)):
                    r = m_regions[i]
                    bw_query_dicts.append({"bw":bw, "chr":r[0], "start":r[1], "end":r[2], "resolution": self.resolution, "bw_obj":bw_obj})

                if multi_p:
                    with mp.Pool(n_p) as p:
                        m_signals = p.map(get_bin_value, bw_query_dicts)
                    
                    for i in range(len(m_signals)):
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], m_signals[i]))
                else:
                    for i in range(len(bw_query_dicts)):
                        signals = get_bin_value(bw_query_dicts[i])
                        bios_data[assay].append((
                            [bw_query_dicts[i]["chr"], bw_query_dicts[i]["start"], bw_query_dicts[i]["end"]], signals))
                    
            file_path = f"{self.path}/{bios}_m{m}_{self.resolution}bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")
                
    def generate_wg_samples(self):
        for bios in self.biosamples.keys():
            bios_data = {}
            for assay in biosamples[bios].keys():
                bios_data[assay] = {}

                bw = biosamples[bios][assay]
                for chr, size in self.util.chr_sizes.items():
                    signals = get_bin_value(bw, chr, 0, size, self.resolution)
                    bios_data[assay][chr] = signals
            
            file_path = f"{self.path}/{bios}_WG_25bp.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(bios_data, f)
            os.system(f"gzip {file_path}")

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

class old__BAM_TO_SIGNAL(object):
    def __init__(self, resolution):
        """
        Initialize the object
        """
        self.resolution = resolution

    def read_chr_sizes(self):
        """
        Read a file with chromosome sizes and return a dictionary where keys are 
        chromosome names and values are chromosome sizes.
        
        Parameters:
        file_path (str): The path to the file with chromosome sizes.

        Returns:
        dict: A dictionary where keys are chromosome names and values are chromosome sizes.
        """

        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        
        self.chr_sizes = {}
        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

    def load_bam(self):
        """
        Load the BAM file using pysam.
        """
        self.bam = pysam.AlignmentFile(self.bam_file, 'rb')

    def initialize_empty_bins(self):
        """
        Initialize empty bins for each chromosome based on the resolution.
        """
        self.bins = {chr: [0] * (size // self.resolution + 1) for chr, size in self.chr_sizes.items()}

    def calculate_coverage(self):
        """
        Calculate the coverage for each bin.
        """
        self.bins = {chr: [0] * (self.chr_sizes[chr] // self.resolution + 1) for chr in self.chr_sizes}
        self.coverage = {}

        for chr in self.chr_sizes:
            self.coverage[chr] = {
                'chr': [],
                'start': [],
                'end': [],
                'read_count': []}

            # print(f"getting {chr} coverage...")
            for read in self.bam.fetch(chr):
                start_bin = read.reference_start // self.resolution
                end_bin = read.reference_end // self.resolution
                for i in range(start_bin, end_bin+1):
                    self.bins[chr][i] += 1

            for i, count in enumerate(self.bins[chr]):
                start = i * self.resolution
                end = start + self.resolution
                self.coverage[chr]["chr"].append(str(chr))
                self.coverage[chr]["start"].append(int(start))
                self.coverage[chr]["end"].append(int(end))
                self.coverage[chr]["read_count"].append(float(count))

    def calculate_signal_pvalues(self):
        """
        Calculate the per position signal p-value according to the MACS2 pipeline.
        """
        self.pvalues = {}

        # Calculate the mean coverage across all bins
        mean_coverage = np.mean([np.mean(self.coverage[chr]["read_count"]) for chr in self.coverage.keys()])

        for chr in self.coverage.keys():
            self.pvalues[chr] = {
                'chr': [],
                'start': [],
                'end': [],
                'pvalue': []}

            for i, count in enumerate(self.coverage[chr]["read_count"]):
                # Calculate the p-value of the Poisson distribution
                pvalue = 1 - poisson.cdf(count, mean_coverage)

                # Convert the p-value to -log10(p-value)
                pvalue = -np.log10(pvalue + 1e-19)

                self.pvalues[chr]["chr"].append(str(chr))
                self.pvalues[chr]["start"].append(self.coverage[chr]["start"][i])
                self.pvalues[chr]["end"].append(self.coverage[chr]["end"][i])
                self.pvalues[chr]["pvalue"].append(pvalue)

    def save_coverage_pkl(self):
        """
        Save the coverage data to a pickle file.

        Parameters:
        file_path (str): The path to the pickle file.
        """

        for chr in self.coverage.keys():
            file_path = self.bam_file.replace(".bam", f"_{chr}_cvrg{self.resolution}bp.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(self.coverage[chr], f)
        
            os.system(f"gzip {file_path}")
    
    def save_coverage_bigwig(self):
        """
        Save the coverage data to a BigWig file.

        Parameters:
        file_path (str): The path to the BigWig file.
        """
        file_path = self.bam_file.replace(".bam", f"_cvrg{self.resolution}bp.bw")
        bw = pyBigWig.open(file_path, 'w')
        bw.addHeader([(k, v) for k, v in self.chr_sizes.items()])

        for chr in self.coverage.keys():
            bw.addEntries(
                self.coverage[chr]["chr"], 
                self.coverage[chr]["start"], 
                ends=self.coverage[chr]["end"], 
                values=self.coverage[chr]["read_count"])
        bw.close()

    def save_signal_pkl(self):
        """
        Save the signal pval data to a pickle file.

        Parameters:
        file_path (str): The path to the pickle file.
        """

        for chr in self.pvalues.keys():
            file_path = self.bam_file.replace(".bam", f"_{chr}_signal{self.resolution}bp.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(self.pvalues[chr], f)
        
            os.system(f"gzip {file_path}")
    
    def save_signal_bigwig(self):
        """
        Save the signal pval data to a BigWig file.

        Parameters:
        file_path (str): The path to the BigWig file.
        """
        file_path = self.bam_file.replace(".bam", f"_signal{self.resolution}bp.bw")
        bw = pyBigWig.open(file_path, 'w')
        bw.addHeader([(k, v) for k, v in self.chr_sizes.items()])

        for chr in self.pvalues.keys():
            bw.addEntries(
                self.pvalues[chr]["chr"], 
                self.pvalues[chr]["start"], 
                ends=self.pvalues[chr]["end"], 
                values=self.pvalues[chr]["read_count"])
                
        bw.close()

    def full_preprocess(self, bam_file, chr_sizes_file, resolution=25):
        t0 = datetime.datetime.now()
        self.bam_file = bam_file
        self.chr_sizes_file = chr_sizes_file
        self.resolution = resolution

        self.read_chr_sizes()
        self.load_bam()
        self.initialize_empty_bins()
        self.calculate_coverage()
        self.calculate_signal_pvalues()

        self.save_coverage_pkl()
        self.save_signal_pkl()

        t1 = datetime.datetime.now()
        print(f"took {t1-t0} to get coverage for {bam_file} at resolution: {resolution}bp")

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

def mask_data(data, mask_value=-1, chunk=False, n_chunks=1, mask_percentage=0.15): # used for epidenoise 1.0
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

    return masked_data, mask

def mask_data15(data, mask_value=-1, chunk=False, n_chunks=1, mask_percentage=0.15): # used for epidenoise 1.5
    """
    in this version, we added special tokens and made sure not to mask them
    similar to BERT, using 3 different maskings:
        1. mask
        2. replace with random data
        3. do nothing
    """
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)
    seq_len = data.size(1)
    seglength = (seq_len - 3)/2

    cls_sep_indices = [0, seglength+1, 2*seglength + 2]
    
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
        if not any(start <= idx + chunk_size and start + chunk_size >= idx for idx in start_indices + cls_sep_indices):
            # If not, add the start index to the list
            start_indices.append(start.item())

    # Create a copy of the data tensor
    masked_data = data.clone()

    # Loop over the start indices
    for start in start_indices:
        # Calculate the end index for the current chunk
        end = start + chunk_size
        # Set the mask values for the current chunk to True
        mask[:, start:end, :] = True

        # For each position in the chunk, decide how to mask it
        for pos in range(start, end):
            rand_num = random.random()
            if rand_num < 0.8:
                # 80% of the time, replace with mask_value
                masked_data[:, pos, :] = mask_value
            elif rand_num < 0.9:
                # 10% of the time, replace with a random value in the range of the data
                data_min = 0
                data_max = torch.max(data)
                random_value = data_min + torch.rand(1) * (data_max - data_min)
                masked_data[:, pos, :] = random_value

    # Return the masked data and the mask
    return masked_data, mask

def mask_data16(data, available_features, mask_value=-1, chunk_size=6, mask_percentage=0.15): # used for epidenoise 1.6 and 1.7
    """
    dimensions of the data: (batch_size, context_length, features)
    in this version, we make the following changes
    find available features -> for unavailable features, are corresponding values are -1. 
    num_all_signals = context * num_available_features
    num_mask_start = (num_all_signals * mask_percentage) / chunk_size
    randomly select mask_start coordinates 
        length: axis2 (start + chunk_size -> no overlap with special tokens)
        feature: random.choice(available_features)
    """
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)
    if mask_percentage == 0:
        return data, mask

    seq_len = data.size(1)
    seglength = (seq_len - 3)/2

    special_tokens = [0, seglength+1, (2*seglength)+2]

    # Calculate total number of signals and number of chunks to be masked
    num_all_signals = data.size(1) * len(available_features)
    num_mask_start = int((num_all_signals * mask_percentage) / chunk_size)

    # Loop over the number of chunks to be masked
    for _ in range(num_mask_start):
        while True:
            # Randomly select start coordinates for the chunk
            length_start = torch.randint(0, seq_len - chunk_size, (1,))
            feature_start = available_features[torch.randint(0, len(available_features), (1,))]

            # Check if the chunk overlaps with any special tokens
            if not any(length_start <= idx < length_start+chunk_size for idx in special_tokens):
                break

        # Apply the masking to the selected chunk
        mask[:, length_start:length_start+chunk_size, feature_start] = True
        data[mask] = mask_value

    return data, mask

def mask_data18(data, available_features, mask_value=-1, mask_percentage=0.15):
    # Initialize a mask tensor with the same shape as the data tensor, filled with False
    mask = torch.zeros_like(data, dtype=torch.bool)

    if len(available_features) == 1:
        mask_percentage = 0

    if mask_percentage == 0:
        return data, mask

    seq_len = data.size(1)
    num_mask_features = int(len(available_features) * mask_percentage)
    
    if num_mask_features == 0:
        num_mask_features += 1

    selected_indices = []
    while len(selected_indices) < num_mask_features:
        randomF = random.choice(available_features)
        if randomF not in selected_indices:
            selected_indices.append(randomF)

    # Loop over the selected indices
    for mask_f in selected_indices:

        # Apply the masking to the selected chunk
        mask[:, :, mask_f] = True
        
    data[mask] = mask_value

    return data, mask
 
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

class Evaluation: # on chr21
    def __init__(
        self, model_path, hyper_parameters_path, 
        traindata_path, evaldata_path, version="16",
        resolution=25, chr_sizes_file="data/hg38.chrom.sizes", is_arcsin=True):

        self.traindata_path = traindata_path
        self.evaldata_path = evaldata_path
        self.is_arcsin = is_arcsin
        self.version = version

        with open(hyper_parameters_path, 'rb') as f:
            self.hyper_parameters = pickle.load(f)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        loader = MODEL_LOADER(model_path, self.hyper_parameters)

        self.model = loader.load_epidenoise(version=self.version)

        print(f"# model_parameters: {count_parameters(self.model)}")

        self.all_assays = ['M{:02d}'.format(i) for i in range(1, 36)]
        self.model.eval()  # set the model to evaluation mode

        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        self.chr_sizes = {}
        self.resolution = resolution

        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

        self.results = []

        self.train_data = {}
        self.eval_data = {}

        # load and bin chr21 of all bigwig files 
        for t in os.listdir(traindata_path):
            if ".bigwig" in t:

                for e in os.listdir(evaldata_path):
                    if ".bigwig" in e:
                        
                        if t[:3] == e[:3]:

                            if t[:3] not in self.train_data:
                                self.train_data[t[:3]] = {}

                            if e[:3] not in self.eval_data:
                                self.eval_data[e[:3]] = {}

                            self.train_data[t[:3]][t[3:6]] = traindata_path + "/" + t
                            self.eval_data[e[:3]][e[3:6]] = evaldata_path + "/" + e

        print(self.eval_data.keys())
        print(self.train_data.keys())

    def load_biosample(self, bios_name, mode="train"):
        chr, start, end = "chr21", 0, self.chr_sizes["chr21"]
        all_samples = []
        missing_ind = []

        if mode  == "train": 
            source = self.train_data
            savepath = self.traindata_path + f"/{bios_name}_chr21_{self.resolution}.pt"
        elif mode == "eval":
            source = self.eval_data
            savepath = self.evaldata_path + f"/{bios_name}_chr21_{self.resolution}.pt"
        
        if os.path.exists(savepath):
            all_samples = torch.load(savepath)
            # fill-in missing_ind
            for i in range(all_samples.shape[1]):
                if (all_samples[:, i] == -1).all():
                    missing_ind.append(i)
                    
            return all_samples, missing_ind

        else:
            for i in range(len(self.all_assays)):
                assay = self.all_assays[i]
                if assay in source[bios_name].keys():
                    print("loading ", assay)
                    bw = pyBigWig.open(source[bios_name][assay])
                    signals = bw.stats(chr, start, end, type="mean", nBins=(end - start) // self.resolution)
                
                else:
                    print(assay, "missing")
                    signals = [-1 for _ in range((end - start) // self.resolution)]
                    missing_ind.append(i)

            
                all_samples.append(signals)

            all_samples = torch.from_numpy(np.array(all_samples, dtype=np.float32)).transpose_(0, 1)

            # replace NaN with zero
            all_samples = torch.where(torch.isnan(all_samples), torch.zeros_like(all_samples), all_samples)

            nan_count = torch.isnan(all_samples).sum().item()
            minus_one_count = (all_samples == -1).sum().item()

            torch.save(all_samples, savepath)
            
            return all_samples, missing_ind
      
    def evaluate_biosample(self, bios_name):
        X, missing_x_i = self.load_biosample(bios_name, mode="train")
        Y, missing_y_i = self.load_biosample(bios_name, mode="eval")

        context_length, batch_size = self.hyper_parameters["context_length"], self.hyper_parameters["batch_size"]
        num_rows = (X.shape[0] // context_length) * context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]
        
        if self.is_arcsin:
            arcmask = (X != -1)
            X[arcmask] = torch.arcsinh_(X[arcmask])

        X = X.view(-1, context_length, X.shape[-1])
        Y = Y.view(-1, context_length, Y.shape[-1])

        d_model = X.shape[-1]

        if self.version == "10":
            fmask = torch.ones(d_model, self.hyper_parameters["d_model"])
            for i in missing_x_i: # input fmask
                fmask[i,:] = 0
            fmask = fmask.to(self.device)

        elif self.version == "16" or self.version == "17":
            CLS_x = torch.full((X.shape[0], 1, X.shape[2]), -2)
            SEP_x = torch.full((X.shape[0], 1, X.shape[2]), -3)
            CLS_y = torch.full((Y.shape[0], 1, Y.shape[2]), -2)
            SEP_y = torch.full((Y.shape[0], 1, Y.shape[2]), -3)

            X = torch.cat([CLS_x, X[:, :context_length//2, :], SEP_x, X[:, context_length//2:, :], SEP_x], dim=1)
            Y = torch.cat([CLS_y, Y[:, :context_length//2, :], SEP_y, Y[:, context_length//2:, :], SEP_y], dim=1)

            segment_label = [0] + [1 for i in range(context_length//2)] + [0] + [2 for i in range(context_length//2)] + [0]
            segment_label = torch.from_numpy(np.array(segment_label))
            segment_label = segment_label.to(self.device)

        # Initialize a tensor to store all predictions
        P = torch.empty_like(X, device="cpu")

        # make predictions in batches
        for i in range(0, len(X), batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+batch_size]

            with torch.no_grad():
                x_batch = x_batch.to(self.device)
                if self.version == "10":
                    # (no position is masked)
                    pmask = torch.zeros((x_batch.shape[0], x_batch.shape[1]), dtype=torch.bool,  device=self.device)
                    outputs = self.model(x_batch, pmask, fmask)

                elif self.version == "16":
                    outputs, pred_mask, SAP = self.model(x_batch, segment_label)

                elif self.version == "17":
                    mask = torch.zeros_like(x_batch, dtype=torch.bool)
                    for i in missing_x_i: 
                        mask[:,:,i] = True

                    outputs, SAP = self.model(x_batch, ~mask, segment_label)

            # Store the predictions in the large tensor
            P[i:i+outputs.shape[0], :, :] = outputs.cpu()
        
        
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1]) # preds
        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1]) # eval data
        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1]) # train data

        if self.is_arcsin:
            arcmask = (X != -1)
            P = torch.sinh_(P)
            X[arcmask] = torch.sinh_(X[arcmask])

        for j in range(Y.shape[-1]):  # for each feature i.e. assay
            pred = P[:, j].numpy()
            metrics_list = []

            if j in missing_x_i and j not in missing_y_i:  # if the feature is missing in the input
                target = Y[:, j].numpy()
                comparison = 'imputed'
            
            elif j not in missing_x_i:
                target = X[:, j].numpy()
                comparison = 'denoised'

            else:
                continue
            
            metrics = {
                'celltype': bios_name,
                'feature': self.all_assays[j],
                'comparison': comparison,
                'available train assays': len(self.all_assays) - len(missing_x_i),
                'available eval assays': len(self.all_assays) - len(missing_y_i),

                'MSE-GW': self.mse(target, pred),
                'Pearson-GW': self.pearson(target, pred),
                'Spearman-GW': self.spearman(target, pred),

                'MSE-1obs': self.mse1obs(target, pred),
                'Pearson_1obs': self.pearson1_obs(target, pred),
                'Spearman_1obs': self.spearman1_obs(target, pred),

                'MSE-1imp': self.mse1imp(target, pred),
                'Pearson_1imp': self.pearson1_imp(target, pred),
                'Spearman_1imp': self.spearman1_imp(target, pred),

                'MSE-gene': self.mse_gene(target, pred),
                'Pearson_gene': self.pearson_gene(target, pred),
                'Spearman_gene': self.spearman_gene(target, pred),

                'MSE-prom': self.mse_prom(target, pred),
                'Pearson_prom': self.pearson_prom(target, pred),
                'Spearman_prom': self.spearman_prom(target, pred),

                "peak_overlap_01thr": self.peak_overlap(target, pred, threshold=0.01),
                "peak_overlap_05thr": self.peak_overlap(target, pred, threshold=0.05),
                "peak_overlap_10thr": self.peak_overlap(target, pred, threshold=0.10)
            }
            self.results.append(metrics)
    
    def biosample_generate_imputations(self, bios_name, savedir="data/imputations/"):
        if os.path.exists(savedir) == False:
            os.mkdir(savedir)

        X, missing_x_i = self.load_biosample(bios_name, mode="train")
        Y, missing_y_i = self.load_biosample(bios_name, mode="eval")

        context_length, batch_size = self.hyper_parameters["context_length"], self.hyper_parameters["batch_size"]
        num_rows = (X.shape[0] // context_length) * context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        if self.is_arcsin:
            arcmask1 = (X != -1)
            X[arcmask1] = torch.arcsinh_(X[arcmask1])

            arcmask2 = (Y != -1)
            Y[arcmask2] = torch.arcsinh_(Y[arcmask2])

        X = X.view(-1, context_length, X.shape[-1])
        Y = Y.view(-1, context_length, Y.shape[-1])

        d_model = X.shape[-1]

        # Initialize a tensor to store all predictions
        P = torch.empty_like(X, device="cpu")

        # make predictions in batches
        for i in range(0, len(X), batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+batch_size]

            with torch.no_grad():
                x_batch = x_batch.to(self.device)
                mask = torch.zeros_like(x_batch, dtype=torch.bool, device=self.device)
                for ii in missing_x_i: 
                    mask[:,:,ii] = True
                mask = mask.to(self.device)

                if self.version == "10":
                    # (no position is masked)
                    pmask = torch.zeros((x_batch.shape[0], x_batch.shape[1]), dtype=torch.bool,  device=self.device)
                    outputs = self.model(x_batch, pmask, fmask)

                elif self.version == "16":
                    outputs, pred_mask, SAP = self.model(x_batch, segment_label)

                elif self.version == "17":
                    outputs, SAP = self.model(x_batch, ~mask, segment_label)
                
                elif self.version == "18":
                    outputs, pred_mask = self.model(x_batch)

                elif self.version == "20":
                    outputs, pred_mask = self.model(x_batch, mask)
                
                elif self.version == "21":
                    outputs, pred_mask = self.model(x_batch, mask)

            # Store the predictions in the large tensor
            P[i:i+outputs.shape[0], :, :] = outputs.cpu()
        
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1]) # preds
        torch.save(P, savedir+ bios_name + "_imp.pt")

    def evaluate_model(self, outdir):
        for bios in self.eval_data.keys():
            print("evaluating ", bios)
            self.evaluate_biosample(bios)

        self.results = pd.DataFrame(self.results)
        self.results.to_csv(outdir, index=False)

    ################################################################################

    def get_gene_positions(self, chrom, bin_size):
        gene_df = pd.read_csv(PROC_GENE_BED_FPATH, sep='\t', header=None,
                              names=['chrom', 'start', 'end', 'gene_id', 'gene_name'])
        chrom_subset = gene_df[gene_df['chrom'] == chrom].copy()

        chrom_subset['start'] = (chrom_subset['start'] / bin_size).apply(lambda s: math.floor(s))
        chrom_subset['end'] = (chrom_subset['end'] / bin_size).apply(lambda s: math.floor(s))
        return chrom_subset

    def get_prom_positions(self, chrom, bin_size):
        prom_df = pd.read_csv(PROC_PROM_BED_PATH, sep='\t', header=None,
                              names=['chrom', 'start', 'end', 'gene_id', 'gene_name', "strand"])
        chrom_subset = prom_df[prom_df['chrom'] == chrom].copy()

        chrom_subset['start'] = (chrom_subset['start'] / bin_size).apply(lambda s: math.floor(s))
        chrom_subset['end'] = (chrom_subset['end'] / bin_size).apply(lambda s: math.floor(s))

        return chrom_subset

    def get_signals(self, array, df):
        signals = []
        for idx, row in df.iterrows():
            gene_bins = slice(row['start'], row['end'])
            signals += array[gene_bins].tolist()

        return signals

    ################################################################################

    def mse(self, y_true, y_pred):
        """
        Calculate the genome-wide Mean Squared Error (MSE). This is a measure of the average squared difference 
        between the true and predicted values across the entire genome at a resolution of 25bp.
        """
        return np.mean((np.array(y_true) - np.array(y_pred))**2)

    def mse_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=gene_df)
        pred_vals = self.get_signals(array=y_pred, df=gene_df)

        return self.mse(y_true=gt_vals, y_pred=pred_vals)

    def pearson_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=gene_df)
        pred_vals = self.get_signals(array=y_pred, df=gene_df)

        return self.pearson(y_true=gt_vals, y_pred=pred_vals)

    def spearman_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=gene_df)
        pred_vals = self.get_signals(array=y_pred, df=gene_df)

        return self.spearman(y_true=gt_vals, y_pred=pred_vals)

    def pearson(self, y_true, y_pred):
        """
        Calculate the genome-wide Pearson Correlation. This measures the linear relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        return pearsonr(y_pred, y_true)[0]

    def spearman(self, y_true, y_pred):
        """
        Calculate the genome-wide Spearman Correlation. This measures the monotonic relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        return spearmanr(y_pred, y_true)[0]

    def mse_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=prom_df)
        pred_vals = self.get_signals(array=y_pred, df=prom_df)

        return self.mse(y_true=gt_vals, y_pred=pred_vals)

    def pearson_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=prom_df)
        pred_vals = self.get_signals(array=y_pred, df=prom_df)

        return self.pearson(y_true=gt_vals, y_pred=pred_vals)

    def spearman_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=prom_df)
        pred_vals = self.get_signals(array=y_pred, df=prom_df)

        return self.spearman(y_true=gt_vals, y_pred=pred_vals)

    def mse1obs(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error at the top 1% of genomic positions ranked by experimental signal (mse1obs). 
        This is a measure of how well predictions match observations at positions with high experimental signal. 
        It's similar to recall.
        """
        top_1_percent = int(0.01 * len(y_true))
        top_1_percent_indices = np.argsort(y_true)[-top_1_percent:]
        return mean_squared_error(y_true[top_1_percent_indices], y_pred[top_1_percent_indices])

    def mse1imp(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error at the top 1% of genomic positions ranked by predicted signal (mse1imp). 
        This is a measure of how well predictions match observations at positions with high predicted signal. 
        It's similar to precision.
        """
        top_1_percent = int(0.01 * len(y_pred))
        top_1_percent_indices = np.argsort(y_pred)[-top_1_percent:]
        return mean_squared_error(y_true[top_1_percent_indices], y_pred[top_1_percent_indices])

    def pearson1_obs(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return self.pearson(y_true[perc_99_pos], y_pred[perc_99_pos])

    def spearman1_obs(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return self.spearman(y_true[perc_99_pos], y_pred[perc_99_pos])

    def pearson1_imp(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return self.pearson(y_true[perc_99_pos], y_pred[perc_99_pos])

    def spearman1_imp(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return self.spearman(y_true[perc_99_pos], y_pred[perc_99_pos])

    def peak_overlap(self, y_true, y_pred, p=0.01):
        top_p_percent = int(p * len(y_true))
        
        # Get the indices of the top p percent of the observed (true) values
        top_p_percent_obs_i = np.argsort(y_true)[-top_p_percent:]
        
        # Get the indices of the top p percent of the predicted values
        top_p_percent_pred_i = np.argsort(y_pred)[-top_p_percent:]

        # Calculate the overlap
        overlap = len(np.intersect1d(top_p_percent_obs_i, top_p_percent_pred_i))

        # Calculate the percentage of overlap
        self.overlap_percent = overlap / top_p_percent 

        return self.overlap_percent

def check_poisson_vs_nbinom(data, assay_name):
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize

    # Fit Negative Binomial Distribution
    def negbinom_loglik(params, data):
        r, p = params
        return -np.sum(stats.nbinom.logpmf(data, r, p))

    initial_params = [1, 0.5]  # Initial guess for r and p
    result_nbinom = minimize(negbinom_loglik, initial_params, args=(data), bounds=[(1e-5, None), (1e-5, 1-1e-5)])
    r, p = result_nbinom.x

    # Fit Poisson Distribution
    lambda_poisson = np.mean(data)

    # Calculate Log-Likelihoods
    log_likelihood_nbinom = -negbinom_loglik([r, p], data)
    log_likelihood_poisson = np.sum(stats.poisson.logpmf(data, lambda_poisson))

    # Calculate AIC and BIC
    def aic_bic(log_likelihood, num_params, num_samples):
        aic = 2 * num_params - 2 * log_likelihood
        bic = num_params * np.log(num_samples) - 2 * log_likelihood
        return aic, bic

    aic_nbinom, bic_nbinom = aic_bic(log_likelihood_nbinom, 2, len(data))
    aic_poisson, bic_poisson = aic_bic(log_likelihood_poisson, 1, len(data))

    print(f"Negative Binomial - AIC: {aic_nbinom}, BIC: {bic_nbinom}")
    print(f"Poisson - AIC: {aic_poisson}, BIC: {bic_poisson}")

    # Plot the fit
    x = np.arange(0, max(data)+1)
    plt.hist(data, bins=x-0.5, density=True, alpha=0.6, color='g', label='Data')

    plt.plot(x, stats.nbinom.pmf(x, r, p), 'o-', label=f'Negative Binomial (r={r:.2f}, p={p:.2f})')
    plt.plot(x, stats.poisson.pmf(x, lambda_poisson), 'o-', label=f'Poisson ($\lambda$={lambda_poisson:.2f})')

    plt.legend()
    plt.xlabel('Data')
    plt.ylabel('Frequency')
    plt.title('Fit Comparison')
    plt.savefig(f"models/evals/examples/{assay_name}", dpi=150)

class EpiDenoise30b_old(nn.Module):
    def __init__(self, 
        input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, n_decoder_layers,
        dropout=0.1, context_length=2000, pos_enc="relative"):
        super(EpiDenoise30b_old, self).__init__()
        self.pos_enc = "abs"#pos_enc

        conv_out_channels = exponential_linspace_int(
            d_model//n_cnn_layers, d_model, n_cnn_layers, divisible_by=2)

        stride = 1
        dilation=1
        self.context_length = context_length
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.metadata_embedder = MetadataEmbeddingModule(input_dim, embedding_dim=metadata_embedding_dim, non_linearity=True)
        self.lin = nn.Linear(input_dim + metadata_embedding_dim, d_model)

        self.signal_layer_norm = nn.LayerNorm(input_dim)
        self.embedd_layer_norm = nn.LayerNorm(d_model)

        self.conv0 = ConvTower(
                input_dim + metadata_embedding_dim, conv_out_channels[0],
                conv_kernel_size[0], stride, dilation, 
                pool_type="max", residuals=True)

        self.convtower = nn.ModuleList([ConvTower(
                conv_out_channels[i], conv_out_channels[i + 1],
                conv_kernel_size[i + 1], stride, dilation, 
                pool_type="max", residuals=True
            ) for i in range(n_cnn_layers - 1)])

        if self.pos_enc == "relative":
            self.encoder_layer = RelativeEncoderLayer(
                d_model=d_model, heads=nhead, feed_forward_hidden=4*d_model, dropout=dropout)

            self.decoder_layer = RelativeDecoderLayer(
                hid_dim=d_model, n_heads=nhead, pf_dim=4*d_model, dropout=dropout)
        else:
            self.position = PositionalEncoding(d_model, dropout, context_length)
            # self.enc_position = AbsPositionalEmbedding15(d_model=d_model, max_len=self.context_length//(2**n_cnn_layers))
            # self.dec_position = AbsPositionalEmbedding15(d_model=d_model, max_len=self.context_length)

            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.ModuleList(
            [self.encoder_layer for _ in range(nlayers)])

        self.transformer_decoder = nn.ModuleList(
            [self.decoder_layer for _ in range(n_decoder_layers)])
        
        self.neg_binom_layer = NegativeBinomialLayer(d_model, output_dim)
        self.mask_pred_layer = nn.Linear(d_model, output_dim)
        self.mask_obs_layer = nn.Linear(d_model, output_dim)
    
    def forward(self, src, x_metadata, y_metadata, availability):
        md_embedding = self.metadata_embedder(x_metadata, y_metadata, availability)
        md_embedding = md_embedding.unsqueeze(1).expand(-1, self.context_length, -1)

        md_embedding = F.relu(md_embedding)
        src = self.signal_layer_norm(src)

        src = F.relu(torch.cat([src, md_embedding], dim=-1)) # N, L, F

        ### CONV ENCODER ###

        e_src = src.permute(0, 2, 1) # to N, F, L
        e_src = self.conv0(e_src)
        for conv in self.convtower:
            e_src = conv(e_src)
        e_src = e_src.permute(0, 2, 1)  # to N, L, F

        ### TRANSFORMER ENCODER ###
        if self.pos_enc != "relative":
            # encpos = torch.permute(self.enc_position(src), (1, 0, 2)) # to N, L, F
            e_src = self.position(e_src)
            # e_src = e_src + encpos

        for enc in self.transformer_encoder:
            e_src = enc(e_src)
        
        src = F.relu(self.embedd_layer_norm(self.lin(src)))

        ### TRANSFORMER DECODER ###
        if self.pos_enc != "relative":
            # trgpos = torch.permute(self.dec_position(src), (1, 0, 2))
            src = self.position(src)
            # src = src + trgpos

        for dec in self.transformer_decoder:
            src = dec(src, e_src)

        p, n = self.neg_binom_layer(src)
        mp = torch.sigmoid(self.mask_pred_layer(src))
        mo = torch.sigmoid(self.mask_obs_layer(src))

        return p, n, mp, mo

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

def fit_negative_binomial(data):
    # Estimate the parameters of the negative binomial distribution
    def negative_binomial_log_likelihood(params):
        n, p = params
        return -np.sum(stats.nbinom.logpmf(data, n, p))
    # Initial guess for n and p
    mean = np.mean(data)
    var = np.var(data)
    p_initial = mean / var
    n_initial = mean * p_initial / (1 - p_initial)
    # Bounds and constraints for parameters
    bounds = [(1e-5, None), (1e-5, 1-1e-5)]  # n > 0, 0 < p < 1
    # Minimize the negative log-likelihood
    result = minimize(negative_binomial_log_likelihood, [n_initial, p_initial], bounds=bounds, method='L-BFGS-B')
    n, p = result.x
    return n, p

def load_and_fit_nb(file_path):
    # Load the array from the npz file
    data = np.load(file_path)['arr_0']
    # Ensure data is integer
    data = np.round(data).astype(int)
    # Fit negative binomial distribution
    n, p = fit_negative_binomial(data)
    return n, p

class DeconvBlock(nn.Module):
    def __init__(self, in_C, out_C, W, S, D):
        super(DeconvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm1d(in_C)
        padding = W // 2
        output_padding = 1 
        self.deconv = nn.ConvTranspose1d(
            in_C, out_C, kernel_size=W, dilation=D, stride=S, 
            padding=padding, output_padding=output_padding)
        
    def forward(self, x):
        x = self.batch_norm(x)
        x = self.deconv(x)
        x = F.gelu(x)
        return x

class MONITOR_VALIDATION(object):
    def __init__(
        self, data_path, context_length, batch_size,
        chr_sizes_file="data/hg38.chrom.sizes", 
        resolution=25, split="val", arch="a", 
        token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}, eic=False):

        self.data_path = data_path
        self.context_length = context_length
        self.batch_size = batch_size
        self.resolution = resolution
        self.arch = arch
        self.eic = eic

        self.dataset = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        self.dataset.init_eval(self.context_length, check_completeness=True, split=split, bios_min_exp_avail_threshold=10, eic=eic)

        self.mark_dict = {v: k for k, v in self.dataset.aliases["experiment_aliases"].items()}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.example_coords = [
            (33481539//self.resolution, 33588914//self.resolution), # GART
            (25800151//self.resolution, 26235914//self.resolution), # APP
            (31589009//self.resolution, 31745788//self.resolution), # SOD1
            (39526359//self.resolution, 39802081//self.resolution), # B3GALT5
            (33577551//self.resolution, 33919338//self.resolution), # ITSN1
            (36260000//self.resolution, 36450000//self.resolution), # RUNX1
            (45000000//self.resolution, 45250000//self.resolution), # COL18A1
            (36600000//self.resolution, 36850000//self.resolution), # MX1
            (39500000//self.resolution, 40000000//self.resolution) # Highly Conserved Non-Coding Sequences (HCNS)
            ]

        self.token_dict = token_dict
        
        if self.arch in ["c"]:
            self.token_dict["cloze_mask"] = self.token_dict["missing_mask"]

        self.chr_sizes = {}
        self.metrics = METRICS()
        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)
    
    def pred(self, X, mX, mY, avail, imp_target=[]):
        # print("making preds")
        # Initialize a tensor to store all predictions
        n = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        # make predictions in batches
        for i in range(0, len(X), self.batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+ self.batch_size]
            mX_batch = mX[i:i+ self.batch_size]
            mY_batch = mY[i:i+ self.batch_size]
            avail_batch = avail[i:i+ self.batch_size]

            with torch.no_grad():
                x_batch = x_batch.clone()
                avail_batch = avail_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()

                x_batch_missing_vals = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing_vals = (mX_batch == self.token_dict["missing_mask"])
                # mY_batch_missing_vals = (mY_batch == self.token_dict["missing_mask"])
                avail_batch_missing_vals = (avail_batch == 0)

                x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing_vals] = self.token_dict["cloze_mask"]
                # mY_batch[mY_batch_missing_vals] = self.token_dict["cloze_mask"]
                if self.arch in ["a", "b"]:
                    avail_batch[avail_batch_missing_vals] = self.token_dict["cloze_mask"]

                if len(imp_target)>0:
                    x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    # mY_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    if self.arch in ["a", "b"]:
                        avail_batch[:, imp_target] = self.token_dict["cloze_mask"]
                    elif self.arch in ["c", "d"]:
                        avail_batch[:, imp_target] = 0

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                if self.arch in ["a", "b", "d"]:
                    outputs_p, outputs_n, _, _ = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)
                elif self.arch in ["c"]:
                    outputs_p, outputs_n = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)

            # Store the predictions in the large tensor
            n[i:i+outputs_n.shape[0], :, :] = outputs_n.cpu()
            p[i:i+outputs_p.shape[0], :, :] = outputs_p.cpu()

            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n  # Free up memory
            torch.cuda.empty_cache()  # Free up GPU memory

        return n, p

    def get_bios(self, bios_name, x_dsf=1, y_dsf=1):
        print(f"getting bios vals for {bios_name}")
        temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx
        
        temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        Y, mY, avY= self.dataset.make_bios_tensor(temp_y, temp_my)
        del temp_y, temp_my

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        subsets_X = []
        subsets_Y = []

        for start, end in self.example_coords:
            segment_length = end - start
            adjusted_length = (segment_length // self.context_length) * self.context_length
            adjusted_end = start + adjusted_length

            subsets_X.append(X[start:adjusted_end, :])
            subsets_Y.append(Y[start:adjusted_end, :])

        # Concatenate the subsets along the sequence length dimension (second dimension)
        X = torch.cat(subsets_X, dim=0)
        Y = torch.cat(subsets_Y, dim=0)

        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        available_indices = torch.where(avX[0, :] == 1)[0]

        n_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        for leave_one_out in available_indices:
            n, p = self.pred(X, mX, mY, avX, imp_target=[leave_one_out])
            
            n_imp[:, :, leave_one_out] = n[:, :, leave_one_out]
            p_imp[:, :, leave_one_out] = p[:, :, leave_one_out]
            # print(f"got imputations for feature #{leave_one_out+1}")
            del n, p  # Free up memory
        
        n_ups, p_ups = self.pred(X, mX, mY, avX, imp_target=[])
        del X, mX, mY, avX, avY  # Free up memoryrm m
        # print("got upsampled")

        p_imp = p_imp.view((p_imp.shape[0] * p_imp.shape[1]), p_imp.shape[-1])
        n_imp = n_imp.view((n_imp.shape[0] * n_imp.shape[1]), n_imp.shape[-1])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        imp_dist = NegativeBinomial(p_imp, n_imp)
        ups_dist = NegativeBinomial(p_ups, n_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])

        return imp_dist, ups_dist, Y, bios_name, available_indices
    
    def get_bios_eic(self, bios_name, x_dsf=1, y_dsf=1):
        print(f"getting bios vals for {bios_name}")

        temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("V_", "T_"), ["chr21", 0, self.chr_sizes["chr21"]], x_dsf, eic=True)
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx
        
        temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        Y, mY, avY= self.dataset.make_bios_tensor(temp_y, temp_my)
        # mY = self.dataset.fill_in_y_prompt(mY)
        del temp_y, temp_my

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        subsets_X = []
        subsets_Y = []

        for start, end in self.example_coords:
            segment_length = end - start
            adjusted_length = (segment_length // self.context_length) * self.context_length
            adjusted_end = start + adjusted_length

            subsets_X.append(X[start:adjusted_end, :])
            subsets_Y.append(Y[start:adjusted_end, :])

        # Concatenate the subsets along the sequence length dimension (second dimension)
        X = torch.cat(subsets_X, dim=0)
        Y = torch.cat(subsets_Y, dim=0)

        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        available_X_indices = torch.where(avX[0, :] == 1)[0]
        available_Y_indices = torch.where(avY[0, :] == 1)[0]
        
        n_ups, p_ups = self.pred(X, mX, mY, avX, imp_target=[])
        # del X, mX, mY, avX, avY  # Free up memoryrm m

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        ups_dist = NegativeBinomial(p_ups, n_ups)

        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1])
        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])

        return ups_dist, Y, X, bios_name, available_X_indices, available_Y_indices

    def get_frame(self, bios_name, x_dsf=1, y_dsf=1):
        temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx

        temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        Y, mY, avY= self.dataset.make_bios_tensor(temp_y, temp_my)
        del temp_y, temp_my

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        subsets_X = []
        subsets_Y = []

        start, end = 33481539//self.resolution, 33588914//self.resolution
        segment_length = end - start
        adjusted_length = (segment_length // self.context_length) * self.context_length
        adjusted_end = start + adjusted_length

        subsets_X.append(X[start:adjusted_end, :])
        subsets_Y.append(Y[start:adjusted_end, :])

        # Concatenate the subsets along the sequence length dimension (second dimension)
        X = torch.cat(subsets_X, dim=0)
        Y = torch.cat(subsets_Y, dim=0)

        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        available_indices = torch.where(avX[0, :] == 1)[0]

        n_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        for leave_one_out in available_indices:
            n, p = self.pred(X, mX, mY, avX, imp_target=[leave_one_out])
            
            n_imp[:, :, leave_one_out] = n[:, :, leave_one_out]
            p_imp[:, :, leave_one_out] = p[:, :, leave_one_out]
            # print(f"got imputations for feature #{leave_one_out+1}")
            del n, p  # Free up memory
        
        n_ups, p_ups = self.pred(X, mX, mY, avX, imp_target=[])
        del X, mX, mY, avX, avY  # Free up memoryrm m
        # print("got upsampled")

        p_imp = p_imp.view((p_imp.shape[0] * p_imp.shape[1]), p_imp.shape[-1])
        n_imp = n_imp.view((n_imp.shape[0] * n_imp.shape[1]), n_imp.shape[-1])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        imp_dist = NegativeBinomial(p_imp, n_imp)
        ups_dist = NegativeBinomial(p_ups, n_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])

        return imp_dist, ups_dist, Y, bios_name, available_indices

    def get_metric(self, imp_dist, ups_dist, Y, bios_name, availability):
        # print(f"getting metrics")
        imp_mean = imp_dist.expect()
        ups_mean = ups_dist.expect()

        # print(f"got nbinom stuff")
        # imp_lower_95, imp_upper_95 = imp_dist.interval(confidence=0.95)
        # ups_lower_95, ups_upper_95 = ups_dist.interval(confidence=0.95)
        
        results = []
        # for j in availability:  # for each feature i.e. assay
        for j in range(Y.shape[1]):

            if j in list(availability):
                # j = j.item()
                for comparison in ['imputed', 'upsampled']:
                    if comparison == "imputed":
                        pred = imp_mean[:, j].numpy()
                        # lower_95 = imp_lower_95[:, j].numpy()
                        # upper_95 = imp_upper_95[:, j].numpy()
                        
                    elif comparison == "upsampled":
                        pred = ups_mean[:, j].numpy()
                        # lower_95 = ups_lower_95[:, j].numpy()
                        # upper_95 = ups_upper_95[:, j].numpy()

                    target = Y[:, j].numpy()

                    # Check if the target values fall within the intervals
                    # within_interval = (target >= lower_95) & (target <= upper_95)
                    
                    # Calculate the fraction
                    # print(
                    #     f"adding {bios_name} | {self.mark_dict[f'M{str(j+1).zfill(len(str(len(self.mark_dict))))}']} | {comparison}")
                    metrics = {
                        'bios':bios_name,
                        'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                        'comparison': comparison,
                        'available assays': len(availability),

                        'MSE': self.metrics.mse(target, pred),
                        'Pearson': self.metrics.pearson(target, pred),
                        'Spearman': self.metrics.spearman(target, pred),
                        'r2': self.metrics.r2(target, pred)
                    }
                    results.append(metrics)

        return results
    
    def get_metric_eic(self, ups_dist, Y, X, bios_name, availability_X, availability_Y):
        ups_mean = ups_dist.expect()
        
        results = []
        for j in range(Y.shape[1]):
            pred = ups_mean[:, j].numpy()
            if j in list(availability_X):
                comparison = "upsampled"
                target = X[:, j].numpy()

                metrics = {
                    'bios':bios_name,
                    'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                    'comparison': comparison,
                    'available assays': len(availability_X),

                    'MSE': self.metrics.mse(target, pred),
                    'Pearson': self.metrics.pearson(target, pred),
                    'Spearman': self.metrics.spearman(target, pred),
                    'r2': self.metrics.r2(target, pred)
                }
                results.append(metrics)
                
            elif j in list(availability_Y):
                comparison = "imputed"
                target = Y[:, j].numpy()

                metrics = {
                    'bios':bios_name,
                    'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                    'comparison': comparison,
                    'available assays': len(availability_X),

                    'MSE': self.metrics.mse(target, pred),
                    'Pearson': self.metrics.pearson(target, pred),
                    'Spearman': self.metrics.spearman(target, pred),
                    'r2': self.metrics.r2(target, pred)
                }
                results.append(metrics)

        return results

    def get_validation(self, model, x_dsf=1, y_dsf=1):
        t0 = datetime.datetime.now()
        self.model = model

        full_res = []
        bioses = list(self.dataset.navigation.keys())
        if not self.eic:
            for bios_name in bioses:
                try:
                    imp_dist, ups_dist, Y, _, available_indices = self.get_bios(bios_name, x_dsf=x_dsf, y_dsf=y_dsf)
                    full_res += self.get_metric(imp_dist, ups_dist, Y, bios_name, available_indices)
                    del imp_dist, ups_dist, Y
                except:
                    pass
        else:
            for bios_name in bioses:
                # try:
                ups_dist, Y, X, bios_name, available_X_indices, available_Y_indices = self.get_bios_eic(bios_name, x_dsf=x_dsf, y_dsf=y_dsf)
                full_res += self.get_metric_eic(ups_dist, Y, X, bios_name, available_X_indices, available_Y_indices)
                del ups_dist, Y, X
                # except:
                #     pass

        del self.model
        df = pd.DataFrame(full_res)

        # Separate the data based on comparison type
        imputed_df = df[df['comparison'] == 'imputed']
        upsampled_df = df[df['comparison'] == 'upsampled']

        # Function to calculate mean, min, and max for a given metric
        def calculate_stats(df, metric):
            return df[metric].mean(), df[metric].min(), df[metric].max()

        # Imputed statistics
        imp_mse_stats = calculate_stats(imputed_df, 'MSE')
        imp_pearson_stats = calculate_stats(imputed_df, 'Pearson')
        imp_spearman_stats = calculate_stats(imputed_df, 'Spearman')
        imp_r2_stats = calculate_stats(imputed_df, 'r2')
        # imp_frac95conf_stats = calculate_stats(imputed_df, 'frac_95_confidence')

        # Upsampled statistics
        ups_mse_stats = calculate_stats(upsampled_df, 'MSE')
        ups_pearson_stats = calculate_stats(upsampled_df, 'Pearson')
        ups_spearman_stats = calculate_stats(upsampled_df, 'Spearman')
        ups_r2_stats = calculate_stats(upsampled_df, 'r2')
        # ups_frac95conf_stats = calculate_stats(upsampled_df, 'frac_95_confidence')

        elapsed_time = datetime.datetime.now() - t0
        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        # Create the compact print statement
        print_statement = f"""
        Took {int(minutes)}:{int(seconds)}
        For Imputed:
        - MSE: mean={imp_mse_stats[0]:.2f}, min={imp_mse_stats[1]:.2f}, max={imp_mse_stats[2]:.2f}
        - PCC: mean={imp_pearson_stats[0]:.2f}, min={imp_pearson_stats[1]:.2f}, max={imp_pearson_stats[2]:.2f}
        - SRCC: mean={imp_spearman_stats[0]:.2f}, min={imp_spearman_stats[1]:.2f}, max={imp_spearman_stats[2]:.2f}
        - R2: mean={imp_r2_stats[0]:.2f}, min={imp_r2_stats[1]:.2f}, max={imp_r2_stats[2]:.2f}

        For Upsampled:
        - MSE: mean={ups_mse_stats[0]:.2f}, min={ups_mse_stats[1]:.2f}, max={ups_mse_stats[2]:.2f}
        - PCC: mean={ups_pearson_stats[0]:.2f}, min={ups_pearson_stats[1]:.2f}, max={ups_pearson_stats[2]:.2f}
        - SRCC: mean={ups_spearman_stats[0]:.2f}, min={ups_spearman_stats[1]:.2f}, max={ups_spearman_stats[2]:.2f}
        - R2: mean={ups_r2_stats[0]:.2f}, min={ups_r2_stats[1]:.2f}, max={ups_r2_stats[2]:.2f}
        """

        return print_statement

    def generate_training_gif_frame(self, model, fig_title):
        def gen_subplt(
            ax, x_values, observed_values, 
            ups11, ups21, ups41, 
            imp11, imp21, imp41, 
            col, assname, ytick_fontsize=6, title_fontsize=6):

            # Define the data and labels
            data = [
                (observed_values, "Observed", "royalblue", f"{assname}_Observed"),
                (ups11, "Upsampled 1->1", "darkcyan", f"{assname}_Ups1->1"),
                (imp11, "Imputed 1->1", "salmon", f"{assname}_Imp1->1"),
                (ups21, "Upsampled 2->1", "darkcyan", f"{assname}_Ups2->1"),
                (imp21, "Imputed 2->1", "salmon", f"{assname}_Imp2->1"),
                (ups41, "Upsampled 4->1", "darkcyan", f"{assname}_Ups4->1"),
                (imp41, "Imputed 4->1", "salmon", f"{assname}_Imp4->1"),
            ]
            
            for i, (values, label, color, title) in enumerate(data):
                ax[i, col].plot(x_values, values, "--" if i != 0 else "-", color=color, alpha=0.7, label=label, linewidth=0.01)
                ax[i, col].fill_between(x_values, 0, values, color=color, alpha=0.7)
                
                if i != len(data)-1:
                    ax[i, col].tick_params(axis='x', labelbottom=False)
                
                ax[i, col].tick_params(axis='y', labelsize=ytick_fontsize)
                ax[i, col].set_xticklabels([])
                ax[i, col].set_title(title, fontsize=title_fontsize)

        self.model = model

        bios = list(self.dataset.navigation.keys())[0]
        # print(bios)

        # dsf4-1
        imp_dist, ups_dist, Y, _, available_indices = self.get_frame(bios, x_dsf=4, y_dsf=1)
        imp_mean41, ups_mean41 = imp_dist.expect(), ups_dist.expect()

        # dsf2-1
        imp_dist, ups_dist, Y, _, available_indices = self.get_frame(bios, x_dsf=2, y_dsf=1)
        imp_mean21, ups_mean21 = imp_dist.expect(), ups_dist.expect()

        # dsf1-1
        imp_dist, ups_dist, Y, _, available_indices = self.get_frame(bios, x_dsf=1, y_dsf=1)
        imp_mean11, ups_mean11 = imp_dist.expect(), ups_dist.expect()

        del self.model

        selected_assays = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K9me3", "CTCF", "DNase-seq", "ATAC-seq"]
        available_selected = []
        for col, jj in enumerate(available_indices):
            assay = self.mark_dict[f"M{str(jj.item()+1).zfill(len(str(len(self.mark_dict))))}"]
            if assay in selected_assays:
                available_selected.append(jj)

        fig, axes = plt.subplots(7, len(available_selected), figsize=(len(available_selected) * 3, 6), sharex=True, sharey=False)
        
        for col, jj in enumerate(available_selected):
            j = jj.item()
            assay = self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"]
            x_values = list(range(len(Y[:, j])))

            obs = Y[:, j].numpy()

            gen_subplt(axes, x_values, 
                    obs, 
                    ups_mean11[:, j].numpy(), ups_mean21[:, j].numpy(), ups_mean41[:, j].numpy(), 
                    imp_mean11[:, j].numpy(), imp_mean21[:, j].numpy(), imp_mean41[:, j].numpy(), 
                    col, assay)

        fig.suptitle(fig_title, fontsize=10)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        
        return buf


    # e = EVAL_EED(
    #     model="/project/compbio-lab/EPD/pretrained/EPD30d_model_checkpoint_Jul8th.pth", 
    #     data_path="/project/compbio-lab/encode_data/", 
    #     context_length=3200, batch_size=50, 
    #     hyper_parameters_path="/project/compbio-lab/EPD/pretrained/hyper_parameters30d_EpiDenoise30d_20240710133714_params237654660.pkl",
    #     train_log={}, chr_sizes_file="data/hg38.chrom.sizes", 
    #     version="30d", resolution=25, savedir="/project/compbio-lab/EPD/eval_30d/", mode="eval")
    
    # print(e.bios_pipeline("ENCBS343AKO", x_dsf=1))

class SyntheticData:
    def __init__(self, n, p, num_features, sequence_length):
        self.n = n
        self.p = p
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.transformations = [
            (self.transform_scale, {'scale': 2}),
            (self.transform_exponential, {'base': 1.03, 'scale': 3}),
            (self.transform_log_scale, {'scale': 40}),
            (self.transform_sqrt_scale, {'scale': 15}),
            (self.transform_piecewise_linear, {'scale_factors': [1.5, 50, 0.5, 100, 0.3]}),
            (self.transform_scaled_sin, {'scale': 30}),
            (self.transform_scaled_cos, {'scale': 30}),
            (self.transform_hyperbolic_sinh, {'scale': 10}),
            (self.transform_polynomial, {'scale': 0.02, 'power': 2}),
            (self.transform_exponential, {'base': 1.05, 'scale': 2}),
            (self.transform_log_scale, {'scale': 60}),
            (self.transform_sqrt_scale, {'scale': 25}),
            (self.transform_piecewise_linear, {'scale_factors': [2, 60, 0.4, 110, 0.2]}),
            (self.transform_scaled_sin, {'scale': 60}),
            (self.transform_scaled_cos, {'scale': 60}),
            (self.transform_hyperbolic_sinh, {'scale': 15}),
            (self.transform_polynomial, {'scale': 0.005, 'power': 3}),
            (self.transform_exponential, {'base': 1.05, 'scale': 1}),
            (self.transform_log_scale, {'scale': 20}),
            (self.transform_sqrt_scale, {'scale': 15}),
            (self.transform_piecewise_linear, {'scale_factors': [2.5, 40, 0.3, 120, 0.25]}),
            (self.transform_scaled_sin, {'scale': 20}),
            (self.transform_scaled_cos, {'scale': 20}),
            (self.transform_hyperbolic_sinh, {'scale': 20}),
            (self.transform_polynomial, {'scale': 0.01, 'power': 3})
        ]

    def generate_base_sequence(self):
        self.base_sequence = np.random.negative_binomial(self.n, self.p, self.sequence_length)
        return self.base_sequence

    def transform_scale(self, sequence, scale):
        return np.clip(sequence * scale, 0, 200)

    def transform_exponential(self, sequence, base, scale):
        return np.clip(np.round(scale * (base ** (sequence))), 0, 1000)

    def transform_log_scale(self, sequence, scale):
        return np.clip(np.round(scale * np.log1p(sequence)), 0, 1000)

    def transform_sqrt_scale(self, sequence, scale):
        return np.clip(np.round(scale * np.sqrt(sequence)), 0, 1000)

    def transform_piecewise_linear(self, sequence, scale_factors):
        transformed = np.zeros_like(sequence)
        for i, value in enumerate(sequence):
            if value < 50:
                transformed[i] = value * scale_factors[0]
            elif value < 150:
                transformed[i] = scale_factors[1] + scale_factors[2] * value
            else:
                transformed[i] = scale_factors[3] + scale_factors[4] * value
        return np.clip(np.round(transformed), 0, 1000)

    def transform_scaled_sin(self, sequence, scale):
        return np.clip(np.round(scale * np.abs(np.sin(sequence))), 0, 1000)

    def transform_scaled_cos(self, sequence, scale):
        return np.clip(np.round(scale * np.abs(np.cos(sequence))), 0, 1000)

    def transform_hyperbolic_sinh(self, sequence, scale):
        return np.clip(np.round(scale * np.sinh(sequence / 50)), 0, 1000)

    def transform_polynomial(self, sequence, scale, power):
        return np.clip(np.round(scale * (sequence ** power)), 0, 1000)

    def apply_transformations(self):
        transformed_sequences = []
        for i in range(self.num_features):
            transform, params = self.transformations[i % len(self.transformations)]
            transformed_seq = transform(self.base_sequence, **params)
            transformed_sequences.append((transformed_seq, transform.__name__))

        return transformed_sequences

    def smooth_sequence(self, sequence, sigma=0.001):
        return gaussian_filter1d(sequence, sigma=sigma)

    def apply_smoothing(self, sequences):
        return [self.smooth_sequence(seq).astype(int) for seq, name in sequences]
    
    def synth_metadata(self, sequences):
        def depth(seq):
            return np.log2((3e9*np.sum(seq))/len(seq))
        def coverage(seq):
            return 100 * np.abs(np.sin(np.sum(seq)))
        def read_length(seq):
            return np.log10(np.sum(seq)+1)
        def run_type(seq):
            if np.mean(seq) <= np.median(seq):
                return 1
            else:
                return 0

        return [np.array([depth(seq), coverage(seq), read_length(seq), run_type(seq)]) for seq, name in sequences]

    def miss(self, sequences, metadata, missing_percentage):
        to_miss = random.choices(range(self.num_features), k=int(self.num_features*missing_percentage))
        avail = [1 for i in range(self.num_features)]

        for miss in to_miss:
            sequences[miss, :] = -1
            metadata[miss, :] = -1
            avail[miss] = 0
        
        return sequences, metadata, avail

    def mask(self, sequences, metadata, avail, mask_percentage):
        to_mask = random.choices([x for x in range(self.num_features) if avail[x]==1], k=int(self.num_features*mask_percentage))

        for mask in to_mask:
            sequences[mask, :] = -2
            metadata[mask, :] = -2
            avail[mask] = -2

        return sequences, metadata, avail

    def get_batch(self, batch_size, miss_perc_range=(0.3, 0.9), mask_perc_range=(0.1, 0.2)):
        batch_X, batch_Y = [], []
        md_batch_X, md_batch_Y = [], []
        av_batch_X, av_batch_Y = [], []
        
        for b in range(batch_size):
            self.generate_base_sequence()
            transformed_sequences = self.apply_transformations()

            smoothed_sequences = self.apply_smoothing(transformed_sequences)
            smoothed_sequences = np.array(smoothed_sequences)

            syn_metadata = self.synth_metadata(transformed_sequences)
            syn_metadata = np.array(syn_metadata)

            miss_p_b = random.uniform(miss_perc_range[0], miss_perc_range[1])
            mask_p_b = random.uniform(mask_perc_range[0], mask_perc_range[1])
            
            y_b, ymd_b, yav_b = self.miss(smoothed_sequences, syn_metadata, miss_p_b)
            x_b, xmd_b, xav_b = self.mask(y_b.copy(), ymd_b.copy(), yav_b.copy(), mask_p_b)

            batch_X.append(x_b)
            batch_Y.append(y_b)

            md_batch_X.append(xmd_b)
            md_batch_Y.append(ymd_b)

            av_batch_X.append(xav_b)
            av_batch_Y.append(yav_b)
        
        batch_X, batch_Y = torch.Tensor(np.array(batch_X)).permute(0, 2, 1), torch.Tensor(np.array(batch_Y)).permute(0, 2, 1)
        md_batch_X, md_batch_Y = torch.Tensor(np.array(md_batch_X)).permute(0, 2, 1), torch.Tensor(np.array(md_batch_Y)).permute(0, 2, 1)
        av_batch_X, av_batch_Y = torch.Tensor(np.array(av_batch_X)), torch.Tensor(np.array(av_batch_Y))

        return batch_X, batch_Y, md_batch_X, md_batch_Y, av_batch_X, av_batch_Y
    

    def __new_epoch(self):
        self.current_bios_batch_pointer = 0
        self.current_loci_batch_pointer = 0
    
    def __update_batch_pointers(self, cycle_biosamples_first=True):
        if cycle_biosamples_first:
            # Cycle through all biosamples for each loci before moving to the next loci
            if self.current_bios_batch_pointer + self.bios_batchsize >= self.num_bios:
                self.current_bios_batch_pointer = 0
                if self.current_loci_batch_pointer + self.loci_batchsize < self.num_regions:
                    self.current_loci_batch_pointer += self.loci_batchsize
                else:
                    self.current_loci_batch_pointer = 0  # Reset loci pointer after the last one
                    return True
            else:
                self.current_bios_batch_pointer += self.bios_batchsize
        else:
            # Cycle through all loci for each batch of biosamples before moving to the next batch of biosamples
            if self.current_loci_batch_pointer + self.loci_batchsize >= self.num_regions:
                self.current_loci_batch_pointer = 0
                if self.current_bios_batch_pointer + self.bios_batchsize < self.num_bios:
                    self.current_bios_batch_pointer += self.bios_batchsize
                else:
                    self.current_bios_batch_pointer = 0  # Reset biosample pointer after the last one
                    return True
            else:
                self.current_loci_batch_pointer += self.loci_batchsize

        return False

    def __get_batch(self, dsf):
        batch_loci_list = self.m_regions[self.current_loci_batch_pointer : self.current_loci_batch_pointer+self.loci_batchsize]
        batch_bios_list = list(self.navigation.keys())[self.current_bios_batch_pointer : self.current_bios_batch_pointer+self.bios_batchsize]
        
        batch_data = []
        batch_metadata = []
        batch_availability = []

        for locus in batch_loci_list:
            self.make_region_tensor
            d, md, avl = self.__make_region_tensor(batch_bios_list, locus, DSF=dsf)
            batch_data.append(d)
            batch_metadata.append(md)
            batch_availability.append(avl)
        
        batch_data, batch_metadata, batch_availability = torch.concat(batch_data), torch.concat(batch_metadata), torch.concat(batch_availability)
        return batch_data, batch_metadata, batch_availability


    
    def __make_region_tensor(self, list_bios, locus, DSF, max_workers=-1):
        """Load and process data for multiple biosamples in parallel."""
        def load_and_process(bios):
            try:
                loaded_data, loaded_metadata = self.load_bios(bios, locus, DSF)
                return self.make_bios_tensor(loaded_data, loaded_metadata)
            except Exception as e:
                print(f"Failed to process {bios}: {e}")
                return None

        if max_workers == -1:
            max_workers = self.bios_batchsize//2

        # Use ThreadPoolExecutor to handle biosamples in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(load_and_process, list_bios))

        # Aggregate results
        data, metadata, availability = [], [], []
        for result in results:
            if result is not None:
                d, md, avl = result
                data.append(d)
                metadata.append(md)
                availability.append(avl)

        data, metadata, availability = torch.stack(data), torch.stack(metadata), torch.stack(availability)
        return data, metadata, availability

class EVAL_EED(object):
    """
    for imputating missing tracks, we should replace mY with 'prompt' metadata.
    prompt = [24, ~max_assay_genome_coverage, ~max_assay_read_length, pair-end]
    """
    def __init__(
        self, model, data_path, context_length, batch_size, hyper_parameters_path="",
        train_log={}, chr_sizes_file="data/hg38.chrom.sizes", version="30a", resolution=25, 
        savedir="models/evals/", mode="eval", split="test"):

        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.mkdir(self.savedir)

        self.data_path = data_path
        self.version = version
        self.context_length = context_length
        self.batch_size = batch_size
        self.resolution = resolution

        self.model = model
        self.dataset = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        self.dataset.init_eval(self.context_length, check_completeness=True, split=split, bios_min_exp_avail_threshold=5)

        self.mark_dict = {v: k for k, v in self.dataset.aliases["experiment_aliases"].items()}

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.token_dict = {
                    "missing_mask": -1, 
                    "cloze_mask": -2,
                    "pad": -3
                }

        self.chr_sizes = {}
        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

        self.train_data = {}
        self.eval_data = {}
        self.metrics = METRICS()
        self.viz = VISUALS(resolution=self.resolution, savedir=self.savedir)

        self.gene_coords = load_gene_coords("data/parsed_genecode_data_hg38_release42.csv")
        self.gene_coords = self.gene_coords[self.gene_coords["chr"] == "chr21"].reset_index(drop=True)

        if mode == "dev":
            return

        if type(self.model) == str:
            with open(hyper_parameters_path, 'rb') as f:
                self.hyper_parameters = pickle.load(f)
            loader = MODEL_LOADER(model, self.hyper_parameters)
            self.model = loader.load_epidenoise(version=self.version)

        self.model = self.model.to(self.device)
        self.model.eval()  # set the model to evaluation mode
        print(f"# model_parameters: {count_parameters(self.model)}")

    def eval_rnaseq(self, bios_name, y_pred, y_true, availability, k_fold=10, plot_REC=True):
        # columns=  chr, start, end, geneID, length, TPM, FPKM
        rna_seq_data = self.dataset.load_rna_seq_data(bios_name, self.gene_coords) 
        print(rna_seq_data)
        
        pred_features = []
        true_features = []
        available_assays = [self.mark_dict[f"M{str(a+1).zfill(len(str(len(self.mark_dict))))}"] for a in range(y_pred.shape[1]) if a in list(availability)]
        print(available_assays)
        
        for i in range(len(rna_seq_data)):
            for a in range(y_pred.shape[1]):
                assay_name = self.mark_dict[f"M{str(a+1).zfill(len(str(len(self.mark_dict))))}"]

                if a in list(availability):
                    true_signal_a = y_true[:, a].numpy()
                    f = signal_feature_extraction(
                        rna_seq_data["start"][i], rna_seq_data["end"][i], 
                        rna_seq_data["strand"][i], true_signal_a
                        )

                    f = [assay_name, rna_seq_data["geneID"][i], f["mean_sig_promoter"], f["mean_sig_gene_body"], 
                        f["mean_sig_around_TES"], rna_seq_data["TPM"][i], rna_seq_data["FPKM"][i]]

                    true_features.append(f)
                
                pred_signal_a = y_pred[:, a].numpy()
                f = signal_feature_extraction(
                        rna_seq_data["start"][i], rna_seq_data["end"][i], 
                        rna_seq_data["strand"][i], pred_signal_a
                        )
                    
                f = [assay_name, rna_seq_data["geneID"][i], f["mean_sig_promoter"], f["mean_sig_gene_body"], 
                    f["mean_sig_around_TES"], rna_seq_data["TPM"][i], rna_seq_data["FPKM"][i]]

                pred_features.append(f)
        
        true_features = pd.DataFrame(true_features, columns=["assay", "geneID", "promoter_signal", "gene_body_signal", "TES_signal", "TPM", "FPKM"])
        pred_features_all = pd.DataFrame(pred_features, columns=["assay", "geneID", "promoter_signal", "gene_body_signal", "TES_signal", "TPM", "FPKM"])
        pred_features_avail = pred_features_all[pred_features_all["assay"].isin(available_assays)]

        report = {}
        # Perform K-Fold Cross Validation for both true and predicted data
        # print("Evaluating Experimental Data")
        report['true_linear'] = k_fold_cross_validation(true_features, k=k_fold, target='TPM', logscale=True, model_type='linear')
        
        # print("Evaluating Denoised + Imputed Data")
        report['denoised_imputed_linear'] = k_fold_cross_validation(pred_features_all, k=k_fold, target='TPM', logscale=True, model_type='linear')

        # print("Evaluating Denoised Data")
        report['denoised_linear'] = k_fold_cross_validation(pred_features_avail, k=k_fold, target='TPM', logscale=True, model_type='linear')

        # Perform K-Fold Cross Validation for both true and predicted data
        # print("Evaluating Experimental Data")
        report['true_svr'] = k_fold_cross_validation(true_features, k=k_fold, target='TPM', logscale=True, model_type='svr')
        
        # print("Evaluating Denoised + Imputed Data")
        report['denoised_imputed_svr'] = k_fold_cross_validation(pred_features_all, k=k_fold, target='TPM', logscale=True, model_type='svr')

        # print("Evaluating Denoised Data")
        report['denoised_svr'] = k_fold_cross_validation(pred_features_avail, k=k_fold, target='TPM', logscale=True, model_type='svr')
        
        # Plotting REC curves for comparison
        if plot_REC:
            plt.figure(figsize=(14, 7))
            
            # Plot REC for SVR models
            plt.subplot(1, 2, 1)
            true_errors_svr = report['true_svr']['errors']
            denoised_errors_svr = report['denoised_svr']['errors']
            denoised_imputed_errors_svr = report['denoised_imputed_svr']['errors']
            
            sorted_true_errors_svr = np.sort(true_errors_svr)
            cumulative_true_svr = np.arange(1, len(sorted_true_errors_svr) + 1) / len(sorted_true_errors_svr)
            
            sorted_denoised_errors_svr = np.sort(denoised_errors_svr)
            cumulative_denoised_svr = np.arange(1, len(sorted_denoised_errors_svr) + 1) / len(sorted_denoised_errors_svr)

            sorted_denoised_imputed_errors_svr = np.sort(denoised_imputed_errors_svr)
            cumulative_denoised_imputed_svr = np.arange(1, len(sorted_denoised_imputed_errors_svr) + 1) / len(sorted_denoised_imputed_errors_svr)
            
            plt.plot(sorted_true_errors_svr, cumulative_true_svr, label='Observed', color='blue', alpha=0.7)
            plt.plot(sorted_denoised_errors_svr, cumulative_denoised_svr, label='Denoised', color='orange', alpha=0.7)
            plt.plot(sorted_denoised_imputed_errors_svr, cumulative_denoised_imputed_svr, label='Denoised+Imputed', color='green', alpha=0.7)
            plt.xlabel('Error Tolerance')
            plt.ylabel('Proportion of Points within Tolerance')
            plt.title('REC Curve - SVR')
            plt.legend()
            plt.grid(True)
            
            # Plot REC for Linear models
            plt.subplot(1, 2, 2)
            true_errors_linear = report['true_linear']['errors']
            denoised_errors_linear = report['denoised_linear']['errors']
            denoised_imputed_errors_linear = report['denoised_imputed_linear']['errors']
            
            sorted_true_errors_linear = np.sort(true_errors_linear)
            cumulative_true_linear = np.arange(1, len(sorted_true_errors_linear) + 1) / len(sorted_true_errors_linear)
            
            sorted_denoised_errors_linear = np.sort(denoised_errors_linear)
            cumulative_denoised_linear = np.arange(1, len(sorted_denoised_errors_linear) + 1) / len(sorted_denoised_errors_linear)

            sorted_denoised_imputed_errors_linear = np.sort(denoised_imputed_errors_linear)
            cumulative_denoised_imputed_linear = np.arange(1, len(sorted_denoised_imputed_errors_linear) + 1) / len(sorted_denoised_imputed_errors_linear)
            
            plt.plot(sorted_true_errors_linear, cumulative_true_linear, label='Observed', color='blue', alpha=0.7)
            plt.plot(sorted_denoised_errors_linear, cumulative_denoised_linear, label='Denoised', color='orange', alpha=0.7)
            plt.plot(sorted_denoised_imputed_errors_linear, cumulative_denoised_imputed_linear, label='Denoised+Imputed', color='green', alpha=0.7)
            plt.xlabel('Error Tolerance')
            plt.ylabel('Proportion of Points within Tolerance')
            plt.title('REC Curve - Linear Regression')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            savepath = os.path.join(self.savedir, bios_name+f"_{len(available_assays)}")
            if os.path.exists(savepath) ==False:
                os.mkdir(savepath)

            plt.savefig(savepath+"/RNAseq_REC.svg", format="svg")

        return report
            
    def get_metrics(self, imp_dist, ups_dist, Y, bios_name, availability):
        """
        reportoir of metrics -- per_bios:

            peak_ovr: 01thr, 05thr, 10thr

            GeWi: MSE, Pearson, Spearman
            1imp: MSE, Pearson, Spearman
            1obs: MSE, Pearson, Spearman
            gene: MSE, Pearson, Spearman
            prom: MSE, Pearson, Spearman
        """

        imp_mean = imp_dist.expect()
        ups_mean = ups_dist.expect()

        imp_std = imp_dist.std()
        ups_std = ups_dist.std()

        if self.dataset.has_rnaseq(bios_name):
            print("got rna-seq data")
            rnaseq_res = self.eval_rnaseq(bios_name, ups_mean, Y, availability, k_fold=10, plot_REC=True)

        # imp_lower_60, imp_upper_60 = imp_dist.interval(confidence=0.6)
        # ups_lower_60, ups_upper_60 = ups_dist.interval(confidence=0.6)

        # imp_lower_80, imp_upper_80 = imp_dist.interval(confidence=0.8)
        # ups_lower_80, ups_upper_80 = ups_dist.interval(confidence=0.8)
        print("getting 0.95 interval conf")

        imp_lower_95, imp_upper_95 = imp_dist.interval(confidence=0.95)
        ups_lower_95, ups_upper_95 = ups_dist.interval(confidence=0.95)

        results = []
        # for j in availability:  # for each feature i.e. assay
        for j in range(Y.shape[1]):

            if j in list(availability):
                target = Y[:, j].numpy()

                for comparison in ['imputed', 'upsampled']:
                    
                    if comparison == "imputed":
                        pred = imp_mean[:, j].numpy()
                        pred_std = imp_std[:, j].numpy()
                        # lower_60 = imp_lower_60[:, j].numpy()
                        # lower_80 = imp_lower_80[:, j].numpy()
                        lower_95 = imp_lower_95[:, j].numpy()

                        # upper_60 = imp_upper_60[:, j].numpy()
                        # upper_80 = imp_upper_80[:, j].numpy()
                        upper_95 = imp_upper_95[:, j].numpy()

                        quantile = self.metrics.confidence_quantile(imp_dist.p[:,j], imp_dist.n[:,j], target)
                        p0bgdf = self.metrics.foreground_vs_background(imp_dist.p[:,j], imp_dist.n[:,j], target)
                        
                    elif comparison == "upsampled":
                        pred = ups_mean[:, j].numpy()
                        pred_std = ups_std[:, j].numpy()
                        # lower_60 = ups_lower_60[:, j].numpy()
                        # lower_80 = ups_lower_80[:, j].numpy()
                        lower_95 = ups_lower_95[:, j].numpy()

                        # upper_60 = ups_upper_60[:, j].numpy()
                        # upper_80 = ups_upper_80[:, j].numpy()
                        upper_95 = ups_upper_95[:, j].numpy()

                        quantile = self.metrics.confidence_quantile(ups_dist.p[:,j], ups_dist.n[:,j], target)
                        p0bgdf = self.metrics.foreground_vs_background(ups_dist.p[:,j], ups_dist.n[:,j], target)


                    # corresp, corresp_deriv = self.metrics.correspondence_curve(target, pred)
                    metrics = {
                        'bios':bios_name,
                        'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                        'comparison': comparison,
                        'available assays': len(availability),

                        "obs":target,
                        "imp":pred,
                        "pred_quantile":quantile,
                        "pred_std":pred_std,

                        # "lower_60" : lower_60,
                        # "lower_80" : lower_80,
                        "lower_95" : lower_95,

                        # "upper_60": upper_60,
                        # "upper_80": upper_80,
                        "upper_95": upper_95,

                        "p0_bg":p0bgdf["p0_bg"],
                        "p0_fg":p0bgdf["p0_fg"],

                        'MSE-GW': self.metrics.mse(target, pred),
                        'Pearson-GW': self.metrics.pearson(target, pred),
                        'Spearman-GW': self.metrics.spearman(target, pred),
                        'r2_GW': self.metrics.r2(target, pred),

                        'MSE-1obs': self.metrics.mse1obs(target, pred),
                        'Pearson_1obs': self.metrics.pearson1_obs(target, pred),
                        'Spearman_1obs': self.metrics.spearman1_obs(target, pred),
                        'r2_1obs': self.metrics.r2_1obs(target, pred),

                        'MSE-1imp': self.metrics.mse1imp(target, pred),
                        'Pearson_1imp': self.metrics.pearson1_imp(target, pred),
                        'Spearman_1imp': self.metrics.spearman1_imp(target, pred),
                        'r2_1imp': self.metrics.r2_1imp(target, pred),

                        'MSE-gene': self.metrics.mse_gene(target, pred),
                        'Pearson_gene': self.metrics.pearson_gene(target, pred),
                        'Spearman_gene': self.metrics.spearman_gene(target, pred),
                        'r2_gene': self.metrics.r2_gene(target, pred),

                        'MSE-prom': self.metrics.mse_prom(target, pred),
                        'Pearson_prom': self.metrics.pearson_prom(target, pred),
                        'Spearman_prom': self.metrics.spearman_prom(target, pred),
                        'r2_prom': self.metrics.r2_prom(target, pred),

                        "peak_overlap_01thr": self.metrics.peak_overlap(target, pred, p=0.01),
                        "peak_overlap_05thr": self.metrics.peak_overlap(target, pred, p=0.05),
                        "peak_overlap_10thr": self.metrics.peak_overlap(target, pred, p=0.10),

                    #     "corresp_curve": corresp,
                    #     "corresp_curve_deriv": corresp_deriv
                    }
                    
                    if self.dataset.has_rnaseq(bios_name):
                        metrics["rnaseq-true-pcc-linear"] = rnaseq_res["true_linear"]["avg_pcc"]
                        metrics["rnaseq-true-pcc-svr"] = rnaseq_res["true_svr"]["avg_pcc"]

                        metrics["rnaseq-denoised-pcc-linear"] = rnaseq_res["denoised_linear"]["avg_pcc"]
                        metrics["rnaseq-denoised-pcc-svr"] = rnaseq_res["denoised_svr"]["avg_pcc"]

                        metrics["rnaseq-true-mse-linear"] = rnaseq_res["true_linear"]["avg_mse"]
                        metrics["rnaseq-true-mse-svr"] = rnaseq_res["true_svr"]["avg_mse"]
                        
                        metrics["rnaseq-denoised-mse-linear"] = rnaseq_res["denoised_linear"]["avg_mse"]
                        metrics["rnaseq-denoised-mse-svr"] = rnaseq_res["denoised_svr"]["avg_mse"]

                    results.append(metrics)

            else:
                # continue
                pred = ups_mean[:, j].numpy()
                # lower_60 = ups_lower_60[:, j].numpy()
                # lower_80 = ups_lower_80[:, j].numpy()
                lower_95 = ups_lower_95[:, j].numpy()

                # upper_60 = ups_upper_60[:, j].numpy()
                # upper_80 = ups_upper_80[:, j].numpy()
                upper_95 = ups_upper_95[:, j].numpy()

                metrics = {
                    'bios':bios_name,
                    'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                    'comparison': "None",
                    'available assays': len(availability),

                    "imp":pred,

                    # "lower_60" : lower_60,
                    # "lower_80" : lower_80,
                    "lower_95" : lower_95,

                    # "upper_60": upper_60,
                    # "upper_80": upper_80,
                    "upper_95": upper_95
                    }
                results.append(metrics)
            
        return results
    
    def load_bios(self, bios_name, x_dsf, y_dsf=1):
        """
        Load biosample data for a specified biosample at given downsampling factors for X and Y.

        Parameters:
        bios_name (str): The name of the biosample.
        x_dsf (int): Downsampling factor for the X dataset.
        y_dsf (int): Downsampling factor for the Y dataset, defaults to 1.

        Returns:
        tuple: A tuple containing the tensors for X, mX, avX, Y, mY, and avY.
        """
        temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], x_dsf)
        # temp_x, temp_mx = self.dataset.load_bios(bios_name, ["chr21", self.chr_sizes["chr21"]//4, self.chr_sizes["chr21"]//2], x_dsf)
        X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
        del temp_x, temp_mx

        temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", 0, self.chr_sizes["chr21"]], y_dsf)
        # temp_y, temp_my = self.dataset.load_bios(bios_name, ["chr21", self.chr_sizes["chr21"]//4, self.chr_sizes["chr21"]//2], y_dsf)
        Y, mY, avY= self.dataset.make_bios_tensor(temp_y, temp_my)
        del temp_y, temp_my

        num_rows = (X.shape[0] // self.context_length) * self.context_length

        X, Y = X[:num_rows, :], Y[:num_rows, :]


        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])

        
        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        return X, mX, avX, Y, mY, avY

    def pred(self, X, mX, mY, avail, imp_target=[]):
        # Initialize a tensor to store all predictions
        n = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        # make predictions in batches
        for i in range(0, len(X), self.batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+ self.batch_size]
            mX_batch = mX[i:i+ self.batch_size]
            mY_batch = mY[i:i+ self.batch_size]
            avail_batch = avail[i:i+ self.batch_size]

            with torch.no_grad():
                x_batch = x_batch.clone()
                avail_batch = avail_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()

                x_batch_missing_vals = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing_vals = (mX_batch == self.token_dict["missing_mask"])
                # mY_batch_missing_vals = (mY_batch == self.token_dict["missing_mask"])
                avail_batch_missing_vals = (avail_batch == 0)

                x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing_vals] = self.token_dict["cloze_mask"]
                # mY_batch[mY_batch_missing_vals] = self.token_dict["cloze_mask"]
                if self.version in ["a", "b"]:
                    avail_batch[avail_batch_missing_vals] = self.token_dict["cloze_mask"]

                if len(imp_target)>0:
                    x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    # mY_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    if self.version in ["a", "b"]:
                        avail_batch[:, imp_target] = self.token_dict["cloze_mask"]
                    elif self.version in ["c", "d"]:
                        avail_batch[:, imp_target] = 0

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                if self.version in ["30a", "30b", "30d"]:
                    outputs_p, outputs_n, _, _ = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)
                elif self.version in ["30c"]:
                    outputs_p, outputs_n = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)

            # Store the predictions in the large tensor
            n[i:i+outputs_n.shape[0], :, :] = outputs_n.cpu()
            p[i:i+outputs_p.shape[0], :, :] = outputs_p.cpu()

        return n, p

    def bios_pipeline(self, bios_name, x_dsf):
        X, mX, avX, Y, mY, avY = self.load_bios(bios_name, x_dsf)  

        available_indices = torch.where(avX[0, :] == 1)[0]

        n_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p_imp = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        for leave_one_out in available_indices:
            n, p = self.pred(X, mX, mY, avX, imp_target=[leave_one_out])
            
            n_imp[:, :, leave_one_out] = n[:, :, leave_one_out]
            p_imp[:, :, leave_one_out] = p[:, :, leave_one_out]
            print(f"got imputations for feature #{leave_one_out+1}")
        
        n_ups, p_ups = self.pred(X, mX, mY, avX, imp_target=[])
        print("got upsampled")

        p_imp = p_imp.view((p_imp.shape[0] * p_imp.shape[1]), p_imp.shape[-1])
        n_imp = n_imp.view((n_imp.shape[0] * n_imp.shape[1]), n_imp.shape[-1])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])

        imp_dist = NegativeBinomial(p_imp, n_imp)
        ups_dist = NegativeBinomial(p_ups, n_ups)

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1]) 

        eval_res = self.get_metrics(imp_dist, ups_dist, Y, bios_name, available_indices)
        return eval_res

    def viz_bios(self, eval_res):
        print("plotting signal tracks")
        try:
            self.viz.BIOS_signal_track(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot signal tracks: {e}")

        print("plotting signal confidence")
        try:
            self.viz.BIOS_signal_confidence(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot signal confidence: {e}")

        # Filter out results without 'obs'
        eval_res = [res for res in eval_res if "obs" in res]

        print("plotting mean vs. std hexbin")
        try:
            self.viz.BIOS_mean_std_hexbin(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot mean vs. std hexbin: {e}")

        # print("plotting quantile heatmap")
        # try:
        #     self.viz.BIOS_quantile_heatmap(eval_res)
        #     self.viz.clear_pallete()
        # except Exception as e:
        #     print(f"Failed to plot quantile heatmap: {e}")

        print("plotting error vs. std hexbin")
        try:
            self.viz.BIOS_error_std_hexbin(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot error vs. std hexbin: {e}")

        print("plotting quantile histogram")
        try:
            self.viz.BIOS_quantile_hist(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot quantile histogram: {e}")

        print("plotting context-specific performance")
        try:
            self.viz.BIOS_context_length_specific_performance(eval_res, self.context_length, bins=10)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot context-specific performance: {e}")

        print("plotting signal scatter with marginals")
        try:
            self.viz.BIOS_signal_scatter_with_marginals(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot signal scatter with marginals: {e}")

        print("plotting signal heatmap")
        try:
            self.viz.BIOS_signal_heatmap(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot signal heatmap: {e}")

        print("plotting signal rank heatmap")
        try:
            self.viz.BIOS_signal_rank_heatmap(eval_res)
            self.viz.clear_pallete()
        except Exception as e:
            print(f"Failed to plot signal rank heatmap: {e}")

        # Uncomment the following blocks if you want to include these plots as well:
        # print("plotting mean vs. std scatter")
        # try:
        #     self.viz.BIOS_mean_std_scatter(eval_res)
        #     self.viz.clear_pallete()
        # except Exception as e:
        #     print(f"Failed to plot mean vs. std scatter: {e}")
        # print("plotting correspondence curve")
        # try:
        #     self.viz.BIOS_corresp_curve(eval_res)
        #     self.viz.clear_pallete()
        # except Exception as e:
        #     print(f"Failed to plot correspondence curve: {e}")

        # print("plotting correspondence curve derivative")
        # try:
        #     self.viz.BIOS_corresp_curve_deriv(eval_res)
        #     self.viz.clear_pallete()
        # except Exception as e:
        #     print(f"Failed to plot correspondence curve derivative: {e}")

    def viz_all(self, dsf=1):
        """
        visualizations -- all_bios:
        
            denoised vs imputed
                boxplots for metric per assay
                    peak_ovr: 01thr, 05thr, 10thr
                    GeWi: MSE, Pearson, Spearman
                    1imp: MSE, Pearson, Spearman
                    1obs: MSE, Pearson, Spearman
                    gene: MSE, Pearson, Spearman
                    prom: MSE, Pearson, Spearman
        """
        
        self.model_res = []
        print(f"Evaluating {len(list(self.dataset.navigation.keys()))} biosamples...")
        for bios in list(self.dataset.navigation.keys()):
            try:
                print("evaluating ", bios)
                eval_res_bios = self.bios_pipeline(bios, dsf)
                print("got results for ", bios)
                self.viz_bios(eval_res_bios)
                
                to_del = [
                    "obs", "imp", "pred_quantile", "pred_std", 
                    "lower_60", "lower_80", "lower_95", 
                    "upper_60", "upper_80", "upper_95"]

                for f in eval_res_bios:
                    fkeys = list(f.keys())
                    for d in to_del:
                        if d in fkeys:
                            del f[d]
                    
                    if f["comparison"] != "None":
                        self.model_res.append(f)
            except:
                pass

        self.model_res = pd.DataFrame(self.model_res)
        self.model_res.to_csv(f"{self.savedir}/model_eval_DSF{dsf}.csv", index=False)

        # boxplot_metrics = [
        #     'MSE-GW', 'Pearson-GW', 'Spearman-GW',
        #     'MSE-1obs', 'Pearson_1obs', 'Spearman_1obs',
        #     'MSE-1imp', 'Pearson_1imp', 'Spearman_1imp',
        #     'MSE-gene', 'Pearson_gene', 'Spearman_gene',
        #     'MSE-prom', 'Pearson_prom', 'Spearman_prom',
        #     'peak_overlap_01thr', 'peak_overlap_05thr', 
        #     'peak_overlap_10thr']
        
        # for m in boxplot_metrics:
        #     self.viz.MODEL_boxplot(self.model_res, metric=m)
        #     self.viz.MODEL_regplot_overall(self.model_res, metric=m)
        #     self.viz.MODEL_regplot_perassay(self.model_res, metric=m)

class VISUALS(object):
    def __init__(self, resolution=25, savedir="models/evals/"):
        self.metrics = METRICS()
        self.resolution = resolution
        self.savedir = savedir

    def clear_pallete(self):
        sns.reset_orig
        plt.close("all")
        plt.style.use('default')
        plt.clf()

    def BIOS_signal_track(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        example_gene_coord = (33481539//self.resolution, 33588914//self.resolution) # GART
        example_gene_coord2 = (25800151//self.resolution, 26235914//self.resolution) # APP
        example_gene_coord3 = (31589009//self.resolution, 31745788//self.resolution) # SOD1
        example_gene_coord4 = (39526359//self.resolution, 39802081//self.resolution) # B3GALT5
        example_gene_coord5 = (33577551//self.resolution, 33919338//self.resolution) # ITSN1

        # Create a list of example gene coordinates for iteration
        example_gene_coords = [
            example_gene_coord, example_gene_coord2, example_gene_coord3,
            example_gene_coord4, example_gene_coord5]

        # Define the size of the figure
        plt.figure(figsize=(6 * len(example_gene_coords), len(eval_res) * 2))

        # Loop over each result
        for j in range(len(eval_res)):
            # Loop over each gene
            for i, gene_coord in enumerate(example_gene_coords):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(example_gene_coords), j * len(example_gene_coords) + i + 1)
                
                # Calculate x_values based on the current gene's coordinates
                x_values = range(gene_coord[0], gene_coord[1])
                imputed_values = eval_res[j]["imp"][gene_coord[0]:gene_coord[1]]

                # Plot the lines
                if "obs" in eval_res[j].keys():
                    observed_values = eval_res[j]["obs"][gene_coord[0]:gene_coord[1]]
                    ax.plot(x_values, observed_values, color="blue", alpha=0.7, label="Observed", linewidth=0.1)
                    ax.fill_between(x_values, 0, observed_values, alpha=0.7, color="blue")

                ax.plot(x_values, imputed_values, "--", color="red", alpha=0.5, label="Imputed", linewidth=0.1)
                ax.fill_between(x_values, 0, imputed_values, color="red", alpha=0.5)

                start_coord = gene_coord[0] * self.resolution
                end_coord = gene_coord[1] * self.resolution
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_ylabel("Signal")

                ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
                ax.set_xticklabels([])

                custom_lines = [mlines.Line2D([], [], color='blue', label='Observed'),
                                mlines.Line2D([], [], color='red',  label='Imputed')]
                ax.legend(handles=custom_lines)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_tracks.png", dpi=300)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_tracks.svg", format="svg")

    def BIOS_signal_confidence(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # example_gene_coord =  (33481539//self.resolution, 33588914//self.resolution) # GART
        # example_gene_coord2 = (25800151//self.resolution, 26235914//self.resolution) # APP
        # example_gene_coord3 = (31589009//self.resolution, 31745788//self.resolution) # SOD1
        # example_gene_coord4 = (39526359//self.resolution, 39802081//self.resolution) # B3GALT5
        # example_gene_coord5 = (33577551//self.resolution, 33919338//self.resolution) # ITSN1

        # Create a list of example gene coordinates for iteration
        example_gene_coords = [
            (33481539//self.resolution, 33588914//self.resolution), # GART
            (25800151//self.resolution, 26235914//self.resolution), # APP
            (31589009//self.resolution, 31745788//self.resolution), # SOD1
            (39526359//self.resolution, 39802081//self.resolution), # B3GALT5
            (33577551//self.resolution, 33919338//self.resolution) # ITSN1
            ]
            # example_gene_coord, example_gene_coord2, example_gene_coord3,
            # example_gene_coord4, example_gene_coord5]

        # Define the size of the figure
        plt.figure(figsize=(8 * len(example_gene_coords), len(eval_res) * 2))
        # plt.subplots_adjust(hspace=0.4, wspace=0.3)

        for j, result in enumerate(eval_res):
            for i, gene_coord in enumerate(example_gene_coords):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(example_gene_coords), j * len(example_gene_coords) + i + 1)
                
                # Calculate x_values based on the current gene's coordinates
                x_values = range(gene_coord[0], gene_coord[1])

                # Fill between for confidence intervals
                ax.fill_between(
                    x_values, result['lower_95'][gene_coord[0]:gene_coord[1]], result['upper_95'][gene_coord[0]:gene_coord[1]], 
                    color='coral', alpha=0.4, label='95% Confidence')

                # ax.fill_between(
                #     x_values, result['lower_80'][gene_coord[0]:gene_coord[1]], result['upper_80'][gene_coord[0]:gene_coord[1]], 
                #     color='coral', alpha=0.2, label='80% Confidence')

                # ax.fill_between(
                #     x_values, result['lower_60'][gene_coord[0]:gene_coord[1]], result['upper_60'][gene_coord[0]:gene_coord[1]], 
                #     color='coral', alpha=0.4, label='60% Confidence')

                # Plot the median predictions
                ax.plot(x_values, result['imp'][gene_coord[0]:gene_coord[1]], label='Mean', color='red', linewidth=0.5)

                if "obs" in result.keys():
                    # Plot the actual observations
                    ax.plot(
                        x_values, result['obs'][gene_coord[0]:gene_coord[1]], 
                        label='Observed', color='royalblue', linewidth=0.4, alpha=0.8)


                start_coord = gene_coord[0] * self.resolution
                end_coord = gene_coord[1] * self.resolution

                # Set plot titles and labels
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_ylabel("Signal")
                ax.set_yscale('log') 
                ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
                ax.set_xticklabels([])

                # Only show legend in the first subplot to avoid redundancy
                if i == 0 and j ==0:
                    ax.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/confidence_intervals.pdf", dpi=300)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/confidence_intervals.svg", format="svg")

    def BIOS_quantile_scatter(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue
            
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["pred_quantile"]
                    pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"

                ax.scatter(xs, ys, color="black", s=5, alpha=0.7)
                # ax.grid(True, linestyle='-', color='gray', alpha=0.5)
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"Obs. vs. Pred. Quantile {eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{pcc}")
                ax.set_xlabel("Observed Values")
                ax.set_ylabel("Predicted Quantile")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_scatter.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_scatter.svg", format="svg")

    def BIOS_quantile_density(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue
            
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["pred_quantile"]
                    pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"

                sns.kdeplot(x=xs, y=ys, cmap="Blues", fill=True, ax=ax)
                # ax.scatter(xs, ys, color='red', alpha=0.3)
                # ax.grid(True, linestyle='-', color='gray', alpha=0.5)

                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"Obs. vs. Pred. Quantile w/ Density {eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{pcc}")
                ax.set_xlabel("Observed Values")
                ax.set_ylabel("Predicted Quantile")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_density_scatter.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_density_scatter.svg", format="svg")
    
    def BIOS_quantile_hist(self, eval_res, b=20):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue
            
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["pred_quantile"]

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])

                ax.hist(ys, bins=b, color='blue', alpha=0.7, density=True)
                # ax.grid(True, linestyle='-', color='gray', alpha=0.5)
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"Obs. vs. Pred. Quantile {eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}")
                ax.set_xlabel("Predicted CDF Quantile")
                ax.set_ylabel("Density")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_hist.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_hist.svg", format="svg")

    def BIOS_quantile_heatmap(self, eval_res, b=20):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue
            
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["pred_quantile"]
                    pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["pred_quantile"])
                    pcc = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"

                # Create the heatmap
                h, xedges, yedges = np.histogram2d(np.asarray(xs), np.asarray(ys), bins=b, density=True)
                h = h.T  # Transpose to correct the orientation
                h = h / h.sum(axis=0, keepdims=True)  # Normalize cols

                im = ax.imshow(
                    h, interpolation='nearest', origin='lower', 
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    aspect='auto', cmap='viridis', norm=LogNorm())
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{pcc}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted Quantiles")
                plt.colorbar(im, ax=ax, orientation='vertical')
                

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_heatmap.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/quantile_heatmap.svg", format="svg")

    def BIOS_mean_std_scatter(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        plt.figure(figsize=(5, len(eval_res) * 5))  # one column with len(eval_res) rows

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(len(eval_res), 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs"], eval_res[j]["imp"], eval_res[j]["pred_std"]
            pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

            sc = ax.scatter(observed, pred_mean, c=pred_std, cmap='viridis', alpha=0.6, s=5)
            plt.colorbar(sc, ax=ax, label='Predicted std')
            ax.plot([observed.min(), observed.max()], [observed.min(), observed.max()], 'k--')
            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted Mean')
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}_{pcc}")
            # plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/mean_std_scatter.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/mean_std_scatter.svg", format="svg")
    
    def BIOS_error_std_hexbin(self, eval_res):
        save_path = f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        num_plots = len(eval_res) * 3  # Each evaluation will have 3 subplots
        plt.figure(figsize=(15, len(eval_res) * 5))  # Adjust width for 3 columns

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                # skip rows without observed signal
                continue

            observed, pred_mean, pred_std = eval_res[j]["obs"], eval_res[j]["imp"], eval_res[j]["pred_std"]
            pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"
            error = np.abs(observed - pred_mean)

            # Calculate the percentiles for x-axis limits
            x_90 = np.percentile(error, 99)
            x_99 = np.percentile(error, 99.9)

            # Define the ranges for subsetting
            ranges = [(0, x_90), (x_90, x_99), (x_99, error.max())]

            for i, (x_min, x_max) in enumerate(ranges):
                # Subset the data for the current range
                mask = (error >= x_min) & (error <= x_max)
                subset_error = error[mask]
                subset_pred_std = pred_std[mask]
                
                ax = plt.subplot(len(eval_res), 3, j * 3 + i + 1)

                # Hexbin plot for the subset data
                hb = ax.hexbin(subset_error, subset_pred_std, gridsize=50, cmap='viridis', mincnt=1, norm=LogNorm())

                ax.set_xlabel('Absolute Error')
                ax.set_ylabel('Predicted Std Dev')
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}_{pcc} (Range: {x_min:.2f}-{x_max:.2f})")

                # Add color bar
                cb = plt.colorbar(hb, ax=ax)
                cb.set_label('Log10(Counts)')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/error_std_hexbin.png", dpi=150)
        plt.savefig(f"{save_path}/error_std_hexbin.svg", format="svg")
    
    def BIOS_mean_std_hexbin(self, eval_res):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        # Define the size of the figure
        plt.figure(figsize=(5, len(eval_res) * 5))  # one column with len(eval_res) rows

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                # skip rows without observed signal
                continue

            ax = plt.subplot(len(eval_res), 1, j + 1)  # One column with len(eval_res) rows

            observed, pred_mean, pred_std = eval_res[j]["obs"], eval_res[j]["imp"], eval_res[j]["pred_std"]
            pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

            hb = ax.hexbin(observed, pred_mean, C=pred_std, gridsize=30, cmap='viridis', reduce_C_function=np.mean)
            plt.colorbar(hb, ax=ax, label='Predicted std')
            ax.plot([observed.min(), observed.max()], [observed.min(), observed.max()], 'k--')
            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted Mean')
            ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}_{pcc}")
            # plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/mean_std_hexbin.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/mean_std_hexbin.svg", format="svg")
        
    def BIOS_signal_scatter(self, eval_res, share_axes=True):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue

            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["imp"]
                    pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    pcc = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    pcc = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"

                ax.scatter(xs, ys, color="black", s=5, alpha=0.7)
                
                if share_axes:
                    # Determine the range for x and y axes
                    common_min = min(min(xs), min(ys))
                    common_max = max(max(xs), max(ys))
                    
                    # Set the same range for x and y axes
                    ax.set_xlim(common_min, common_max)
                    ax.set_ylim(common_min, common_max)
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{pcc}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_scatters.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_scatters.svg", format="svg")

    def BIOS_signal_scatter_with_marginals(self, eval_res, share_axes=True, percentile_cutoff=99):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]
        num_rows = len(eval_res)
        num_cols = len(cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

        for j, result in enumerate(eval_res):
            if "obs" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = axes[j, i] if num_rows > 1 else axes[i]

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["imp"]
                    pcc = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    pcc = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    pcc = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    pcc = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    pcc = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"
                    
                sns.scatterplot(x=xs, y=ys, ax=ax, color="#4CB391", s=3, alpha=0.9)

                # Calculate percentile cutoffs for both axes
                x_upper = np.percentile(xs, percentile_cutoff)
                y_upper = np.percentile(ys, percentile_cutoff)
                
                # Use the same upper bound for both axes to maintain aspect ratio
                upper_bound = min(x_upper, y_upper)
                
                # Filter points within bounds
                mask = (xs <= upper_bound) & (ys <= upper_bound)
                xs_filtered = xs[mask]
                ys_filtered = ys[mask]

                # Update bin range for histograms using filtered data
                bin_range = np.linspace(0, upper_bound, 50)

                ax_histx = ax.inset_axes([0, 1.05, 1, 0.2])
                ax_histy = ax.inset_axes([1.05, 0, 0.2, 1])
                
                ax_histx.hist(xs_filtered, bins=bin_range, alpha=0.9, color="#f25a64")
                ax_histy.hist(ys_filtered, bins=bin_range, orientation='horizontal', alpha=0.9, color="#f25a64")
                
                ax_histx.set_xticklabels([])
                ax_histx.set_yticklabels([])
                ax_histy.set_xticklabels([])
                ax_histy.set_yticklabels([])

                # Set title, labels, and range if share_axes is True
                ax.set_title(f"{result['feature']}_{c}_{result['comparison']}_{pcc}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

                if share_axes:
                    # Set axis limits
                    ax.set_xlim(0, upper_bound)
                    ax.set_ylim(0, upper_bound)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_scatters_with_marginals.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_scatters_with_marginals.svg", format="svg")

    def BIOS_signal_heatmap(self, eval_res, share_axes=True, bins=50):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["imp"]
                    title_suffix = f"PCC_GW: {eval_res[j]['Pearson-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    title_suffix = f"PCC_Gene: {eval_res[j]['Pearson_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    title_suffix = f"PCC_TSS: {eval_res[j]['Pearson_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    title_suffix = f"PCC_1obs: {eval_res[j]['Pearson_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    title_suffix = f"PCC_1imp: {eval_res[j]['Pearson_1imp']:.2f}"

                # Create the heatmap
                h, xedges, yedges = np.histogram2d(xs, ys, bins=bins, density=True)
                h = h.T  # Transpose to correct the orientation
                ax.imshow(h, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', norm=LogNorm())

                if share_axes:
                    common_min = min(min(xs), min(ys))
                    common_max = max(max(xs), max(ys))
                    ax.set_xlim(common_min, common_max)
                    ax.set_ylim(common_min, common_max)

                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{title_suffix}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_heatmaps.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_heatmaps.svg", format="svg")
        
    def BIOS_signal_scatter_rank(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                continue
            # Loop over each gene
            for i, c in enumerate(cols):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["imp"]
                    scc = f"SRCC_GW: {eval_res[j]['Spearman-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    scc = f"SRCC_Gene: {eval_res[j]['Spearman_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    scc = f"SRCC_TSS: {eval_res[j]['Spearman_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    scc = f"SRCC_1obs: {eval_res[j]['Spearman_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    scc = f"SRCC_1imp: {eval_res[j]['Spearman_1imp']:.2f}"


                # Convert values to ranks
                xs = rankdata(xs)
                ys = rankdata(ys)

                ax.scatter(xs, ys, color="black", s=5, alpha=0.7)

                # Set the formatter for both axes
                formatter = mticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-1, 1))  # This will use scientific notation for numbers outside this range

                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_formatter(formatter)

                # Update the subplot with the new formatter
                plt.draw()  # This updates the current figure and applies the formatter
                
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{scc}", fontsize=9)
                ax.set_xlabel("Observed | rank")
                ax.set_ylabel("Predicted | rank")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_rank_scatters.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_rank_scatters.svg", format="svg")
    
    def BIOS_signal_rank_heatmap(self, eval_res, share_axes=True, bins=50):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            if "obs" not in eval_res[j]:
                continue
            for i, c in enumerate(cols):
                ax = plt.subplot(len(eval_res), len(cols), j * len(cols) + i + 1)

                if c == "GW":
                    xs, ys = eval_res[j]["obs"], eval_res[j]["imp"]
                    scc = f"SRCC_GW: {eval_res[j]['Spearman-GW']:.2f}"

                elif c == "gene":
                    xs, ys = self.metrics.get_gene_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    scc = f"SRCC_Gene: {eval_res[j]['Spearman_gene']:.2f}"
                    
                elif c == "TSS":
                    xs, ys = self.metrics.get_prom_signals(eval_res[j]["obs"], eval_res[j]["imp"], bin_size=self.resolution)
                    scc = f"SRCC_TSS: {eval_res[j]['Spearman_prom']:.2f}"

                elif c == "1obs":
                    xs, ys = self.metrics.get_1obs_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    scc = f"SRCC_1obs: {eval_res[j]['Spearman_1obs']:.2f}"

                elif c == "1imp":
                    xs, ys = self.metrics.get_1imp_signals(eval_res[j]["obs"], eval_res[j]["imp"])
                    scc = f"SRCC_1imp: {eval_res[j]['Spearman_1imp']:.2f}"

                # Convert values to ranks
                xs = rankdata(xs)
                ys = rankdata(ys)

                # Create the heatmap for ranked values
                h, xedges, yedges = np.histogram2d(xs, ys, bins=bins, density=True)
                h = h.T  # Transpose to correct the orientation
                ax.imshow(h, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='viridis', norm=LogNorm())

                if share_axes:
                    common_min = min(xedges[0], yedges[0])
                    common_max = max(xedges[-1], yedges[-1])
                    ax.set_xlim(common_min, common_max)
                    ax.set_ylim(common_min, common_max)

                ax.set_title(f"{eval_res[j]['feature']}_{c}_{eval_res[j]['comparison']}_{scc}")
                ax.set_xlabel("Observed | rank")
                ax.set_ylabel("Predicted | rank")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_rank_heatmaps.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_rank_heatmaps.svg", format="svg")
        
    def BIOS_corresp_curve(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        num_assays = len(eval_res)
        n_cols = math.floor(math.sqrt(num_assays))
        n_rows = math.ceil(num_assays / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=((4*n_cols), (4*n_rows)))

        c = 0

        for i in range(n_rows):
            for j in range(n_cols):

                if "obs" not in eval_res[c]:
                    continue

                if c>=num_assays:
                    continue
                
                t = [p[0] for p in eval_res[c]['corresp_curve']]
                psi = [p[1] for p in eval_res[c]['corresp_curve']]

                axs[i,j].plot(t, psi, c="red")

                axs[i,j].plot(t, t, "--", c="black")

                axs[i,j].set_title(f"{eval_res[c]['feature']}_{eval_res[c]['comparison']}")

                axs[i,j].fill_between(t, t, psi, color="red", alpha=0.4)

                c += 1
                axs[i,j].set_xlabel("t")
                axs[i,j].set_ylabel("psi")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/corresp_curve.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/corresp_curve.svg", format="svg")
    
    def BIOS_corresp_curve_deriv(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")
            
        num_assays = len(eval_res)
        n_cols = math.floor(math.sqrt(num_assays))
        n_rows = math.ceil(num_assays / n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=((4*n_cols), (4*n_rows)))

        c = 0

        for i in range(n_rows):
            for j in range(n_cols):

                if "obs" not in eval_res[c]:
                    continue

                if c>=num_assays:
                    continue
                    
                t = [p[0] for p in eval_res[c]['corresp_curve_deriv']]
                psii = [p[1] for p in eval_res[c]['corresp_curve_deriv']]

                axs[i,j].plot(t, psii, c="red")

                axs[i,j].plot(t, [1 for _ in range(len(t))], "--", c="black")

                axs[i,j].set_title(f"{eval_res[c]['feature']}_{eval_res[c]['comparison']}")

                axs[i,j].fill_between(t, [1 for _ in range(len(t))], psii, color="red", alpha=0.4)

                c += 1
                axs[i,j].set_xlabel("t")
                axs[i,j].set_ylabel("psi'")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/corresp_curve_deriv.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/corresp_curve_deriv.svg", format="svg")
    
    def BIOS_context_length_specific_performance(self, eval_res, context_length, bins=10):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        list_of_metrics = ['MSE-GW', 'Pearson-GW', 'Spearman-GW']

        # Define the size of the figure
        plt.figure(figsize=(6 * len(list_of_metrics), len(eval_res) * 2))

        # Loop over each result
        for j in range(len(eval_res)):
            # Loop over each gene
            if "obs" not in eval_res[j]:
                continue

            observed_values = eval_res[j]["obs"]
            imputed_values = eval_res[j]["imp"]

            bin_size = context_length // bins

            observed_values = observed_values.reshape(-1, context_length)
            imputed_values = imputed_values.reshape(-1, context_length)

            observed_values = observed_values.reshape(observed_values.shape[0]*bin_size, bins)
            imputed_values = imputed_values.reshape(imputed_values.shape[0]*bin_size, bins)

            for i, m in enumerate(list_of_metrics):
                # Create subplot for each result and gene combination
                ax = plt.subplot(len(eval_res), len(list_of_metrics), j * len(list_of_metrics) + i + 1)
                
                xs = [float(xt)/bins for xt in range(bins)]
                # Calculate x_values based on the current gene's coordinates
                ys = []
                for b in range(bins):
                    
                    obs, imp = observed_values[:,b].flatten(), imputed_values[:,b].flatten()
                    if m == 'MSE-GW':
                        ys.append(self.metrics.mse(obs, imp))

                    elif m == 'Pearson-GW':
                        ys.append(self.metrics.pearson(obs, imp))

                    elif m == 'Spearman-GW':
                        ys.append(self.metrics.spearman(obs, imp))
                
                ax.plot(xs, ys, color="grey", linewidth=3)
                # ax.fill_between(xs, 0, ys, alpha=0.7, color="grey")
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_xlabel("position in context")
                ax.set_ylabel(m)
        
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/context.png", dpi=150)
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/context.svg", format="svg")
    
    def MODEL_boxplot(self, df, metric):
        df = df.copy()
        # Sort the dataframe by 'feature'
        df.sort_values('feature', inplace=True)
        fig, axs = plt.subplots(2, figsize=(10, 6))
        fig.suptitle('Boxplots for Imputed and Denoised')

        # Boxplot for Imputed
        imputed_df = df[df['comparison'] == 'imputed']

        if "MSE" in metric:
            imputed_df[metric] = np.log(imputed_df[metric])
            
        sns.boxplot(x='feature', y=metric, data=imputed_df, ax=axs[0], color="grey")
        axs[0].set_title('Imputed')
        axs[0].set(xlabel='Assay', ylabel='log('+metric+')' if "MSE" in metric else metric)
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)  # Rotate x-axis labels

        # Boxplot for Denoised
        denoised_df = df[df['comparison'] == 'denoised']
        if "MSE" in metric:
            denoised_df[metric] = np.log(denoised_df[metric])

        sns.boxplot(x='feature', y=metric, data=denoised_df, ax=axs[1], color="grey")
        axs[1].set_title('Denoised')
        axs[1].set(xlabel='Assay', ylabel='log('+metric+')' if "MSE" in metric else metric)
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90)  # Rotate x-axis labels

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{metric}_boxplot.png", dpi=150)
        plt.savefig(f"{self.savedir}/{metric}_boxplot.svg", format="svg")
    
    def MODEL_regplot_overall(self, df, metric):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Scatter plots for Imputed and Denoised')

        # Plot for Imputed
        imputed_df = df[df['comparison'] == 'imputed']
        x_imputed = imputed_df['available train assays']
        y_imputed = imputed_df[metric]

        if "MSE" in metric:
            y_imputed = np.log(y_imputed)
        sns.regplot(x=x_imputed, y=y_imputed, scatter=True, line_kws={"color": "red"}, scatter_kws={"color": "red"}, ax=ax, label='Imputed')
        
        # Plot for Denoised
        denoised_df = df[df['comparison'] == 'denoised']
        x_denoised = denoised_df['available train assays']
        y_denoised = denoised_df[metric]

        if "MSE" in metric:
            y_denoised = np.log(y_denoised)
        sns.regplot(x=x_denoised, y=y_denoised, scatter=True, line_kws={"color": "green"}, scatter_kws={"color": "green"}, ax=ax, label='Denoised')
        
        ax.set(xlabel='Number of Available Train Assays', ylabel='log('+metric+')' if "MSE" in metric else metric)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{metric}_overall_regplot.png", dpi=200)
        plt.savefig(f"{self.savedir}/{metric}_overall_regplot.svg", format="svg")
    
    def MODEL_regplot_perassay(self, df, metric):
        # Get the unique features (assays)
        features = df['feature'].unique()
        num_features = len(features)

        # Determine the layout of the subplots
        n_cols = math.ceil(math.sqrt(num_features))
        n_rows = math.ceil(num_features / n_cols)

        # Create a large figure to accommodate all subplots
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)
        
        # Flatten the array of axes for easy iteration
        axs = axs.flatten()

        # Iterate over each unique feature and create a subplot
        for i, feature in enumerate(features):
            # Data for current feature
            feature_df = df[df['feature'] == feature]
            
            # Plot for Imputed
            imputed_df = feature_df[feature_df['comparison'] == 'imputed']
            x_imputed = imputed_df['available train assays']
            y_imputed = imputed_df[metric]

            if "MSE" in metric:
                y_imputed = np.log(y_imputed)
            
            sns.regplot(x=x_imputed, y=y_imputed, scatter=True, line_kws={"color": "red"}, scatter_kws={"color": "red"}, ax=axs[i], label='Imputed')
            
            # Plot for Denoised
            denoised_df = feature_df[feature_df['comparison'] == 'denoised']
            x_denoised = denoised_df['available train assays']
            y_denoised = denoised_df[metric]

            if "MSE" in metric:
                y_denoised = np.log(y_denoised)
            
            sns.regplot(x=x_denoised, y=y_denoised, scatter=True, line_kws={"color": "green"}, scatter_kws={"color": "green"}, ax=axs[i], label='Denoised')
            
            # Set the title and labels
            axs[i].set_title(feature)
            axs[i].set_xlabel('Number of Available Train Assays')
            axs[i].set_ylabel('log('+metric+')' if "MSE" in metric else metric)
            axs[i].legend()

            # Turn off axes for any empty subplots
            if i >= num_features:
                axs[i].axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{metric}_per_assay_metric.png", dpi=200)
        plt.savefig(f"{self.savedir}/{metric}_per_assay_metric.svg", format="svg")

class EVAL_EIC(object): # on chr21
    def __init__(
        self, model, traindata_path, evaldata_path, context_length, batch_size, hyper_parameters_path="",
        train_log={}, chr_sizes_file="data/hg38.chrom.sizes", version="22", resolution=25, 
        is_arcsin=True, savedir="models/evals/", mode="eval"):

        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.mkdir(self.savedir)

        self.traindata_path = traindata_path
        self.evaldata_path = evaldata_path
        self.is_arcsin = is_arcsin
        self.version = version
        self.context_length = context_length
        self.batch_size = batch_size

        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.all_assays = ['M{:02d}'.format(i) for i in range(1, 36)]
        self.mark_dict = {
            "M01": "ATAC-seq", "M02": "DNase-seq", "M03": "H2AFZ",
            "M04": "H2AK5ac", "M05": "H2AK9ac", "M06": "H2BK120ac",
            "M07": "H2BK12ac", "M08": "H2BK15ac", "M09": "H2BK20ac",
            "M10": "H2BK5ac", "M11": "H3F3A", "M12": "H3K14ac",
            "M13": "H3K18ac", "M14": "H3K23ac", "M15": "H3K23me2",
            "M16": "H3K27ac", "M17": "H3K27me3", "M18": "H3K36me3",
            "M19": "H3K4ac", "M20": "H3K4me1", "M21": "H3K4me2",
            "M22": "H3K4me3", "M23": "H3K56ac", "M24": "H3K79me1",
            "M25": "H3K79me2", "M26": "H3K9ac", "M27": "H3K9me1",
            "M28": "H3K9me2", "M29": "H3K9me3", "M30": "H3T11ph",
            "M31": "H4K12ac", "M32": "H4K20me1", "M33": "H4K5ac",
            "M34": "H4K8ac", "M35": "H4K91ac"
        }

        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        self.chr_sizes = {}
        self.resolution = resolution

        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

        self.train_data = {}
        self.eval_data = {}
        self.metrics = METRICS()
        self.viz = VISUALS(resolution=self.resolution, savedir=self.savedir)

        if mode == "dev":
            return

        if type(self.model) == str:
            with open(hyper_parameters_path, 'rb') as f:
                self.hyper_parameters = pickle.load(f)
            loader = MODEL_LOADER(model, self.hyper_parameters)
            self.model = loader.load_epidenoise(version=self.version)

        self.model = self.model.to(self.device)
        self.model.eval()  # set the model to evaluation mode
        print(f"# model_parameters: {count_parameters(self.model)}")

        # load and bin chr21 of all bigwig files 
        for t in os.listdir(traindata_path):
            if ".bigwig" in t:

                for e in os.listdir(evaldata_path):
                    if ".bigwig" in e:
                        
                        if t[:3] == e[:3]:

                            if t[:3] not in self.train_data:
                                self.train_data[t[:3]] = {}

                            if e[:3] not in self.eval_data:
                                self.eval_data[e[:3]] = {}

                            self.train_data[t[:3]][t[3:6]] = traindata_path + "/" + t
                            self.eval_data[e[:3]][e[3:6]] = evaldata_path + "/" + e

        print(self.eval_data.keys())
        # print(self.train_data.keys())

    def load_tensor(self, bios_name, mode="train"):
        chr, start, end = "chr21", 0, self.chr_sizes["chr21"]
        all_samples = []
        missing_ind = []

        if mode  == "train": 
            source = self.train_data
            savepath = self.traindata_path + f"/{bios_name}_chr21_{self.resolution}.pt"
        elif mode == "eval":
            source = self.eval_data
            savepath = self.evaldata_path + f"/{bios_name}_chr21_{self.resolution}.pt"
        
        if os.path.exists(savepath):
            all_samples = torch.load(savepath)
            # fill-in missing_ind
            for i in range(all_samples.shape[1]):
                if (all_samples[:, i] == -1).all():
                    missing_ind.append(i)
                    
            return all_samples, missing_ind

        else:
            for i in range(len(self.all_assays)):
                assay = self.all_assays[i]
                if assay in source[bios_name].keys():
                    print("loading ", assay)
                    bw = pyBigWig.open(source[bios_name][assay])
                    signals = bw.stats(chr, start, end, type="mean", nBins=(end - start) // self.resolution)
                
                else:
                    print(assay, "missing")
                    signals = [-1 for _ in range((end - start) // self.resolution)]
                    missing_ind.append(i)

                all_samples.append(signals)

            all_samples = torch.from_numpy(np.array(all_samples, dtype=np.float32)).transpose_(0, 1)

            # replace NaN with zero
            all_samples = torch.where(torch.isnan(all_samples), torch.zeros_like(all_samples), all_samples)

            nan_count = torch.isnan(all_samples).sum().item()
            minus_one_count = (all_samples == -1).sum().item()

            torch.save(all_samples, savepath)
            
            return all_samples, missing_ind

    def load_bios(self, bios_name):
        X, missing_x_i = self.load_tensor(bios_name, mode="train")
        Y, missing_y_i = self.load_tensor(bios_name, mode="eval")

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        if self.is_arcsin:
            arcmask1 = (X != -1)
            X[arcmask1] = torch.arcsinh_(X[arcmask1])

            arcmask2 = (Y != -1)
            Y[arcmask2] = torch.arcsinh_(Y[arcmask2])

        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])
        
        return X, Y, missing_x_i, missing_y_i

    def get_imp(self, X, missing_x_i): # X: train data
        d_model = X.shape[-1]

        # Initialize a tensor to store all predictions
        P = torch.empty_like(X, device="cpu")

        # make predictions in batches
        for i in range(0, len(X), self.batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+ self.batch_size]

            with torch.no_grad():
                x_batch = x_batch.to(self.device)

                if self.version == "18":
                    outputs, pred_mask = self.model(x_batch)

                elif self.version in ["20", "21"]:
                    mask = torch.zeros_like(x_batch, dtype=torch.bool, device=self.device)
                    for ii in missing_x_i: 
                        mask[:,:,ii] = True

                    mask = mask.to(self.device)
                    outputs, pred_mask = self.model(x_batch, mask)
                
                elif self.version=="22":
                    token_dict = {
                        "missing_mask": -1, 
                        "cloze_mask": -2,
                        "pad": -3
                    }
                    # change missing token to cloze token to force prediction
                    x_batch_missing_vals = (x_batch == -1)
                    x_batch[x_batch_missing_vals] = token_dict["cloze_mask"] 

                    mask = torch.zeros_like(x_batch, dtype=torch.bool, device=self.device)
                    for ii in missing_x_i: 
                        mask[:,:,ii] = True

                    mask = mask.to(self.device)


                    # outputs, aggrmean, aggrstd = self.model(x_batch, mask, None)
                    outputs = self.model(x_batch, mask, None)

            # Store the predictions in the large tensor
            P[i:i+outputs.shape[0], :, :] = outputs.cpu()
        
        return P

    def get_metrics(self, X, Y, P, missing_x_i, missing_y_i, bios_name):
        """
        reportoir of metrics -- per_bios:

            peak_ovr: 01thr, 05thr, 10thr

            GeWi: MSE, Pearson, Spearman
            1imp: MSE, Pearson, Spearman
            1obs: MSE, Pearson, Spearman
            gene: MSE, Pearson, Spearman
            prom: MSE, Pearson, Spearman
        """

        results = []
        
        for j in range(Y.shape[-1]):  # for each feature i.e. assay
            pred = P[:, j].numpy()
            metrics_list = []

            if j in missing_x_i and j not in missing_y_i:  # if the feature is missing in the input
                target = Y[:, j].numpy()
                comparison = 'imputed'
            
            elif j not in missing_x_i:
                target = X[:, j].numpy()
                comparison = 'denoised'

            else:
                continue
                
            # if np.isnan(pred).any():
            #     print(f"{self.mark_dict[self.all_assays[j]]} contains nan. skipping")
            #     continue
            # else:
            #     print(f"{self.mark_dict[self.all_assays[j]]} worked")

            # corresp, corresp_deriv = self.metrics.correspondence_curve(target, pred)
            metrics = {
                'bios':bios_name,
                'feature': self.mark_dict[self.all_assays[j]],
                'comparison': comparison,
                'available train assays': len(self.all_assays) - len(missing_x_i),
                'available eval assays': len(self.all_assays) - len(missing_y_i),

                "obs":target,
                "imp":pred,

                'MSE-GW': self.metrics.mse(target, pred),
                'Pearson-GW': self.metrics.pearson(target, pred),
                'Spearman-GW': self.metrics.spearman(target, pred),

                'MSE-1obs': self.metrics.mse1obs(target, pred),
                'Pearson_1obs': self.metrics.pearson1_obs(target, pred),
                'Spearman_1obs': self.metrics.spearman1_obs(target, pred),

                'MSE-1imp': self.metrics.mse1imp(target, pred),
                'Pearson_1imp': self.metrics.pearson1_imp(target, pred),
                'Spearman_1imp': self.metrics.spearman1_imp(target, pred),

                'MSE-gene': self.metrics.mse_gene(target, pred),
                'Pearson_gene': self.metrics.pearson_gene(target, pred),
                'Spearman_gene': self.metrics.spearman_gene(target, pred),

                'MSE-prom': self.metrics.mse_prom(target, pred),
                'Pearson_prom': self.metrics.pearson_prom(target, pred),
                'Spearman_prom': self.metrics.spearman_prom(target, pred),

                # "peak_overlap_01thr": self.metrics.peak_overlap(target, pred, p=0.01),
                # "peak_overlap_05thr": self.metrics.peak_overlap(target, pred, p=0.05),
                # "peak_overlap_10thr": self.metrics.peak_overlap(target, pred, p=0.10),

            #     "corresp_curve": corresp,
            #     "corresp_curve_deriv": corresp_deriv
            }
            results.append(metrics)
        
        return results

    def bios_pipeline(self, bios_name):
        X, Y, missing_x_i, missing_y_i = self.load_bios(bios_name)
        P = self.get_imp(X, missing_x_i)

        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1]) # preds
        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1]) # eval data
        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1]) # train data

        eval_res = self.get_metrics(X, Y, P, missing_x_i, missing_y_i, bios_name)

        return eval_res

    def bios_test(self):
        missing_x_i, missing_y_i = [], []
        
        X = torch.load("data/C23_trn.pt")
        Y = torch.load("data/C23_val.pt")
        P = torch.load("data/C23_imp.pt")

        
        # fill-in missing_ind
        for i in range(X.shape[1]):
            if (X[:, i] == -1).all():
                missing_x_i.append(i)
        
        # fill-in missing_ind
        for i in range(Y.shape[1]):
            if (Y[:, i] == -1).all():
                missing_y_i.append(i)

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        if self.is_arcsin:
            arcmask1 = (X != -1)
            X[arcmask1] = torch.arcsinh_(X[arcmask1])

            arcmask2 = (Y != -1)
            Y[arcmask2] = torch.arcsinh_(Y[arcmask2])

        eval_res = self.get_metrics(X, Y, P, missing_x_i, missing_y_i, "test")

        self.viz.BIOS_context_length_specific_performance(eval_res, self.context_length, bins=10)
        self.viz.clear_pallete()

        # self.viz.BIOS_signal_scatter_with_marginals(eval_res)
        # self.viz.clear_pallete()

        # self.viz.BIOS_signal_heatmap(eval_res)
        # self.viz.clear_pallete()

        # self.viz.BIOS_signal_rank_heatmap(eval_res)
        # self.viz.clear_pallete()

    def viz_bios(self, eval_res):
        """
        visualizations -- per_bios:

            highlight imputed vs denoised
            corresp curve + deriv

            scatter_gewi: value, rank 
            scatter_gene: value, rank 
            scatter_prom: value, rank 
            scatter_1imp: value, rank 
            scatter_1obs: value, rank 

            selected regions' signals
        """

        try: 
            print("plotting signal tracks")
            self.viz.BIOS_signal_track(eval_res)
            self.viz.clear_pallete()
        except:
            print("faild to plot signal tracks")

        try:
            print("plotting context_specific performance")
            self.viz.BIOS_context_length_specific_performance(eval_res, self.context_length, bins=10)
            self.viz.clear_pallete()
        except:
            print("faild to plot context_specific performance")
            
        try:
            print("plotting signal scatter")
            self.viz.BIOS_signal_scatter(eval_res)
            self.viz.clear_pallete()
        except:
            print("faild to plot  signal scatter")

        try:
            print("plotting signal scatter with marginals")
            self.viz.BIOS_signal_scatter_with_marginals(eval_res)
            self.viz.clear_pallete()
        except:
            print("faild to plot scatter with marginals")

        try:
            print("plotting signal heatmap")
            self.viz.BIOS_signal_heatmap(eval_res)
            self.viz.clear_pallete()
        except:
            print("faild to plot  signal heatmap")

        try:
            print("plotting signal rank heatmap")
            self.viz.BIOS_signal_rank_heatmap(eval_res)
            self.viz.clear_pallete()
        except:
            print("faild to plot  signal rank heatmap")

        # try:
        #     print("plotting corresp_curve")
        #     self.viz.BIOS_corresp_curve(eval_res)
        #     self.viz.clear_pallete()
        # except:
        #     print("faild to plot corresp_curve")

        # try:
        #     print("plotting corresp_curve_deriv")
        #     self.viz.BIOS_corresp_curve_deriv(eval_res)
        #     self.viz.clear_pallete()
        # except:
        #     print("faild to plot corresp_curve_deriv")
    
    def viz_all(self):
        """
        visualizations -- all_bios:
        
            denoised vs imputed
                boxplots for metric per assay
                    peak_ovr: 01thr, 05thr, 10thr
                    GeWi: MSE, Pearson, Spearman
                    1imp: MSE, Pearson, Spearman
                    1obs: MSE, Pearson, Spearman
                    gene: MSE, Pearson, Spearman
                    prom: MSE, Pearson, Spearman
        """
        self.model_res = []
        for bios in self.eval_data.keys():
            print("evaluating ", bios)
            eval_res_bios = self.bios_pipeline(bios)
            print("got results for ", bios)
            self.viz_bios(eval_res_bios)

            for f in eval_res_bios:
                del f["obs"], f["imp"]
                self.model_res.append(f)

        self.model_res = pd.DataFrame(self.model_res)
        self.model_res.to_csv(f"{self.savedir}/model_eval.csv", index=False)

        boxplot_metrics = [
            'MSE-GW', 'Pearson-GW', 'Spearman-GW',
            'MSE-1obs', 'Pearson_1obs', 'Spearman_1obs',
            'MSE-1imp', 'Pearson_1imp', 'Spearman_1imp',
            'MSE-gene', 'Pearson_gene', 'Spearman_gene',
            'MSE-prom', 'Pearson_prom', 'Spearman_prom',
            'peak_overlap_01thr', 'peak_overlap_05thr', 
            'peak_overlap_10thr']
        
        for m in boxplot_metrics:
            self.viz.MODEL_boxplot(self.model_res, metric=m)
            self.viz.MODEL_regplot_overall(self.model_res, metric=m)
            self.viz.MODEL_regplot_perassay(self.model_res, metric=m)

class EmbedMetadata(nn.Module):
    def __init__(self, input_dim, embedding_dim, non_linearity=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim 
        self.non_linearity = non_linearity
        self.continuous_size = embedding_dim // 3

        self.runtype_embedding = nn.Embedding(4, self.continuous_size)  # 4 classes: single_end, pair_end, missing, cloze_masked
        self.depth_transform = nn.Linear(1, self.continuous_size) 
        self.coverage_transform = nn.Linear(1, self.continuous_size)
        self.read_length_transform = nn.Linear(1, self.continuous_size)

        self.final_embedding = nn.Linear(self.input_dim * self.continuous_size * 4, embedding_dim)  # Adjusted for all inputs
        self.final_emb_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, metadata):
        depth = metadata[:, 0, :].unsqueeze(-1).float() 
        coverage = metadata[:, 1, :].unsqueeze(-1).float() 
        read_length = metadata[:, 2, :].unsqueeze(-1).float() 
        runtype = metadata[:, 3, :].long() 
        
        runtype = torch.where(runtype == -1, torch.tensor(2, device=runtype.device), runtype) # missing
        runtype = torch.where(runtype == -2, torch.tensor(3, device=runtype.device), runtype) # cloze_masked

        depth_embed = self.depth_transform(depth)
        coverage_embed = self.coverage_transform(coverage)
        read_length_embed = self.read_length_transform(read_length)
        runtype_embed = self.runtype_embedding(runtype)

        embeddings = torch.cat([depth_embed, coverage_embed, read_length_embed, runtype_embed], dim=-1)
        embeddings = embeddings.view(embeddings.shape[0], -1)

        embeddings = self.final_emb_layer_norm(self.final_embedding(embeddings))
        
        if self.non_linearity:
            embeddings = F.relu(embeddings)
        
        return embeddings

class EmbedMetadata(nn.Module):
    def __init__(self, input_dim, embedding_dim, non_linearity=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim 
        self.non_linearity = non_linearity
        self.continuous_size = embedding_dim // 3

        # Separate embeddings for each pair of metadata and feature
        self.depth_transform = nn.ModuleList([nn.Linear(1, self.continuous_size) for _ in range(input_dim)])
        self.coverage_transform = nn.ModuleList([nn.Linear(1, self.continuous_size) for _ in range(input_dim)])
        self.read_length_transform = nn.ModuleList([nn.Linear(1, self.continuous_size) for _ in range(input_dim)])
        self.runtype_embedding = nn.ModuleList([nn.Embedding(4, self.continuous_size) for _ in range(input_dim)])
        
        # Separate final embedding layers for each feature
        self.final_embedding = nn.ModuleList([nn.Linear(self.continuous_size * 4, 1) for _ in range(input_dim)])

    def forward(self, metadata):
        embeddings_list = []

        for i in range(metadata.shape[2]):
            depth = metadata[:, 0, i].unsqueeze(-1).float()
            coverage = metadata[:, 1, i].unsqueeze(-1).float()
            read_length = metadata[:, 2, i].unsqueeze(-1).float()
            runtype = metadata[:, 3, i].long()
            
            runtype = torch.where(runtype == -1, torch.tensor(2, device=runtype.device), runtype) # missing
            runtype = torch.where(runtype == -2, torch.tensor(3, device=runtype.device), runtype) # cloze_masked

            depth_embed = self.depth_transform[i](depth)
            coverage_embed = self.coverage_transform[i](coverage)
            read_length_embed = self.read_length_transform[i](read_length)
            runtype_embed = self.runtype_embedding[i](runtype)

            feature_embedding = torch.cat([depth_embed, coverage_embed, read_length_embed, runtype_embed], dim=-1)
            feature_embedding = self.final_embedding[i](feature_embedding)

            if self.non_linearity:
                feature_embedding = F.relu(feature_embedding)
            
            embeddings_list.append(feature_embedding)

        embeddings = torch.cat(embeddings_list, dim=-1)
        return embeddings

class DualConvEmbedding(nn.Module):
    def __init__(self, in_C, out_C, do_batchnorm=True):
        super(DualConvEmbedding, self).__init__()
        self.do_batchnorm = do_batchnorm
        if self.do_batchnorm:
            self.batch_norm = nn.BatchNorm1d(out_C)
        
        self.convF = nn.Conv1d(in_C, out_C, kernel_size=1, dilation=1, stride=1, padding="same")
        self.convM = nn.Conv1d(in_C, out_C, kernel_size=1, dilation=1, stride=1, padding="same")
        
        self.act = nn.ReLU()
        
    def forward(self, F, M): # F is feature matrix # M is the binary mask matrix
        F = self.convF(F)
        if self.do_batchnorm:
            F = self.batch_norm(F)
        F = self.act(F)

        M = self.convM(M)
        if self.do_batchnorm:
            M = self.batch_norm(M)
        M = self.act(M)
        
        return F * M

class SoftmaxPooling1D(nn.Module):
    def __init__(self, pool_size, per_channel=False, w_init_scale=1.0):
        super(SoftmaxPooling1D, self).__init__()
        self.pool_size = pool_size
        self.per_channel = per_channel
        self.w_init_scale = w_init_scale
        self.weights_initialized = False  # Track whether weights have been initialized

    def forward(self, inputs):
        device = inputs.device
        inputs = inputs.permute(0, 2, 1)
        batch_size, length, num_features = inputs.size()
        
        if not self.weights_initialized:
            output_size = num_features if self.per_channel else 1
            identity = torch.eye(num_features, device=device)
            init_val = identity * self.w_init_scale
            weights_val = init_val.repeat(1, output_size) if self.per_channel else init_val.mean(dim=0, keepdim=True).repeat(output_size, 1)
            self.weights = nn.Parameter(weights_val)
            self.register_parameter('softmax_weights', self.weights)
            self.weights_initialized = True
        else:
            self.weights = self.weights.to(device)
        
        # Ensure the length is divisible by the pool size for simplicity
        if length % self.pool_size != 0:
            padding = self.pool_size - length % self.pool_size
            inputs = F.pad(inputs, (0, 0, 0, padding))
            _, length, _ = inputs.size()
        
        # Reshape inputs for pooling
        inputs = inputs.unfold(
            1, self.pool_size, self.pool_size).contiguous().view(batch_size, -1, self.pool_size, num_features)

        # Calculate logits using einsum for simplicity here
        logits = torch.einsum('bnpc,pc->bnp', inputs, self.weights)

        # Apply softmax to the logits along the pooling window dimension
        softmax_weights = F.softmax(logits, dim=2)
        
        # Multiply inputs by softmax weights and sum over the pooling window
        pooled = torch.einsum('bnpc,bnp->bnc', inputs, softmax_weights)
        
        return pooled.permute(0,2,1)

class AttentionPooling1D(nn.Module):
    def __init__(self, in_channels, pooling_size=2):
        super(AttentionPooling1D, self).__init__()
        self.pooling_size = pooling_size
        self.attention_weights = nn.Parameter(torch.randn(in_channels, pooling_size))

    def forward(self, x):
        # x shape: (batch_size, channels, length)

        # Split the input tensor into pools
        unfolded = x.unfold(dimension=2, size=self.pooling_size, step=self.pooling_size)
        # unfolded shape: (batch_size, channels, num_pools, pooling_size)

        # Compute attention scores and apply them
        scores = F.softmax(self.attention_weights, dim=1)
        pooled = (unfolded * scores.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
        # pooled shape: (batch_size, channels, num_pools)

        pooled = pooled.permute(0, 2, 1)
        
        return pooled

class MetadataEmbeddingModule(nn.Module):
    def __init__(self, input_dim, embedding_dim, non_linearity=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim 
        self.non_linearity = non_linearity
        self.continuous_size = embedding_dim // 3

        # self.avail_embedding = nn.Embedding(3, self.continuous_size)  # 3 classes: available, missing, cloze_masked

        # X metadata embedding parameters
        self.xruntype_embedding = nn.Embedding(4, self.continuous_size)  # 4 classes: single_end, pair_end, missing, cloze_masked
        self.xdepth_transform = nn.Linear(1, self.continuous_size) 
        self.xcoverage_transform = nn.Linear(1, self.continuous_size)
        self.xread_length_transform = nn.Linear(1, self.continuous_size)

        # Y metadata embedding parameters
        self.yruntype_embedding = nn.Embedding(3, self.continuous_size)  # 3 classes: single_end, pair_end, missing
        self.ydepth_transform = nn.Linear(1, self.continuous_size) 
        self.ycoverage_transform = nn.Linear(1, self.continuous_size)
        self.yread_length_transform = nn.Linear(1, self.continuous_size)

        # Final layer to combine all embeddings
        # self.final_embedding = nn.Linear(self.input_dim * self.continuous_size * 9, embedding_dim)  # Adjusted for all inputs
        self.final_embedding = nn.Linear(self.input_dim * self.continuous_size * 8, embedding_dim)  # Adjusted for all inputs
        self.final_emb_layer_norm = nn.LayerNorm(embedding_dim)

    def embed_metadata(self, metadata, side="x"):
        depth = metadata[:, 0, :].unsqueeze(-1).float() 
        coverage = metadata[:, 1, :].unsqueeze(-1).float() 
        read_length = metadata[:, 2, :].unsqueeze(-1).float() 
        runtype = metadata[:, 3, :].long() 
        
        runtype = torch.where(runtype == -1, torch.tensor(2, device=runtype.device), runtype) # missing
        runtype = torch.where(runtype == -2, torch.tensor(3, device=runtype.device), runtype) # cloze_masked

        if side == "x":
            depth_embed = self.xdepth_transform(depth)
            coverage_embed = self.xcoverage_transform(coverage)
            read_length_embed = self.xread_length_transform(read_length)
            runtype_embed = self.xruntype_embedding(runtype)

        elif side == "y":
            depth_embed = self.ydepth_transform(depth)
            coverage_embed = self.ycoverage_transform(coverage)
            read_length_embed = self.yread_length_transform(read_length)
            runtype_embed = self.yruntype_embedding(runtype)

        if self.non_linearity:
            depth_embed = F.relu(depth_embed)
            coverage_embed = F.relu(coverage_embed)
            read_length_embed = F.relu(read_length_embed)

        # Concatenate all embeddings
        embeddings = torch.cat([depth_embed, coverage_embed, read_length_embed, runtype_embed], dim=-1)
        return embeddings

    def embed_avail(self, availability):
        availability = availability.long()
        availability = torch.where(
            availability == -2, torch.tensor(2, device=availability.device), availability)

        availability_embed = self.avail_embedding(availability)
        return availability_embed

    def forward(self, x_metadata, y_metadata, availability):
        Xmd_embed = self.embed_metadata(x_metadata, side="x")
        Ymd_embed = self.embed_metadata(y_metadata, side="y")
        # av_embed = self.embed_avail(availability)

        # Concatenate all embeddings along the last dimension
        # full_embed = torch.cat([Xmd_embed, Ymd_embed, av_embed], dim=-1)
        full_embed = torch.cat([Xmd_embed, Ymd_embed], dim=-1)

        full_embed = full_embed.view(full_embed.shape[0], -1)
        full_embed = self.final_embedding(full_embed)

        full_embed = F.relu(self.final_emb_layer_norm(full_embed))

        return full_embed

class PositionalEncoding10(nn.Module):

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

class AbsPositionalEmbedding15(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        for pos in range(max_len):   
            # for each dimension of the each position
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # include the batch size
        self.pe = pe.unsqueeze(0)   
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pe = pe.to(device)
        # self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe.unsqueeze(1)

class ComboEmbedding15(nn.Module):
    """
    Combo Embedding which is consisted with under features
        1. AbsPositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding segment info, (seg_A:1, seg_B:2)
        sum of all these features are output of ComboEmbedding
    """

    def __init__(self, d_model, seq_len=64, dropout=0.1):
        """
        :param d_model: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.segment = torch.nn.Embedding(3, d_model, padding_idx=0)
        self.position = AbsPositionalEmbedding15(d_model=d_model, max_len=seq_len)
        self.dropout = torch.nn.Dropout(p=dropout)
       
    def forward(self, sequence, segment_label):
        x = sequence + self.position(sequence).unsqueeze(1) + self.segment(segment_label).unsqueeze(1) 
        return self.dropout(x)
        
class MatrixFactorizationEmbedding(nn.Module):
    """
    learns two factorizations from input matrix M of size l*d
    U of size l*k
    V of size d*k
    UV^T is should reconstruct the input matrix M
    """

    def __init__(self, l, d, k):
        super().__init__()
        self.dense_U = nn.Linear(l, k)
        self.dense_V = nn.Linear(d, k)
        self.relu = nn.ReLU()
        # self.dense_U = FeedForwardNN(l, 4*k, k, 2)
        # self.dense_V = FeedForwardNN(d, 4*k, k, 2)
       
    def forward(self, M, linear=False):
        # shape of M is (N, L, D)
        U = self.dense_U(torch.permute(M, (0, 2, 1))) 
        V = self.dense_V(M)

        if not linear:
            U = self.relu(U)
            V = self.relu(V)
        
        V = torch.permute(V, (0, 2, 1))
        M = torch.matmul(U, V)

        return torch.permute(M, (0, 2, 1))

#========================================================================================================#
#=========================================== Loss Functions =============================================#
#========================================================================================================#

class ComboLoss15(nn.Module):
    def __init__(self, alpha=0.5):
        super(ComboLoss15, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.alpha = alpha

    def forward(self, pred_signals, true_signals, pred_adjac, true_adjac):

        mse_loss = self.mse_loss(pred_signals, true_signals)

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_adjac).any() or torch.isnan(true_adjac).any():
            # print("NaN value encountered in pred_adjac or true_adjac.")
            return torch.tensor(float('nan')).to(pred_signals.device)

        bce_loss = self.bce_loss(pred_adjac, true_adjac)

        if torch.isnan(mse_loss) or torch.isnan(bce_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device)
        
        return self.alpha * mse_loss + (1 - self.alpha) * bce_loss

class ComboLoss16(nn.Module):
    def __init__(self):
        super(ComboLoss16, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, pred_adjac, true_adjac, pred_mask, cloze_mask, union_mask):
        mse_obs_loss =  self.l1_loss(pred_signals[~union_mask], true_signals[~union_mask])
        mse_pred_loss = self.l1_loss(pred_signals[cloze_mask], true_signals[cloze_mask])

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_adjac).any() or torch.isnan(true_adjac).any() or torch.isnan(pred_mask).any():
            # print("NaN value encountered in pred_adjac or true_adjac.")
            return torch.tensor(float('nan')).to(pred_signals.device)

        bce_mask_loss = self.bce_loss(pred_mask, union_mask.float())
        # SAP_bce_loss = self.bce_loss(pred_adjac, true_adjac)

        if torch.isnan(mse_obs_loss) or torch.isnan(mse_pred_loss) or torch.isnan(bce_mask_loss):# or torch.isnan(SAP_bce_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device)
        
        return mse_obs_loss + mse_pred_loss + bce_mask_loss #+ SAP_bce_loss

class ComboLoss17(nn.Module):
    def __init__(self):
        super(ComboLoss17, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, pred_adjac, true_adjac, pred_mask, obs_mask):

        mse_obs_loss = self.l1_loss(pred_signals[obs_mask], true_signals[obs_mask])
        mse_pred_loss = self.l1_loss(pred_signals[pred_mask], true_signals[pred_mask])

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_adjac).any() or torch.isnan(true_adjac).any():
            # print("NaN value encountered in pred_adjac or true_adjac.")
            return torch.tensor(float('nan')).to(pred_signals.device)

        bce_loss = self.bce_loss(pred_adjac, true_adjac)

        if torch.isnan(mse_obs_loss) or torch.isnan(mse_pred_loss) or torch.isnan(bce_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device)
        
        return mse_obs_loss + mse_pred_loss #+ bce_loss

class ComboLoss18(nn.Module):
    def __init__(self):
        super(ComboLoss18, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, pred_mask, cloze_mask, union_mask):
        mse_obs_loss =  self.l1_loss(pred_signals[~union_mask], true_signals[~union_mask])
        mse_pred_loss = self.l1_loss(pred_signals[cloze_mask], true_signals[cloze_mask])

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_signals).any() or torch.isnan(pred_mask).any():
            return torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device)

        bce_mask_loss = self.bce_loss(pred_mask, union_mask.float())

        if torch.isnan(mse_obs_loss) or torch.isnan(mse_pred_loss) or torch.isnan(bce_mask_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device)
        
        return mse_obs_loss, mse_pred_loss, bce_mask_loss

class ComboLoss20(nn.Module):
    def __init__(self):
        super(ComboLoss20, self).__init__()
        self.l1_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, pred_mask, cloze_mask, union_mask):
        mse_obs_loss =  self.l1_loss(pred_signals[~union_mask], true_signals[~union_mask])
        mse_pred_loss = self.l1_loss(pred_signals[cloze_mask], true_signals[cloze_mask])

        # Check for nan values in pred_adjac and true_adjac
        if torch.isnan(pred_signals).any() or torch.isnan(pred_mask).any():
            return torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device)

        bce_mask_loss = self.bce_loss(pred_mask, union_mask.float())

        if torch.isnan(mse_obs_loss) or torch.isnan(mse_pred_loss) or torch.isnan(bce_mask_loss):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device), torch.tensor(float('nan')).to(pred_signals.device)
        
        return mse_obs_loss, mse_pred_loss, bce_mask_loss

class ComboLoss21(nn.Module):
    def __init__(self):
        super(ComboLoss21, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, union_mask, next_pos_mask):
        # mse_obs_loss =  self.mse_loss(pred_signals[~union_mask], true_signals[~union_mask])
        mse_next_pos = self.mse_loss(pred_signals[next_pos_mask], true_signals[next_pos_mask])

        if torch.isnan(pred_signals).any() or torch.isnan(mse_next_pos):
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(pred_signals.device)
        return mse_next_pos

class ComboLoss22(nn.Module):
    def __init__(self, alpha=0.75):
        super(ComboLoss22, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, pred_signals, true_signals, cloze_mask, union_mask):#, aggrmean, aggrstd, aggr_mask):

        # true_seq_mean = true_signals.mean(dim=1)
        # true_seq_std = true_signals.std(dim=1)

        # mse_aggrmean_loss = self.mse_loss(aggrmean[aggr_mask], true_seq_mean[aggr_mask]) 
        # mse_aggrstd_loss  = self.mse_loss(aggrstd[aggr_mask], true_seq_std[aggr_mask]) 

        mse_obs_loss =  self.mse_loss(pred_signals[~union_mask], true_signals[~union_mask])
        mse_pred_loss = self.mse_loss(pred_signals[cloze_mask], true_signals[cloze_mask])

        if torch.isnan(mse_pred_loss).any() or torch.isnan(mse_obs_loss).any():
            print("NaN value encountered in loss components.")
            return torch.tensor(float('nan')).to(mse_pred_loss.device), torch.tensor(float('nan')).to(mse_pred_loss.device)

        return self.alpha*mse_pred_loss, (1-self.alpha)*mse_obs_loss#, mse_aggrmean_loss, mse_aggrstd_loss

class ComboLoss_PoissonNLL(nn.Module):
    def __init__(self, alpha=0.5):
        super(ComboLoss30a_PoissonNLL, self).__init__()
        self.alpha = alpha
        self.nll_loss = nn.PoissonNLLLoss(log_input=False, reduction='mean', full=True)

    def forward(self, pred_signals, true_signals, masked_map, obs_map):

        obs_loss =  self.nll_loss(pred_signals[obs_map], true_signals[obs_map])
        pred_loss = self.nll_loss(pred_signals[masked_map], true_signals[masked_map])
        
        return self.alpha*pred_loss, (1-self.alpha)*obs_loss

class ComboLoss_NBNLL(nn.Module):
    def __init__(self, alpha=0.5):
        super(ComboLoss_NBNLL, self).__init__()
        self.alpha = alpha
        self.reduction = 'sum'

    def forward(self, p_pred, n_pred, true_signals, masked_map, obs_map):
        ups_y_true, ups_n_pred, ups_p_pred = true_signals[obs_map], n_pred[obs_map], p_pred[obs_map]
        imp_y_true, imp_n_pred, imp_p_pred = true_signals[masked_map], n_pred[masked_map], p_pred[masked_map]

        upsampling_loss = negative_binomial_loss(ups_y_true, ups_n_pred, ups_p_pred)
        imputation_loss = negative_binomial_loss(imp_y_true, imp_n_pred, imp_p_pred)
        
        if self.reduction == "mean":
            upsampling_loss = upsampling_loss.mean()
            imputation_loss = imputation_loss.mean()
        else:
            upsampling_loss = upsampling_loss.sum()
            imputation_loss = imputation_loss.sum()

        return self.alpha * imputation_loss, (1-self.alpha) * upsampling_loss

class ComboLoss_NBNLL_msk(nn.Module):
    def __init__(self):
        super(ComboLoss_NBNLL_msk, self).__init__()
        self.reduction = 'sum'
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, p_pred, n_pred, pred_mask, obs_mask, true_signals, masked_map, obs_map):
        ups_y_true, ups_n_pred, ups_p_pred = true_signals[obs_map], n_pred[obs_map], p_pred[obs_map]
        imp_y_true, imp_n_pred, imp_p_pred = true_signals[masked_map], n_pred[masked_map], p_pred[masked_map]

        upsampling_loss = negative_binomial_loss(ups_y_true, ups_n_pred, ups_p_pred)
        imputation_loss = negative_binomial_loss(imp_y_true, imp_n_pred, imp_p_pred)

        if self.reduction == "mean":
            upsampling_loss = upsampling_loss.mean()
            imputation_loss = imputation_loss.mean()
        else:
            upsampling_loss = upsampling_loss.sum()
            imputation_loss = imputation_loss.sum()

        bce_mask_prd_loss = self.bce_loss(pred_mask, masked_map.float())
        bce_mask_obs_loss = self.bce_loss(obs_mask, obs_map.float())

        return imputation_loss, upsampling_loss, bce_mask_prd_loss, bce_mask_obs_loss

class MatrixFactor_NBLL(nn.Module):
    def __init__(self):
        super(MatrixFactor_NBLL, self).__init__()
        self.reduction = 'sum'

    def forward(self, p_pred, n_pred, true_signals, obs_map):
        ups_y_true, ups_n_pred, ups_p_pred = true_signals[obs_map], n_pred[obs_map], p_pred[obs_map]
        upsampling_loss = negative_binomial_loss(ups_y_true, ups_n_pred, ups_p_pred)
        
        if self.reduction == "mean":
            upsampling_loss = upsampling_loss.mean()
        else:
            upsampling_loss = upsampling_loss.sum()

        return upsampling_loss

#========================================================================================================#
#====================================== EpiDenoise Versions =============================================#
#========================================================================================================#

class EpiDenoise10(nn.Module): 
    def __init__(self, input_dim, nhead, d_model, nlayers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise10, self).__init__()
        
        self.masked_linear = MaskedLinear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding10(d_model=d_model, max_len=context_length) 

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(d_model, output_dim)
        
    def forward(self, src, pmask, fmask):
        src = self.masked_linear(src, fmask)
        src = torch.permute(src, (1, 0, 2)) # to L, N, F
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, src_key_padding_mask=pmask) 
        src = self.decoder(src)
        src = torch.permute(src, (1, 0, 2))
        return src

class EpiDenoise15(nn.Module):
    """
    updates since EpiDenoise1.0:
        1. add CLS and SEP tokens
        2. segment adjacency prediction (SAP):
            - add segment encodings
            - custom loss function (masking + segment adjacency)
        3. dynamic masking chunks (gradually increasing)
    """

    def __init__(self, input_dim, nhead, d_model, nlayers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise15, self).__init__()
        
        self.masked_linear = MaskedLinear(input_dim, d_model)
        self.embeddings = ComboEmbedding15(d_model=d_model, seq_len=context_length+3, dropout=dropout) # segment + positional

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(d_model, output_dim)

        self.SAP = nn.Linear(d_model, 2) # segment adjacency prediction head
        self.softmax = torch.nn.Softmax(dim=-1)

class EpiDenoise16(nn.Module):
    """
    VIME
    gets masked_x as input
    loss: masked value prediction + observation reconstruction + mask reconstruction + segment adjacency prediction
    returns:
        - reconstructed input
        - reconstructed mask
        - SAP
    """

    def __init__(self, input_dim, nhead, d_model, nlayers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise16, self).__init__()

        self.embedding_linear = nn.Linear(input_dim, d_model)
        self.embeddings = ComboEmbedding15(d_model=d_model, seq_len=context_length+3, dropout=dropout) # segment + positional

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.signal_decoder = nn.Linear(d_model, output_dim)
        self.mask_decoder = nn.Linear(d_model, output_dim)

        self.SAP = nn.Linear(d_model, 2) # segment adjacency prediction head

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, src, segment_label):
        src = self.embedding_linear(src)

        src = torch.permute(src, (1, 0, 2)) # to L, N, F
        src = self.embeddings(src, segment_label)
        src = self.transformer_encoder(src) 

        cls_token = src[0, :, :].unsqueeze(0)
        SAP = self.softmax(self.SAP(cls_token))
        
        msk = torch.sigmoid(self.mask_decoder(src))
        src = self.signal_decoder(src)

        src = torch.permute(src, (1, 0, 2))  # to N, L, F
        msk = torch.permute(msk, (1, 0, 2))  # to N, L, F

        return src, msk, SAP   

class EpiDenoise17(nn.Module):
    """
    SAITS:
    concatenates x and mask from assays dimension:
        (N, L, A) -> (N, L, 2A)

    LOSS: observed reconstruction + masked value prediction + segment adjacency prediction
    returns:
        - reconstructed input
        - SAP

    """

    def __init__(self, input_dim, nhead, d_model, nlayers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise17, self).__init__()

        self.embedding_linear = nn.Linear(2*input_dim, d_model)
        self.embeddings = ComboEmbedding15(d_model=d_model, seq_len=context_length+3, dropout=dropout) # segment + positional

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.decoder = nn.Linear(d_model, output_dim)
        self.SAP = nn.Linear(d_model, 2) # segment adjacency prediction head
        self.softmax = torch.nn.Softmax(dim=-1)
    

    def forward(self, src, mask, segment_label):
        src = torch.cat([src, mask.float()], dim=2)
        src = self.embedding_linear(src)

        src = torch.permute(src, (1, 0, 2)) # to L, N, F
        src = self.embeddings(src, segment_label)
        src = self.transformer_encoder(src) 

        cls_token = src[0, :, :].unsqueeze(0)
        SAP = self.softmax(self.SAP(cls_token))

        src = self.decoder(src)
        src = torch.permute(src, (1, 0, 2))  # to N, L, F

        return src, SAP  

class EpiDenoise18(nn.Module):
    """
    no SAP
    added relative position encodings
    """
    def __init__(self, input_dim, nhead, d_model, nlayers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise18, self).__init__()

        # self.mf_embedding = MatrixFactorizationEmbedding(l=context_length, d=input_dim, k=d_model)
        self.embedding_linear = nn.Linear(input_dim, d_model)

        self.encoder_layer = RelativeEncoderLayer(d_model=d_model, heads=nhead, feed_forward_hidden=4*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.signal_decoder =  nn.Linear(d_model, output_dim)
        self.signal_softplus = nn.Softplus()
        # self.signal_decoder = FeedForwardNN(d_model, 4*d_model, output_dim, 2)
        self.mask_decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src = self.mf_embedding(src, linear=True)
        src = self.embedding_linear(src)

        src = torch.permute(src, (1, 0, 2)) # to L, N, F
        
        src = self.transformer_encoder(src) 
        
        msk = torch.sigmoid(self.mask_decoder(src))
        src = self.signal_softplus(self.signal_decoder(src))

        src = torch.permute(src, (1, 0, 2))  # to N, L, F
        msk = torch.permute(msk, (1, 0, 2))  # to N, L, F

        return src, msk

class EpiDenoise20(nn.Module):
    def __init__(self, 
                 input_dim, conv_out_channels, conv_kernel_sizes, dilation,
                 nhead, d_model, n_encoder_layers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise20, self).__init__()

        stride = 1
        n_cnn_layers = len(conv_out_channels)

        # Convolutional layers
        self.conv1 = ConvTower(
            input_dim, conv_out_channels[0], 
            1, stride, dilation, pool_type="None", residuals=False)

        self.convm = ConvTower(
            input_dim, conv_out_channels[0], 
            1, stride, dilation, pool_type="None", residuals=False)

        self.convtower = nn.Sequential(*[
            ConvTower(
                conv_out_channels[i], 
                conv_out_channels[i + 1],
                conv_kernel_sizes[i + 1], stride, dilation
            ) for i in range(n_cnn_layers - 1)
        ])

        # self.position = AbsPositionalEmbedding15(d_model=d_model, max_len=context_length)
        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=dropout)
        # self.transformer_encoder = nn.TransformerEncoder(
        #     self.encoder_layer, num_layers=n_encoder_layers)

        self.encoder_layer = RelativeEncoderLayer(
            d_model=d_model, heads=nhead, feed_forward_hidden=2*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_encoder_layers)

        # Deconvolution layers
        reversed_channels = list(reversed(conv_out_channels))
        reversed_kernels = list(reversed(conv_kernel_sizes))

        self.deconvtower = nn.Sequential(*[
            DeconvBlock(
                reversed_channels[i], reversed_channels[i + 1], 
                reversed_kernels[i + 1], 2, dilation) for i in range(n_cnn_layers - 2)
        ])
        self.deconvF = DeconvBlock(reversed_channels[-2], output_dim, reversed_kernels[-1], 2, dilation)

        self.signal_decoder = nn.Linear(output_dim, output_dim)
        self.mask_decoder = nn.Linear(output_dim, output_dim)

    def forward(self, x, m):
        x = x.permute(0, 2, 1) # to N, F, L
        m = m.permute(0, 2, 1) # to N, F, L

        m = self.convm(m.float())
        x = self.conv1(x)

        x = x + m

        x = self.convtower(x)

        x = x.permute(0, 2, 1)  # to N, L, F

        # x = x + self.position(x)
        x = self.transformer_encoder(x)

        x = x.permute(0, 2, 1) # to N, F, L'
        x = self.deconvtower(x)
        x = self.deconvF(x)
        x = x.permute(2, 0, 1)  # to L, N, F

        mask = torch.sigmoid(self.mask_decoder(x))
        x = self.signal_decoder(x)

        x = torch.permute(x, (1, 0, 2))  # to N, L, F
        mask = torch.permute(mask, (1, 0, 2))  # to N, L, F

        return x, mask

class EpiDenoise21(nn.Module):
    def __init__(self, 
                input_dim, conv_out_channels, conv_kernel_sizes, dilation, nhead, 
                d_model, n_encoder_layers, output_dim, dropout=0.1, context_length=2000):
        super(EpiDenoise21, self).__init__()

        stride = 1
        n_cnn_layers = len(conv_out_channels)

        # Convolutional layers
        self.conv1 = ConvTower(
            input_dim, conv_out_channels[0], 
            1, stride, dilation, pool_type="None", residuals=False)

        self.convm = ConvTower(
            input_dim, conv_out_channels[0], 
            1, stride, dilation, pool_type="None", residuals=False)

        self.convtower = nn.Sequential(*[
            ConvTower(
                conv_out_channels[i], 
                conv_out_channels[i + 1],
                conv_kernel_sizes[i + 1], stride, dilation
            ) for i in range(n_cnn_layers - 1)
        ])

        self.encoder_layer = RelativeEncoderLayer(
            d_model=d_model, heads=nhead, feed_forward_hidden=2*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_encoder_layers)

        # Convolutional layers
        self.d_conv1 = ConvTower(
            input_dim, d_model, 
            1, stride, dilation, pool_type="None", residuals=False)

        self.d_convm = ConvTower(
            input_dim, d_model, 
            1, stride, dilation, pool_type="None", residuals=False)

        self.relative_decoder = RelativeDecoderLayer(
            hid_dim=d_model, 
            n_heads=nhead, 
            pf_dim=2*d_model, 
            dropout=dropout
        )

        self.linear_output = nn.Linear(d_model, output_dim)

    def forward(self, src, src_missing_mask, trg, trg_missing_mask, trg_mask):
        src_missing_mask = src_missing_mask.permute(0, 2, 1) # to N, F, L
        src_missing_mask = ~src_missing_mask
        src_missing_mask = self.convm(src_missing_mask.float())

        src = src.permute(0, 2, 1) # to N, F, L
        src = self.conv1(src)

        src = src + src_missing_mask        
        src = self.convtower(src)

        src = src.permute(0, 2, 1)  # to N, L, F
        src = self.transformer_encoder(src)
        
        trg = trg.permute(0, 2, 1) # to N, F, L
        trg = self.d_conv1(trg) 

        trg_missing_mask = trg_missing_mask.permute(0, 2, 1) # to N, F, L
        trg_missing_mask = ~trg_missing_mask
        trg_missing_mask = self.d_convm(trg_missing_mask.float())

        trg = trg + trg_missing_mask  
        trg = trg.permute(0, 2, 1)  # to N, L, F

        # Apply the relative decoder
        trg = self.relative_decoder(trg, src, trg_mask)

        # Apply the final linear layers
        trg = self.linear_output(trg)

        return trg

class EpiDenoise22(nn.Module):
    def __init__(
        self, input_dim, conv_out_channels, conv_kernel_sizes, nhead, 
        d_model, n_encoder_layers, n_decoder_layers, output_dim, dilation=1, 
        dropout=0.1, context_length=2000, aggr=False):

        super(EpiDenoise22, self).__init__()

        stride = 1
        n_cnn_layers = len(conv_out_channels)
        self.aggr=aggr

        self.dual_conv_emb_src = DualConvEmbedding(in_C=input_dim, out_C=conv_out_channels[0])

        self.convtower = nn.ModuleList([ConvTower(
                conv_out_channels[i], conv_out_channels[i + 1],
                conv_kernel_sizes[i + 1], stride, dilation, 
                pool_type="attn", residuals=True
            ) for i in range(n_cnn_layers - 1)])

        if self.aggr:
            self.aggr_token = nn.Parameter(torch.randn(1, 1, d_model))
            # Additional linear layers for predicting mean and standard deviation
            self.mean_prediction = nn.Linear(d_model, output_dim)
            self.stddev_prediction = nn.Linear(d_model, output_dim)


        self.transformer_encoder = nn.ModuleList([RelativeEncoderLayer(
                d_model=d_model, heads=nhead, 
                feed_forward_hidden=2*d_model, 
                dropout=dropout) for _ in range(n_encoder_layers)])

        self.dual_conv_emb_trg = DualConvEmbedding(in_C=input_dim, out_C=d_model)

        self.transformer_decoder = nn.ModuleList([RelativeDecoderLayer(
            hid_dim=d_model, n_heads=nhead, 
            pf_dim=2*d_model, dropout=dropout) for _ in range(n_decoder_layers)])
    
        self.linear_output = nn.Linear(d_model, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, seq, mask):
        mask = mask.permute(0, 2, 1) # to N, F, L
        mask = mask.float()
        src = seq.permute(0, 2, 1) # to N, F, L
        trg = seq.permute(0, 2, 1) # to N, F, L

        src = self.dual_conv_emb_src(src, mask)
        trg = self.dual_conv_emb_trg(trg, mask)

        for conv in self.convtower:
            src = conv(src)

        src = src.permute(0, 2, 1)  # to N, L, F
        if self.aggr:
            # Concatenate special token to src
            batch_size, seq_len, _ = src.shape
            aggr_token = self.aggr_token.repeat(batch_size, 1, 1)
            src = torch.cat([aggr_token, src], dim=1)
            # print("agg+src", src.shape)

        for enc in self.transformer_encoder:
            src = enc(src)
            # print("src", src.shape)
        
        # print(src.shape)
        if self.aggr:
            aggr_token = src[:, 0, :]
            src = src[:, 1:, :]

            aggrmean = self.softplus(self.mean_prediction(aggr_token))
            aggrstddev = self.softplus(self.stddev_prediction(aggr_token))

        # print("trg",trg.shape)
        trg = trg.permute(0, 2, 1)  # to N, L, F
        for dec in self.transformer_decoder:
            trg = dec(trg, src, pad)
            # print("trg", trg.shape)
        
        trg = self.linear_output(trg)
        trg = self.softplus(trg)

        if self.aggr:
            return trg, aggrmean, aggrstddev
        else:
            return trg

class EpiDenoise30a(nn.Module):
    def __init__(
        self, input_dim, metadata_embedding_dim, nhead, d_model, nlayers, output_dim, 
        dropout=0.1, context_length=2000, pos_enc="relative"):
        super(EpiDenoise30a, self).__init__()

        self.pos_enc = "abs"
        self.context_length = context_length

        self.signal_layer_norm = nn.LayerNorm(input_dim)
        
        self.metadata_embedder = MetadataEmbeddingModule(input_dim, embedding_dim=metadata_embedding_dim)
        # self.embedding_linear = nn.Linear(input_dim + metadata_embedding_dim, d_model)

        self.ConvEmb = ConvTower(input_dim, d_model - metadata_embedding_dim,
                W=1, S=1, D=1, 
                pool_type="none", residuals=False, 
                groups=input_dim)

        self.SE_block = SE_Block_1D(d_model)

        if self.pos_enc == "relative":
            self.encoder_layer = RelativeEncoderLayer(
                d_model=d_model, heads=nhead, feed_forward_hidden=4*d_model, dropout=dropout)
        else:

            self.position = PositionalEncoding(d_model, dropout, context_length)

            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, 
                dropout=dropout, batch_first=True)
        
        self.transformer_encoder = nn.ModuleList(
            [self.encoder_layer for _ in range(nlayers)])

        self.neg_binom_layer = NegativeBinomialLayer(d_model, output_dim)
        self.mask_pred_layer = nn.Linear(d_model, output_dim)
        self.mask_obs_layer = nn.Linear(d_model, output_dim)
    
    def forward(self, src, x_metadata, y_metadata, availability):
        md_embedding = self.metadata_embedder(x_metadata, y_metadata, availability)
        # md_embedding = md_embedding.unsqueeze(1).expand(-1, self.context_length, -1)

        # md_embedding = F.relu(md_embedding)

        src = self.signal_layer_norm(src)

        src = src.permute(0, 2, 1) # to N, F1, L
        src = self.ConvEmb(src)

        src = torch.cat([src, md_embedding.unsqueeze(2).expand(-1, -1, self.context_length)], dim=1)
        src = self.SE_block(src)

        src = src.permute(0, 2, 1) # to N, L, F2

        if self.pos_enc != "relative":
            src = self.position(src)
        
        for enc in self.transformer_encoder:
            src = enc(src)

        p, n = self.neg_binom_layer(src)
        mp = torch.sigmoid(self.mask_pred_layer(src))
        mo = torch.sigmoid(self.mask_obs_layer(src))

        return p, n, mp, mo

class EpiDenoise30b(nn.Module):
    def __init__(self, 
        input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, n_decoder_layers, pool_size=2,
        dropout=0.1, context_length=2000, pos_enc="relative"):
        super(EpiDenoise30b, self).__init__()

        self.pos_enc = "abs" #pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = input_dim 
        self.f2 = (self.f1 * (2**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        d_model = self.f3 

        conv_channels = [(self.f1)*(2**l) for l in range(n_cnn_layers)]
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.metadata_embedder = MetadataEmbeddingModule(input_dim, embedding_dim=metadata_embedding_dim, non_linearity=True)
        self.signal_layer_norm = nn.LayerNorm(input_dim)
        
        self.convDec = ConvTower(self.f1, self.f2,
                W=1, S=1, D=1, 
                pool_type="none", residuals=False, 
                groups=self.f1)

        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else 2 * conv_channels[i],
                conv_kernel_size[i], S=1, D=1,
                pool_type="max", residuals=True,
                groups=conv_channels[i],
                pool_size=pool_size) for i in range(n_cnn_layers)])

        self.SE_enc = SE_Block_1D(self.f3)
        self.SE_dec = SE_Block_1D(self.f3)

        if self.pos_enc == "relative":
            self.encoder_layer = RelativeEncoderLayer(
                d_model=d_model, heads=nhead, feed_forward_hidden=4*d_model, dropout=dropout)

            self.decoder_layer = RelativeDecoderLayer(
                hid_dim=d_model, n_heads=nhead, pf_dim=4*d_model, dropout=dropout)
        else:
            self.posEnc = PositionalEncoding(d_model, dropout, self.l2)
            self.posDec = PositionalEncoding(d_model, dropout, self.l1)

            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=dropout, batch_first=True)

            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=2*d_model, dropout=dropout, batch_first=True)

        self.transformer_encoder = nn.ModuleList(
            [self.encoder_layer for _ in range(nlayers)])

        self.transformer_decoder = nn.ModuleList(
            [self.decoder_layer for _ in range(n_decoder_layers)])
        
        # self.f3 = d_model + metadata_embedding_dim
        self.neg_binom_layer = NegativeBinomialLayer(self.f3, output_dim)
        self.mask_pred_layer = nn.Linear(self.f3, output_dim)
        self.mask_obs_layer = nn.Linear(self.f3, output_dim)
    
    def forward(self, src, x_metadata, y_metadata, availability):
        md_embedding = self.metadata_embedder(x_metadata, y_metadata, availability)

        src = self.signal_layer_norm(src)
        ### CONV ENCODER ###
        e_src = src.permute(0, 2, 1) # to N, F1, L
        for conv in self.convEnc:
            e_src = conv(e_src)
        # e_src.shape = N, F2, L'
        e_src = torch.cat([e_src, md_embedding.unsqueeze(2).expand(-1, -1, self.l2)], dim=1)
        e_src = self.SE_enc(e_src)

        e_src = e_src.permute(0, 2, 1)  # to N, L', F2
        ### TRANSFORMER ENCODER ###
        if self.pos_enc != "relative":
            e_src = self.posEnc(e_src)
        for enc in self.transformer_encoder:
            e_src = enc(e_src)

        ### Conv DECODER ###
        src = src.permute(0, 2, 1) # to N, F1, L
        src = self.convDec(src)

        src = torch.cat([src, md_embedding.unsqueeze(2).expand(-1, -1, self.l1)], dim=1)
        src = self.SE_dec(src)

        src = src.permute(0, 2, 1) # to N, L, F2
        ### TRANSFORMER DECODER ###
        if self.pos_enc != "relative":
            src = self.posDec(src)
        for dec in self.transformer_decoder:
            src = dec(src, e_src)
        # src.shape = N, L, F2

        p, n = self.neg_binom_layer(src)
        mp = torch.sigmoid(self.mask_pred_layer(src))
        mo = torch.sigmoid(self.mask_obs_layer(src))

        return p, n, mp, mo

class EpiDenoise30c(nn.Module):
    def __init__(self, 
        input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, pool_size,
        dropout=0.1, context_length=2000):
        super(EpiDenoise30c, self).__init__()

        self.pos_enc = "abs" #pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = input_dim
        self.f2 = self.f1 * (2**(n_cnn_layers))
        assert d_model == self.f2, "mismatch in dimensions -- f2 != d_model"

        conv_channels = [(self.f1)*(2**l) for l in range(n_cnn_layers)]
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.metadata_embedder = MetadataEmbeddingModule(input_dim, embedding_dim=metadata_embedding_dim, non_linearity=True)
        self.signal_layer_norm = nn.LayerNorm(input_dim)
        self.position = PositionalEncoding(d_model, dropout, self.l1)
        
        self.convL = ConvTower(self.f1, self.f2,
                W=1, S=1, D=1, 
                pool_type="none", residuals=False, 
                groups=self.f1)

        self.convD = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else 2 * conv_channels[i],
                conv_kernel_size[i], S=1, D=1,
                pool_type="max", residuals=True,
                groups=conv_channels[i],
                pool_size=pool_size) for i in range(n_cnn_layers)])

        self.SE_L = SE_Block_1D(self.f2)
        self.SE_D = SE_Block_1D(self.f2)
        
        self.transL = nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=self.f2, nhead=nhead, 
                dim_feedforward=2*self.f2, 
                dropout=dropout, batch_first=True) for _ in range(nlayers)]) # input (B, L, F) -> output (B, L, d_model)

        self.transD = nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=self.l2, nhead=1, 
                dim_feedforward=2*self.f2, 
                dropout=dropout, batch_first=True) for _ in range(nlayers)]) # input (B, F, L) -> output (B, d_model, L')
        
        self.neg_binom_layer = NegativeBinomialLayer(self.l2, output_dim)
        # self.neg_binom_layer = NegativeBinomialLayer(self.f2 + metadata_embedding_dim, output_dim)
    
    def forward(self, src, x_metadata, y_metadata, availability):
        md_embedding = self.metadata_embedder(x_metadata, y_metadata, availability)
        # md_embedding = md_embedding.unsqueeze(1).expand(-1, self.l1, -1)

        src = self.signal_layer_norm(src)

        # src = torch.cat([src, md_embedding], dim=-1)

        W = src.permute(0, 2, 1) # to B, F, L
        W = self.convL(W)
        W = torch.cat([W, md_embedding.unsqueeze(2).expand(-1, -1, self.l1)], dim=1)
        W = self.SE_L(W, recal=True)
        W = W.permute(0, 2, 1) # to B, L, F'

        # W = self.fusionL(W)
        if self.pos_enc != "relative":
            W = self.position(W) 
        for encL in self.transL:
            W = encL(W)
        # W.shape = B, L, F'
        
        H = src.permute(0, 2, 1) # to B, F, L
        for conv in self.convD:
            H = conv(H)
        H = torch.cat([H, md_embedding.unsqueeze(2).expand(-1, -1, self.l2)], dim=1)
        H = self.SE_D(H, recal=False)

        # H.shape =  N, F', L'
        for encD in self.transD:
            H = encD(H)
        # H.shape =  N, F', L'

        Z = torch.matmul(W, H)

        # Z = torch.cat([Z, md_embedding], dim=-1) 

        # Z.shape = B, L, L'+metadata_dim

        p, n = self.neg_binom_layer(Z)

        return p, n

class EpiDenoise30d(nn.Module):
    def __init__(self, 
        input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, pool_size=3,
        dropout=0.1, context_length=2000, pos_enc="relative"):
        super(EpiDenoise30d, self).__init__()

        self.pos_enc = "abs" #pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = input_dim 
        self.f2 = (self.f1 * (2**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        d_model = self.f2

        conv_channels = [(self.f1)*(2**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [2 * x for x in conv_channels[::-1]]
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.signal_layer_norm = nn.LayerNorm(input_dim)

        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else 2 * conv_channels[i],
                conv_kernel_size[i], S=1, D=1,
                pool_type="max", residuals=True,
                groups=self.f1,
                pool_size=pool_size) for i in range(n_cnn_layers)])

        self.SE_enc = SE_Block_1D(self.f2)

        self.xmd_emb = EmbedMetadata(input_dim, metadata_embedding_dim, non_linearity=True)
        self.xmd_fusionlin = nn.Linear(self.f3, self.f2)

        self.ymd_emb = EmbedMetadata(input_dim, metadata_embedding_dim, non_linearity=True)
        self.ymd_fusionlin = nn.Linear(self.f3, self.f2)


        if self.pos_enc == "relative":
            self.encoder_layer = RelativeEncoderLayer(
                d_model=d_model, heads=nhead, feed_forward_hidden=4*d_model, dropout=dropout)
            self.transformer_encoder = nn.ModuleList([self.encoder_layer for _ in range(nlayers)])
            
        else:
            self.posEnc = PositionalEncoding(d_model, dropout, self.l2)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.deconv = nn.ModuleList(
            [DeconvTower(
                reverse_conv_channels[i], reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / 2),
                conv_kernel_size[-(i + 1)], S=pool_size, D=1,
                pool_type="up", residuals=True,
                groups=1,
                pool_size=pool_size) for i in range(n_cnn_layers)])
        
        # self.f3 = d_model + metadata_embedding_dim
        self.neg_binom_layer = NegativeBinomialLayer(self.f1, output_dim)
        self.mask_pred_layer = nn.Linear(self.f1, output_dim)
        self.mask_obs_layer = nn.Linear(self.f1, output_dim)
    
    def forward(self, src, x_metadata, y_metadata, availability):
        src = torch.where(src == -2, torch.tensor(-1, device=src.device), src)
        x_metadata = torch.where(x_metadata == -2, torch.tensor(-1, device=x_metadata.device), x_metadata)
        y_metadata = torch.where(y_metadata == -2, torch.tensor(-1, device=y_metadata.device), y_metadata)
        availability = torch.where(availability == -2, torch.tensor(-1, device=availability.device), availability)

        # md_embedding = self.metadata_embedder(x_metadata, y_metadata, availability)

        src = self.signal_layer_norm(src)
        ### CONV ENCODER ###
        src = src.permute(0, 2, 1) # to N, F1, L
        for conv in self.convEnc:
            src = conv(src)
        # e_src.shape = N, F2, L'
        src = self.SE_enc(src)

        src = src.permute(0, 2, 1)  # to N, L', F2
        xmd_embedding = self.xmd_emb(x_metadata)
        src = torch.cat([src, xmd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.xmd_fusionlin(src)

        ### TRANSFORMER ENCODER ###
        if self.pos_enc != "relative":
            src = self.posEnc(src)
            src = self.transformer_encoder(src)
        else:
            for enc in self.transformer_encoder:
                src = enc(src)

        ymd_embedding = self.ymd_emb(y_metadata)
        src = torch.cat([src, ymd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.ymd_fusionlin(src)
        
        src = src.permute(0, 2, 1) # to N, F2, L'
        for dconv in self.deconv:
            src = dconv(src)

        src = src.permute(0, 2, 1) # to N, L, F1
        
        p, n = self.neg_binom_layer(src)
        mp = torch.sigmoid(self.mask_pred_layer(src))
        mo = torch.sigmoid(self.mask_obs_layer(src))

        return p, n, mp, mo

#========================================================================================================#
#=========================================Pretraining====================================================#
#========================================================================================================#

class PRE_TRAINER(object):  
    def __init__(self, model, dataset, criterion, optimizer, scheduler, eed=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        print(self.device)

        self.model = model.to(self.device)
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        if not eed:
            self.test_x_dir = "/project/compbio-lab/EIC/training_data/C23_chr21_25.pt"
            self.test_y_dir = "/project/compbio-lab/EIC/validation_data/C23_chr21_25.pt"
    
    def test_model(self, context_length, version, is_arcsin, batch_size):
        self.model.eval()
        """
        load X and Y
        pred = model(X)
        compare X and Y
        """
        missing_x_i = []
        missing_y_i = []

        X = torch.load(self.test_x_dir)
        # fill-in missing_ind
        for i in range(X.shape[1]):
            if (X[:, i] == -1).all():
                missing_x_i.append(i)

        Y = torch.load(self.test_y_dir)
        # fill-in missing_ind
        for i in range(Y.shape[1]):
            if (Y[:, i] == -1).all():
                missing_y_i.append(i)

        num_rows = (X.shape[0] // context_length) * context_length
        X, Y = X[:num_rows, :], Y[:num_rows, :]

        if is_arcsin:
            arcmask1 = (X != -1)
            X[arcmask1] = torch.arcsinh_(X[arcmask1])

            arcmask2 = (Y != -1)
            Y[arcmask2] = torch.arcsinh_(Y[arcmask2])

        X = X.view(-1, context_length, X.shape[-1])
        Y = Y.view(-1, context_length, Y.shape[-1])

        d_model = X.shape[-1]

        if version == "10":
            fmask = torch.ones(d_model, self.hyper_parameters["d_model"])
            for i in missing_x_i: # input fmask
                fmask[i,:] = 0
            fmask = fmask.to(self.device)

        elif version == "16" or version == "17":
            CLS_x = torch.full((X.shape[0], 1, X.shape[2]), -2)
            SEP_x = torch.full((X.shape[0], 1, X.shape[2]), -3)
            CLS_y = torch.full((Y.shape[0], 1, Y.shape[2]), -2)
            SEP_y = torch.full((Y.shape[0], 1, Y.shape[2]), -3)

            X = torch.cat([CLS_x, X[:, :context_length//2, :], SEP_x, X[:, context_length//2:, :], SEP_x], dim=1)
            Y = torch.cat([CLS_y, Y[:, :context_length//2, :], SEP_y, Y[:, context_length//2:, :], SEP_y], dim=1)

            segment_label = [0] + [1 for i in range(context_length//2)] + [0] + [2 for i in range(context_length//2)] + [0]
            segment_label = torch.from_numpy(np.array(segment_label))
            segment_label = segment_label.to(self.device)

        # Initialize a tensor to store all predictions
        P = torch.empty_like(X, device="cpu")

        # make predictions in batches
        for i in range(0, len(X), batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+batch_size]

            with torch.no_grad():
                x_batch = x_batch.to(self.device)
                mask = torch.zeros_like(x_batch, dtype=torch.bool, device=self.device)
                for ii in missing_x_i: 
                    mask[:,:,ii] = True
                mask = mask.to(self.device)

                if version == "10":
                    # (no position is masked)
                    pmask = torch.zeros((x_batch.shape[0], x_batch.shape[1]), dtype=torch.bool,  device=self.device)
                    outputs = self.model(x_batch, pmask, fmask)

                elif version == "16":
                    outputs, pred_mask, SAP = self.model(x_batch, segment_label)

                elif version == "17":
                    outputs, SAP = self.model(x_batch, ~mask, segment_label)
                
                elif version == "18":
                    outputs, pred_mask = self.model(x_batch)

                elif version == "20":

                    outputs, pred_mask = self.model(x_batch, mask)
                
                elif version == "21":
                    outputs, pred_mask = self.model(x_batch, mask)

            # Store the predictions in the large tensor
            P[i:i+outputs.shape[0], :, :] = outputs.cpu()
        
        P = P.view((P.shape[0] * P.shape[1]), P.shape[-1]) # preds
        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1]) # eval data
        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1]) # train data

        mses = []
        spearmans = []
        peak_overlaps = []
        for j in range(Y.shape[-1]):  # for each feature i.e. assay
            pred = P[:, j].numpy()
            metrics_list = []

            if j in missing_x_i and j not in missing_y_i:  # if the feature is missing in the input
                target = Y[:, j].numpy()   
                mse_GW = np.mean((np.array(target) - np.array(pred))**2)
                spearman_GW = spearmanr(pred, target)[0]
                ovr = peak_overlap(target, pred, p=0.05)

                mses.append(mse_GW)
                spearmans.append(spearman_GW)
                peak_overlaps.append(ovr)

        self.model.train()

        if version in ["10", "16", "17"]:
            return sum(mses)/len(mses), sum(spearmans)/len(spearmans)
        else:
            return mses, spearmans, peak_overlaps

    def test_autoregressive_model(self, context_length, is_arcsin, step_size, p=0.01):
        self.model.eval()

        missing_x_i = []
        missing_y_i = []

        X = torch.load(self.test_x_dir)
        # fill-in missing_ind
        for i in range(X.shape[1]):
            if (X[:, i] == -1).all():
                missing_x_i.append(i)

        Y = torch.load(self.test_y_dir)
        # fill-in missing_ind
        for i in range(Y.shape[1]):
            if (Y[:, i] == -1).all():
                missing_y_i.append(i)

        L, num_features = X.shape

        subset_size = int(L * p)  # The total number of rows to include in the subset
        start_index = (L - subset_size) // 2  # Start index for the middle subset
        end_index = start_index + subset_size  # End index for the middle subset

        start_index -= start_index % step_size
        end_index -= end_index % step_size
        
        # Slice X and Y to get the middle subset
        X = X[start_index:end_index, :]
        Y = Y[start_index:end_index, :]
        
        if is_arcsin:
            arcmask1 = (X != -1)
            X[arcmask1] = torch.arcsinh_(X[arcmask1])

            arcmask2 = (Y != -1)
            Y[arcmask2] = torch.arcsinh_(Y[arcmask2])
        
        mask = torch.zeros_like(X, dtype=torch.bool)
        for ii in missing_x_i: 
            mask[:,ii] = True

        # ============================================ #
        #               SENSE PREDICTION
        # ============================================ #
        src_context = []
        trg_context = []

        missing_mask = []

        for i in range(0, X.shape[0] - context_length, step_size):
            if i+context_length+step_size > X.shape[0]:
                break

            src_context.append(X[i : i+context_length, :].numpy())
            trg_context.append(X[i+step_size : i+context_length+step_size, :].numpy())

            missing_mask.append(mask[i : i+context_length, :].numpy())

        src_context = torch.from_numpy(np.array(src_context)).to(self.device)
        trg_context = torch.from_numpy(np.array(trg_context)).to(self.device)

        missing_mask = torch.from_numpy(np.array(missing_mask)).to(self.device)

        trg_msk = torch.zeros((trg_context.shape[0], trg_context.shape[1]), dtype=torch.bool, device=self.device)
        trg_msk[:, -step_size:] = True

        with torch.no_grad():
            fw_outputs = self.model(
                src_context, missing_mask, trg_context, missing_mask, trg_msk) 
            
            fw_outputs = fw_outputs[:,-step_size:, :]
        
        fw_outputs = fw_outputs.reshape(fw_outputs.shape[0]*fw_outputs.shape[1], fw_outputs.shape[2])
        
        # ============================================ #
        #           ANTI-SENSE PREDICTION
        # ============================================ #
        X = torch.flip(X, dims=(0,))
        src_context = []
        trg_context = []
        missing_mask = []

        start_i = X.shape[0] - (2 * context_length) - 1

        for i in range(start_i, X.shape[0] - context_length, step_size):
            if i+context_length+step_size > X.shape[0]:
                break

            src_context.append(X[i : i+context_length, :].numpy())
            trg_context.append(X[i+step_size : i+context_length+step_size, :].numpy())

            missing_mask.append(mask[i : i+context_length, :].numpy())

        src_context = torch.from_numpy(np.array(src_context)).to(self.device)
        trg_context = torch.from_numpy(np.array(trg_context)).to(self.device)

        missing_mask = torch.from_numpy(np.array(missing_mask)).to(self.device)

        trg_msk = torch.zeros((trg_context.shape[0], trg_context.shape[1]), dtype=torch.bool, device=self.device)
        trg_msk[:, -step_size:] = True

        with torch.no_grad():
            bw_outputs = self.model(
                src_context, missing_mask, trg_context, missing_mask, trg_msk) 
            
            bw_outputs = bw_outputs[:,-step_size:, :]
        
        bw_outputs = bw_outputs.reshape(bw_outputs.shape[0]*bw_outputs.shape[1], bw_outputs.shape[2])
        bw_outputs = torch.flip(bw_outputs, dims=(0,))

        P = torch.cat((bw_outputs, fw_outputs), dim=0).cpu()

        del src_context, trg_context, missing_mask, trg_msk, bw_outputs, fw_outputs

        mses = []
        spearmans = []
        peak_overlaps = []
        for j in range(Y.shape[-1]):  # for each feature i.e. assay
            pred = P[:, j].numpy()
            metrics_list = []

            if j in missing_x_i and j not in missing_y_i:  # if the feature is missing in the input
                target = Y[:, j].numpy()   
                mse_GW = np.mean((np.array(target) - np.array(pred))**2)
                spearman_GW = spearmanr(pred, target)[0]
                ovr = peak_overlap(target, pred, p=0.05)

                mses.append(mse_GW)
                spearmans.append(spearman_GW)
                peak_overlaps.append(ovr)

        self.model.train()
        return mses, spearmans, peak_overlaps

    def pretrain_epidenoise_10(self, 
        d_model, outer_loop_epochs=1, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, 
        context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        for ole in range(outer_loop_epochs):
            ds=0

            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                num_features = x.shape[2]

                if arcsinh_transform:
                    x = torch.arcsinh_(x)
                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
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

                        fmask = fmask.to(self.device)

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
                            pmask = pmask.to(self.device)

                            cloze_mask = cloze_mask & ~missing_mask_batch
                            x_batch = x_batch.to(self.device)

                            masked_x_batch = masked_x_batch.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs = self.model(masked_x_batch, pmask, fmask)
                            loss = self.criterion(outputs[cloze_mask], x_batch[cloze_mask])

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
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
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
                    self.optimizer.step()
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/log.txt", "w")

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
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
                    torch.save(self.model.state_dict(), f'models/model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def pretrain_epidenoise_15(self, 
        d_model, outer_loop_epochs=6, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, 
        context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        for ole in range(outer_loop_epochs):
            ds=0

            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                num_features = x.shape[2]

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])
                                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
                    epoch_loss = []
                    t0 = datetime.now()

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1

                        pattern_batch = x[indices]
                        missing_mask_patten_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]

                        # print(pattern_batch.shape, (fmask.sum(dim=1) > 0).sum().item(), len(pattern))

                        if context_length < pattern_batch.shape[1]:
                            context_length_factor = context_length / pattern_batch.shape[1]

                            pattern_batch = reshape_tensor(pattern_batch, context_length_factor)
                            missing_mask_patten_batch = reshape_tensor(missing_mask_patten_batch, context_length_factor)

                        # Break down x into smaller batches
                        for i in range(0, len(pattern_batch), batch_size):

                            torch.cuda.empty_cache()
                            seg_length = context_length // 2
                            is_adjacent = random.choice([True, False])

                            seg_1 = pattern_batch[i:i+batch_size, :seg_length, :]
                            seg1m = missing_mask_patten_batch[i:i+batch_size, :seg_length, :]
                            
                            if is_adjacent:
                                seg_2 = pattern_batch[i:i+batch_size, seg_length:, :]
                                seg2m = missing_mask_patten_batch[i:i+batch_size, seg_length:, :]
                                
                            else:
                                seg_1.shape[0]
                                # Randomly select a start index
                                start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                # If the start index overlaps with the range i:i+batch_size, choose again
                                while i <= start < i + seg_1.shape[0]:
                                    start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                seg_2 = pattern_batch[start:start+seg_1.shape[0], :seg_length, :]
                                seg2m = missing_mask_patten_batch[start:start+seg_1.shape[0], :seg_length, :]

                            CLS = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -3)
                            SEP = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -4)
                            
                            x_batch = torch.cat((CLS, seg_1, SEP, seg_2, SEP), 1)
                            missing_mask_batch = torch.cat((seg1m[:,0,:].unsqueeze(1), seg1m, seg1m[:,0,:].unsqueeze(1), seg2m, seg2m[:,0,:].unsqueeze(1)), 1)

                            special_token_indices = [0, seg_length, (2*seg_length)+1]

                            # 0 are for special tokens, 1 for segment1 and 2 for segment2
                            segment_label = [0] + [1 for i in range(seg_1.shape[1])] + [0] + [2 for i in range(seg_2.shape[1])] + [0]
                            segment_label = torch.from_numpy(np.array(segment_label))
                            segment_label = segment_label.to(self.device)

                            # create segment adjacency prediction labels based on is_adjacent
                            target_SAP = torch.full((1, x_batch.shape[0], 2), float(0))
                            target_SAP[:,:,int(is_adjacent)] = 1 
                            target_SAP = target_SAP.to(self.device)

                            # Masking a subset of the input data -- genomic position mask
                            masked_x_batch, cloze_mask = mask_data15(x_batch, mask_value=-1, chunk=chunk, n_chunks=n_chunks, mask_percentage=mask_percentage)
                            
                            pmask = cloze_mask[:,:,0].squeeze()
                            
                            cloze_mask = cloze_mask & ~missing_mask_batch

                            """
                            if num_available features > 1, 
                                in each batch, randomly mask one of the available features
                                update the fmask
                                get the model to predict the whole track based on input
                            """

                            fmask = torch.ones(num_features, d_model)

                            for i in pattern:
                                fmask[i,:] = 0

                            if len(available_assays_ind) > 1:
                                assaymask_ind = random.choice(available_assays_ind)
                                masked_x_batch[:,:,available_assays_ind] = -1
                                fmask[assaymask_ind, :] = 0
                                cloze_mask[:, :, available_assays_ind] = True
                                cloze_mask[:, special_token_indices, :] = False

                            # move to GPU
                            fmask = fmask.to(self.device)
                            pmask = pmask.to(self.device)
                            x_batch = x_batch.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs, SAP = self.model(masked_x_batch, pmask, fmask, segment_label)

                            loss = self.criterion(outputs[cloze_mask], x_batch[cloze_mask], SAP, target_SAP)

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
                            
                            mean_pred, std_pred = outputs[cloze_mask].mean().item(), outputs[cloze_mask].std().item()
                            mean_target, std_target = x_batch[cloze_mask].mean().item(), x_batch[cloze_mask].std().item()

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
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
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
                    self.optimizer.step()
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/log.txt", "w")

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Epoch Loss Mean: {np.mean(epoch_loss):.4f}", 
                        f"Epoch Loss std: {np.std(epoch_loss):.4f}",
                        f"Epoch took: {t1 - t0}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    torch.save(self.model.state_dict(), f'models/model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def pretrain_epidenoise_16(self, 
        d_model, outer_loop_epochs=3, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, 
        context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/EPD16_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        for ole in range(outer_loop_epochs):
            ds=0

            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                num_features = x.shape[2]

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])
                                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
                    epoch_loss = []
                    t0 = datetime.now()

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1

                        pattern_batch = x[indices]
                        missing_mask_patten_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]

                        # print(pattern_batch.shape, (fmask.sum(dim=1) > 0).sum().item(), len(pattern))

                        if context_length < pattern_batch.shape[1]:
                            context_length_factor = context_length / pattern_batch.shape[1]

                            pattern_batch = reshape_tensor(pattern_batch, context_length_factor)
                            missing_mask_patten_batch = reshape_tensor(missing_mask_patten_batch, context_length_factor)

                        # Break down x into smaller batches
                        for i in range(0, len(pattern_batch), batch_size):
                            self.optimizer.zero_grad()

                            torch.cuda.empty_cache()
                            seg_length = context_length // 2
                            is_adjacent = random.choice([True, False])

                            seg_1 = pattern_batch[i:i+batch_size, :seg_length, :]
                            seg1m = missing_mask_patten_batch[i:i+batch_size, :seg_length, :]
                            
                            if is_adjacent:
                                seg_2 = pattern_batch[i:i+batch_size, seg_length:, :]
                                seg2m = missing_mask_patten_batch[i:i+batch_size, seg_length:, :]
                                
                            else:
                                seg_1.shape[0]
                                # Randomly select a start index
                                start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                # If the start index overlaps with the range i:i+batch_size, choose again
                                while i <= start < i + seg_1.shape[0]:
                                    start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                seg_2 = pattern_batch[start:start+seg_1.shape[0], :seg_length, :]
                                seg2m = missing_mask_patten_batch[start:start+seg_1.shape[0], :seg_length, :]

                            CLS = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -2)
                            SEP = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -3)
                            
                            x_batch = torch.cat((CLS, seg_1, SEP, seg_2, SEP), 1)
                            missing_mask_batch = torch.cat((seg1m[:,0,:].unsqueeze(1), seg1m, seg1m[:,0,:].unsqueeze(1), seg2m, seg2m[:,0,:].unsqueeze(1)), 1)

                            special_token_indices = [0, seg_length, (2*seg_length)+1]

                            # 0 are for special tokens, 1 for segment1 and 2 for segment2
                            segment_label = [0] + [1 for i in range(seg_1.shape[1])] + [0] + [2 for i in range(seg_2.shape[1])] + [0]
                            segment_label = torch.from_numpy(np.array(segment_label))
                            segment_label = segment_label.to(self.device)

                            # create segment adjacency prediction labels based on is_adjacent
                            target_SAP = torch.full((1, x_batch.shape[0], 2), float(0))
                            target_SAP[:,:,int(is_adjacent)] = 1 
                            target_SAP = target_SAP.to(self.device)

                            # Masking a subset of the input data -- genomic position mask
                            masked_x_batch, cloze_mask = mask_data16(x_batch, available_assays_ind, mask_value=-1, mask_percentage=mask_percentage)
                            union_mask = cloze_mask | missing_mask_batch

                            # move to GPU
                            x_batch = x_batch.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            union_mask = union_mask.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs, pred_mask, SAP = self.model(masked_x_batch, segment_label)

                            #pred_signals, true_signals, pred_adjac, true_adjac, pred_mask, obs_mask
                            loss = self.criterion(outputs, x_batch, SAP, target_SAP, pred_mask, cloze_mask, union_mask)

                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                del x_batch
                                del masked_x_batch
                                del outputs
                                torch.cuda.empty_cache()
                                continue

                            del x_batch
                            del masked_x_batch
                            del outputs
                            epoch_loss.append(loss.item())

                            # Clear GPU memory again
                            torch.cuda.empty_cache()

                            loss.backward()  
                            self.optimizer.step()
                            # self.scheduler.step()
                        
                        if p%8 == 0:
                            logfile = open("models/EPD16_log.txt", "w")

                            logstr = [
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Epoch {epoch+1}/{num_epochs}', f'Missing Pattern {p}/{len(missing_f_pattern)}', 
                                f"Loss: {loss.item():.4f}", 
                                # f"Mean_P: {mean_pred:.3f}", f"Mean_T: {mean_target:.3f}", 
                                # f"Std_P: {std_pred:.2f}", f"Std_T: {std_target:.2f}"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)
                        
                    # update parameters over all batches and all patterns of missing data
                    # self.optimizer.step()
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/EPD16_log.txt", "w")
                    
                    test_mse, test_corr = self.test_model(
                        context_length, version="16", 
                        is_arcsin=arcsinh_transform, batch_size=batch_size)

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Epoch Loss Mean: {np.mean(epoch_loss):.4f}", 
                        f"Epoch Loss std: {np.std(epoch_loss):.4f}",
                        f"Test_MSE: {test_mse:.4f}",
                        f"Test Corr: {test_corr:.4f}",
                        f"Epoch took: {t1 - t0}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    if ds%5 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD16_model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model
    
    def pretrain_epidenoise_17(self, 
        d_model, outer_loop_epochs=3, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, 
        context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/EPD17_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        for ole in range(outer_loop_epochs):
            ds=0

            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                num_features = x.shape[2]

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])
                                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
                    epoch_loss = []
                    t0 = datetime.now()

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1

                        pattern_batch = x[indices]
                        missing_mask_patten_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]

                        # print(pattern_batch.shape, (fmask.sum(dim=1) > 0).sum().item(), len(pattern))

                        if context_length < pattern_batch.shape[1]:
                            context_length_factor = context_length / pattern_batch.shape[1]

                            pattern_batch = reshape_tensor(pattern_batch, context_length_factor)
                            missing_mask_patten_batch = reshape_tensor(missing_mask_patten_batch, context_length_factor)

                        # Break down x into smaller batches
                        for i in range(0, len(pattern_batch), batch_size):
                            self.optimizer.zero_grad()

                            torch.cuda.empty_cache()
                            seg_length = context_length // 2
                            is_adjacent = random.choice([True, False])

                            seg_1 = pattern_batch[i:i+batch_size, :seg_length, :]
                            seg1m = missing_mask_patten_batch[i:i+batch_size, :seg_length, :]
                            
                            if is_adjacent:
                                seg_2 = pattern_batch[i:i+batch_size, seg_length:, :]
                                seg2m = missing_mask_patten_batch[i:i+batch_size, seg_length:, :]
                                
                            else:
                                seg_1.shape[0]
                                # Randomly select a start index
                                start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                # If the start index overlaps with the range i:i+batch_size, choose again
                                while i <= start < i + seg_1.shape[0]:
                                    start = random.randint(0, len(pattern_batch) - batch_size)
                                
                                seg_2 = pattern_batch[start:start+seg_1.shape[0], :seg_length, :]
                                seg2m = missing_mask_patten_batch[start:start+seg_1.shape[0], :seg_length, :]

                            CLS = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -2)
                            SEP = torch.full((seg_1.shape[0], 1, seg_1.shape[2]), -3)
                            
                            x_batch = torch.cat((CLS, seg_1, SEP, seg_2, SEP), 1)
                            missing_mask_batch = torch.cat((seg1m[:,0,:].unsqueeze(1), seg1m, seg1m[:,0,:].unsqueeze(1), seg2m, seg2m[:,0,:].unsqueeze(1)), 1)

                            special_token_indices = [0, seg_length, (2*seg_length)+1]

                            # 0 are for special tokens, 1 for segment1 and 2 for segment2
                            segment_label = [0] + [1 for i in range(seg_1.shape[1])] + [0] + [2 for i in range(seg_2.shape[1])] + [0]
                            segment_label = torch.from_numpy(np.array(segment_label))
                            segment_label = segment_label.to(self.device)

                            # create segment adjacency prediction labels based on is_adjacent
                            target_SAP = torch.full((1, x_batch.shape[0], 2), float(0))
                            target_SAP[:,:,int(is_adjacent)] = 1 
                            target_SAP = target_SAP.to(self.device)

                            # Masking a subset of the input data -- genomic position mask
                            masked_x_batch, cloze_mask = mask_data16(x_batch, available_assays_ind, mask_value=-1, mask_percentage=mask_percentage)
                            union_mask = cloze_mask | missing_mask_batch

                            # move to GPU
                            x_batch = x_batch.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            union_mask = union_mask.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs, SAP = self.model(masked_x_batch, ~union_mask, segment_label)

                            loss = self.criterion(outputs, x_batch, SAP, target_SAP, cloze_mask, ~union_mask)

                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                del x_batch
                                del masked_x_batch
                                del outputs
                                torch.cuda.empty_cache()
                                continue

                            del x_batch
                            del masked_x_batch
                            del outputs
                            epoch_loss.append(loss.item())

                            # Clear GPU memory again
                            torch.cuda.empty_cache()

                            loss.backward()  
                            self.optimizer.step()
                        
                        if p%8 == 0:
                            logfile = open("models/EPD17_log.txt", "w")

                            logstr = [
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Epoch {epoch+1}/{num_epochs}', f'Missing Pattern {p}/{len(missing_f_pattern)}', 
                                f"Loss: {loss.item():.4f}", 
                                # f"Mean_P: {mean_pred:.3f}", f"Mean_T: {mean_target:.3f}", 
                                # f"Std_P: {std_pred:.2f}", f"Std_T: {std_target:.2f}"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)
                        
                    # update parameters over all batches and all patterns of missing data
                    # self.optimizer.step()
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/EPD17_log.txt", "w")

                    test_mse, test_corr = self.test_model(
                        context_length, version="17", 
                        is_arcsin=arcsinh_transform, batch_size=batch_size)

                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Epoch Loss Mean: {np.mean(epoch_loss):.4f}", 
                        f"Epoch Loss std: {np.std(epoch_loss):.4f}",
                        f"Test_MSE: {test_mse:.4f}",
                        f"Test Corr: {test_corr:.4f}",
                        f"Epoch took: {t1 - t0}"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    if ds%5 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD17_model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def pretrain_epidenoise_18(self, 
        d_model, outer_loop_epochs=2, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, chunk=False, n_chunks=1, 
        context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/EPD18_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        for ole in range(outer_loop_epochs):
            ds=0

            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                num_features = x.shape[2]

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])
                                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
                    epoch_loss = []
                    epoch_obs_loss = []
                    epoch_msk_loss = []
                    epoch_clz_loss = []
                    t0 = datetime.now()

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1

                        pattern_batch = x[indices]
                        missing_mask_patten_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]

                        # print(pattern_batch.shape, (fmask.sum(dim=1) > 0).sum().item(), len(pattern))

                        if context_length < pattern_batch.shape[1]:
                            context_length_factor = context_length / pattern_batch.shape[1]

                            pattern_batch = reshape_tensor(pattern_batch, context_length_factor)
                            missing_mask_patten_batch = reshape_tensor(missing_mask_patten_batch, context_length_factor)

                        # Break down x into smaller batches
                        for i in range(0, len(pattern_batch), batch_size):
                            self.optimizer.zero_grad()

                            torch.cuda.empty_cache()

                            x_batch = pattern_batch[i:i+batch_size]
                            missing_mask_batch = missing_mask_patten_batch[i:i+batch_size]
                            
                            # Masking a subset of the input data -- genomic position mask
                            masked_x_batch, cloze_mask = mask_data16(
                                x_batch, available_assays_ind, mask_value=-1, mask_percentage=mask_percentage)

                            union_mask = cloze_mask | missing_mask_batch

                            masked_x_batch = add_noise(masked_x_batch, 0.2)
                            masked_x_batch[union_mask] = -1

                            # move to GPU
                            x_batch = x_batch.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            union_mask = union_mask.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs, pred_mask = self.model(masked_x_batch)
                            mse_obs_loss, mse_pred_loss, bce_mask_loss = self.criterion(
                                outputs, x_batch, pred_mask, cloze_mask, union_mask)
                            loss = mse_obs_loss + mse_pred_loss + bce_mask_loss

                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                print(len(available_assays_ind), mse_obs_loss + mse_pred_loss + bce_mask_loss)
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                del x_batch
                                del masked_x_batch
                                del outputs
                                torch.cuda.empty_cache()
                                continue

                            del x_batch
                            del masked_x_batch
                            del outputs

                            epoch_loss.append(loss.item())
                            epoch_obs_loss.append(mse_obs_loss.item())
                            epoch_clz_loss.append(mse_pred_loss.item())
                            epoch_msk_loss.append(bce_mask_loss.item())

                            # Clear GPU memory again
                            torch.cuda.empty_cache()

                            loss.backward()  
                            self.optimizer.step()
                        
                        if p == 1 or p%8 == 0:
                            logfile = open("models/EPD18_log.txt", "w")

                            logstr = [
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Epoch {epoch+1}/{num_epochs}', f'Missing Pattern {p}/{len(missing_f_pattern)}', 
                                f"Obs Loss: {mse_obs_loss.item():.4f}",
                                f"Clz Loss: {mse_pred_loss.item():.4f}",
                                f"Msk Loss: {bce_mask_loss.item():.4f}"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)
                        
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/EPD18_log.txt", "w")
                    
                    test_mse, test_corr, test_ovr = self.test_model(
                        context_length, version="18", 
                        is_arcsin=arcsinh_transform, batch_size=batch_size)

                    test_mse = np.mean(test_mse)
                    test_corr = np.mean(test_corr)

                    test_ovr_mean = np.mean(test_ovr)
                    test_ovr_min = np.min(test_ovr)
                    test_ovr_max = np.max(test_ovr)

                    logstr = [
                        "\n----------------------------------------------------\n"
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Obs Loss: {np.mean(epoch_obs_loss):.3f}", 
                        f"Clz Loss: {np.mean(epoch_clz_loss):.3f}", 
                        f"Msk Loss: {np.mean(epoch_msk_loss):.3f}", 
                        f"Val_MSE: {test_mse:.4f}",
                        f"Val_POmean: {test_ovr_mean:.3f}",
                        f"Val_POmin: {test_ovr_min:.3f}",
                        f"Val_POmax: {test_ovr_max:.3f}",
                        f"Epoch took: {t1 - t0}"
                        "\n----------------------------------------------------\n"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    if ds%5 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD18_model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def pretrain_epidenoise_20(self, 
        d_model, outer_loop_epochs=3, arcsinh_transform=True,
        num_epochs=25, mask_percentage=0.15, context_length=2000, batch_size=100, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/EPD20_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        for ole in range(outer_loop_epochs):
            ds=0

            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                num_features = x.shape[2]

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])

                                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
                    epoch_loss = []
                    epoch_obs_loss = []
                    epoch_msk_loss = []
                    epoch_clz_loss = []
                    t0 = datetime.now()

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1

                        pattern_batch = x[indices]
                        missing_mask_patten_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]

                        # print(pattern_batch.shape, (fmask.sum(dim=1) > 0).sum().item(), len(pattern))

                        if context_length < pattern_batch.shape[1]:
                            context_length_factor = context_length / pattern_batch.shape[1]

                            pattern_batch = reshape_tensor(pattern_batch, context_length_factor)
                            missing_mask_patten_batch = reshape_tensor(missing_mask_patten_batch, context_length_factor)

                        # Break down x into smaller batches
                        for i in range(0, len(pattern_batch), batch_size):
                            self.optimizer.zero_grad()

                            torch.cuda.empty_cache()

                            x_batch = pattern_batch[i:i+batch_size]
                            missing_mask_batch = missing_mask_patten_batch[i:i+batch_size]
                            
                            # if len(available_assays_ind) == 1:
                            masked_x_batch, cloze_mask = mask_data16(
                                x_batch, available_assays_ind, mask_value=-1, mask_percentage=mask_percentage)
                            # else:
                            #     masked_x_batch, cloze_mask = mask_data18(
                                    # x_batch, available_assays_ind, mask_value=-1, mask_percentage=mask_percentage)
                                
                            union_mask = cloze_mask | missing_mask_batch

                            masked_x_batch = add_noise(masked_x_batch, 0.2)
                            masked_x_batch[union_mask] = -1

                            # move to GPU
                            x_batch = x_batch.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            union_mask = union_mask.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)

                            outputs, pred_mask = self.model(masked_x_batch, ~union_mask)

                            mse_obs_loss, mse_pred_loss, bce_mask_loss = self.criterion(
                                outputs, x_batch, pred_mask, cloze_mask, union_mask)

                            loss = mse_obs_loss + mse_pred_loss + bce_mask_loss

                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                print(len(available_assays_ind), mse_obs_loss + mse_pred_loss + bce_mask_loss)
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                del x_batch
                                del masked_x_batch
                                del outputs
                                torch.cuda.empty_cache()
                                continue

                            del x_batch
                            del masked_x_batch
                            del outputs

                            epoch_loss.append(loss.item())
                            epoch_obs_loss.append(mse_obs_loss.item())
                            epoch_clz_loss.append(mse_pred_loss.item())
                            epoch_msk_loss.append(bce_mask_loss.item())

                            # Clear GPU memory again
                            torch.cuda.empty_cache()

                            loss.backward()  
                            self.optimizer.step()
                        
                        if p == 1 or p%8 == 0:
                            logfile = open("models/EPD20_log.txt", "w")

                            logstr = [
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Epoch {epoch+1}/{num_epochs}', 
                                f'Missing Pattern {p}/{len(missing_f_pattern)}', 
                                f"Obs Loss: {mse_obs_loss.item():.4f}",
                                f"Clz Loss: {mse_pred_loss.item():.4f}",
                                f"Msk Loss: {bce_mask_loss.item():.4f}"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)
                        
                    self.scheduler.step()

                    t1 = datetime.now()
                    logfile = open("models/EPD20_log.txt", "w")
                    
                    test_mse, test_corr, test_ovr = self.test_model(
                        context_length, version="20", 
                        is_arcsin=arcsinh_transform, batch_size=batch_size)

                    test_mse = np.mean(test_mse)
                    test_corr = np.mean(test_corr)

                    test_ovr_mean = np.mean(test_ovr)
                    # test_ovr_min = np.min(test_ovr)
                    # test_ovr_max = np.max(test_ovr)

                    logstr = [
                        "\n----------------------------------------------------\n"
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}', 
                        f"Obs Loss: {np.mean(epoch_obs_loss):.3f}", 
                        f"Clz Loss: {np.mean(epoch_clz_loss):.3f}", 
                        f"Msk Loss: {np.mean(epoch_msk_loss):.3f}", 
                        f"Val_MSE: {test_mse:.3f}",
                        f"Val_Corr: {test_corr:.3f}",
                        f"Val_POmean: {test_ovr_mean:.3f}",
                        # f"Val_POmin: {test_ovr_min:.3f}",
                        # f"Val_POmax: {test_ovr_max:.3f}",
                        f"Epoch took: {t1 - t0}"
                        "\n----------------------------------------------------\n"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                # Save the model after each dataset
                try:
                    if ds%5 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD20_model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model
    
    def pretrain_epidenoise_21(self, 
        d_model, outer_loop_epochs=1, arcsinh_transform=True, step_size=40,
        num_epochs=25, context_length=2000, start_ds=0):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/EPD21_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        for ole in range(outer_loop_epochs):
            ds=0

            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                N, L, num_features = x.shape

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])
                                
                for epoch in range(0, num_epochs):
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    epoch_loss = []

                    # zero grads before going over all batches and all patterns of missing data
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    t0 = datetime.now()

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1

                        p_loss = []
                        x_batch = x[indices]

                        if epoch%2==1:
                            x_batch = torch.flip(x_batch, dims=(1,))

                        missing_mask_patten_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]
                        if len(available_assays_ind) < 2:
                            continue

                        for i in range(0, L - context_length, step_size):
                            self.optimizer.zero_grad()

                            if i+context_length+step_size > L:
                                break

                            # Extract the context and the target for this step
                            context = x_batch[:, i:i+context_length, :].to(self.device)
                            missing_msk_src = missing_mask_patten_batch[:, i:i+context_length, :].to(self.device)

                            target_context = x_batch[:, i+step_size:i+context_length+step_size, :].to(self.device)

                            if torch.isnan(context).sum() > 0 or torch.isnan(target_context).sum() > 0:
                                skipmessage = "Encountered nan data! Skipping..."
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                continue

                            trg_msk = torch.zeros(
                                (target_context.shape[0], target_context.shape[1]), 
                                dtype=torch.bool, device=self.device)
                            
                            trg_msk[:, -step_size:] = True

                            next_pos_mask = torch.zeros_like(target_context, dtype=torch.bool, device=self.device)
                            next_pos_mask[:,-step_size:, :] = True
                            next_pos_mask = next_pos_mask & ~missing_msk_src

                            try:
                                outputs = self.model(
                                    context, missing_msk_src, target_context, missing_msk_src, trg_msk)

                                loss = self.criterion(outputs, target_context, missing_msk_src, next_pos_mask)

                                if torch.isnan(loss).sum() > 0:
                                    skipmessage = "Encountered nan loss! Skipping batch..."
                                    log_strs.append(skipmessage)
                                    print(skipmessage)
                                    del context, target_context, missing_msk_src, outputs, next_pos_mask, loss
                                    torch.cuda.empty_cache()
                                    continue
                                
                                p_loss.append(loss.item())
                                loss.backward()  
                                self.optimizer.step()

                                del context, target_context, missing_msk_src, outputs, next_pos_mask, loss

                            except:
                                skipmessage = f"Exception! Skipping... [e:{epoch}, p:{p}]"
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                del context, target_context, missing_msk_src, outputs, next_pos_mask, loss
                                torch.cuda.empty_cache()
                                continue

                        # Clear GPU memory again
                        torch.cuda.empty_cache()
                        if not math.isnan(np.mean(p_loss)):
                            epoch_loss.append(np.mean(p_loss))

                        if p==1 or p%4==0:
                            logfile = open("models/EPD21_log.txt", "w")
                            logstr = [
                                "\n----------------------------------------------------\n"
                                f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                                f'Pattern #: {p}/{len(missing_f_pattern)}', 
                                f'P_epoch: {np.mean(p_loss):.4f}'
                                "\n----------------------------------------------------"
                                ]
                            logstr = " | ".join(logstr)

                            log_strs.append(logstr)
                            logfile.write("\n".join(log_strs))
                            logfile.close()
                            print(logstr)

                    test_mse, test_corr, test_ovr = self.test_autoregressive_model(
                        context_length, is_arcsin=arcsinh_transform, step_size=step_size)

                    test_mse = np.mean(test_mse)
                    test_corr = np.mean(test_corr)

                    test_ovr_mean = np.mean(test_ovr)
                    torch.cuda.empty_cache()

                    t1 = datetime.now()
                    logfile = open("models/EPD21_log.txt", "w")
                    logstr = [
                        "\n----------------------------------------------------\n"
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Pattern #: {p}/{len(missing_f_pattern)}', 
                        f'epoch_loss: {np.mean(epoch_loss):.4f}', 
                        f"Val_MSE: {test_mse:.3f}",
                        f"Val_POmean: {test_ovr_mean:.3f}",
                        f"Val_Corr: {test_corr:.3f}",
                        f"Epoch took: {t1 - t0}"
                        "\n----------------------------------------------------\n"
                        ]
                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)
                        
                    self.scheduler.step()

                # Save the model after each dataset
                try:
                    if ds%5 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD21_model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def pretrain_epidenoise_22(self, 
        d_model, outer_loop_epochs=1, arcsinh_transform=True, focus_middle=False, num_random_segs=10,
        num_epochs=25, mask_percentage=0.15, context_length=2000, start_ds=0, batch_size=50):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"# model_parameters: {count_parameters(self.model)}")
        logfile = open("models/EPD22_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }

        self.masker = DataMasker(token_dict["cloze_mask"], mask_percentage)

        for ole in range(outer_loop_epochs):

            ds=0
            for ds_path in self.dataset.preprocessed_datasets:
                ds+=1
                
                if ds < start_ds:
                    continue
                
                print('-_-' * 10)
                x, missing_mask, missing_f_pattern = self.dataset.get_dataset_pt(ds_path)
                N, L, num_features = x.shape

                if arcsinh_transform:
                    arcmask = (x != -1)
                    x[arcmask] = torch.arcsinh_(x[arcmask])
                                
                for epoch in range(0, num_epochs):
                    t0 = datetime.now()
                    print('-' * 10)
                    print(f'Epoch {epoch+1}/{num_epochs}')

                    epoch_loss = []

                    pred_loss = []
                    obs_loss = []

                    mseobs_loss = []
                    msepred_loss = []

                    r2obs_loss = []
                    r2pred_loss = []

                    p = 0
                    for pattern, indices in missing_f_pattern.items():
                        p += 1
                        
                        p_batch = x[indices]
                        missing_p_batch = missing_mask[indices]

                        available_assays_ind = [feat_ind for feat_ind in range(num_features) if feat_ind not in pattern]
                        
                        if focus_middle:
                            # Select m random points along the L dimension
                            random_points = torch.randint(low=0, high=L, size=(num_random_segs,))

                            # Initialize the output tensor with padding value (e.g., 0) and the padding tracker
                            xp_batch = torch.zeros((num_random_segs * p_batch.shape[0], context_length, num_features))

                            for i, point in enumerate(random_points):
                                start = point - context_length // 2
                                end = point + context_length // 2

                                adjusted_start = max(start, 0)
                                adjusted_end = min(end, L)
                                ival = p_batch[:, adjusted_start:adjusted_end, :]

                                if ival.shape[1] < context_length:
                                    pad_length = context_length - ival.shape[1]
                                    pad = torch.full((ival.shape[0], pad_length, ival.shape[2]), token_dict["pad"])
                                    if start < 0:
                                        ival = torch.cat([pad, ival], dim=1)
                                    elif end > L:
                                        ival = torch.cat([ival, pad], dim=1)

                                xp_batch[i*p_batch.shape[0]:(i+1)*p_batch.shape[0]] = ival   
                        
                        else:
                            context_length_factor = context_length / p_batch.shape[1]

                            xp_batch = reshape_tensor(p_batch, context_length_factor)
                            missing_p_batch = reshape_tensor(missing_p_batch, context_length_factor)
                            del p_batch

                        for x_p in range(0, xp_batch.shape[0], batch_size):
                            self.optimizer.zero_grad()
                            torch.cuda.empty_cache()
                            x_batch = xp_batch[x_p:x_p+batch_size, :, :]

                            """
                            src: cloze(x_batch)
                            trg: cloze(x_batch)
                            trg_pad: x_batch_pad

                            src_mask and trg_msk
                                union_mask: union(missing, cloze, x_batch_pad)
                            """

                            x_batch_pad = (x_batch == token_dict["pad"])

                            if focus_middle:
                                masked_x_batch, cloze_mask = self.masker.mid_slice_focused_full_feature_mask(x_batch, token_dict["missing_mask"], available_assays_ind)
                            else:                                
                                masked_x_batch, cloze_mask = self.masker.mask_chunk_features(x_batch, available_assays_ind)

                            # ensure that padded regions remain padded
                            x_batch[x_batch_pad] = token_dict["pad"]
                            
                            if torch.isnan(x_batch).sum() > 0:
                                skipmessage = "Encountered nan input! Skipping batch..."
                                log_strs.append(skipmessage)
                                # print(skipmessage)
                                del x_batch
                                torch.cuda.empty_cache()
                                continue

                            x_batch_missing = (x_batch == token_dict["missing_mask"])
                            # ensure that padded regions remain padded
                            
                            union_mask = x_batch_pad | cloze_mask | x_batch_missing

                            x_batch_pad = x_batch_pad[:, :, 0]

                            x_batch_pad = x_batch_pad.to(self.device)
                            masked_x_batch = masked_x_batch.to(self.device)
                            union_mask = union_mask.to(self.device)
                            cloze_mask = cloze_mask.to(self.device)
                            x_batch = x_batch.to(self.device)

                            # outputs, aggrmean, aggrstd = self.model(masked_x_batch, union_mask, x_batch_pad) #(sequence, mask, pad)
                            outputs = self.model(masked_x_batch, union_mask, x_batch_pad) #(sequence, mask, pad)

                            aggr_mask = torch.zeros((x_batch.shape[0], x_batch.shape[2]), dtype=torch.bool)
                            for av_i in available_assays_ind:
                                aggr_mask[:,av_i] = True

                            # mse_pred_loss, mse_obs_loss, mse_aggrmean_loss, mse_aggrstd_loss = self.criterion(
                            #     outputs, x_batch, cloze_mask, union_mask, aggrmean, aggrstd, aggr_mask)
                            mse_pred_loss, mse_obs_loss = self.criterion(
                                outputs, x_batch, cloze_mask, union_mask)#, aggrmean, aggrstd, aggr_mask)


                            obs_mse = (((x_batch[~union_mask]) - (outputs[~union_mask]))**2).mean().item()
                            mseobs_loss.append(obs_mse)
                            prd_mse = (((x_batch[cloze_mask]) - (outputs[cloze_mask]))**2).mean().item()
                            msepred_loss.append(prd_mse)

                            r2_obs = r2_score(
                                (x_batch[~union_mask]).cpu().detach().numpy(), 
                                (outputs[~union_mask]).cpu().detach().numpy())
                            r2obs_loss.append(r2_obs)
                            r2_pred = r2_score(
                                (x_batch[cloze_mask]).cpu().detach().numpy(), 
                                (outputs[cloze_mask]).cpu().detach().numpy())
                            r2pred_loss.append(r2_pred)

                            loss = mse_pred_loss + mse_obs_loss #+ mse_aggrmean_loss + mse_aggrstd_loss
                            if torch.isnan(loss).sum() > 0:
                                skipmessage = "Encountered nan loss! Skipping batch..."
                                print(len(available_assays_ind), loss)
                                log_strs.append(skipmessage)
                                print(skipmessage)
                                del x_batch
                                del masked_x_batch
                                del outputs
                                torch.cuda.empty_cache()
                                continue

                            del x_batch
                            del masked_x_batch
                            del outputs

                            pred_loss.append(mse_pred_loss.item())
                            obs_loss.append(mse_obs_loss.item())
                            
                            # aggrmean_loss.append(mse_aggrmean_loss.item())
                            # aggrstd_loss.append(mse_aggrstd_loss.item())

                            epoch_loss.append(loss.item())
                            loss.backward()  
                            self.optimizer.step()
                    
                    # self.scheduler.step(np.mean(r2pred_loss))
                    logfile = open("models/EPD22_log.txt", "w")

                    elapsed_time = datetime.now() - t0
                    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    logstr = [
                        f"DataSet #{ds}/{len(self.dataset.preprocessed_datasets)}", 
                        f'Epoch {epoch+1}/{num_epochs}',
                        f"L_tot: {np.mean(epoch_loss):.2f}",
                        f"L_clz: {np.mean(pred_loss):.2f}",
                        f"L_obs: {np.mean(obs_loss):.2f}",
                        f"mse_prd: {np.mean(msepred_loss):.2f}",
                        f"mse_obs: {np.mean(mseobs_loss):.2f}",
                        f"R2_prd: {np.mean(r2pred_loss):.2f}",
                        f"R2_obs: {np.mean(r2obs_loss):.2f}",
                        f"took: {int(minutes)}:{int(seconds)}"]

                    logstr = " | ".join(logstr)

                    log_strs.append(logstr)
                    logfile.write("\n".join(log_strs))
                    logfile.close()
                    print(logstr)

                try:
                    if ds%11 == 0:
                        torch.save(self.model.state_dict(), f'models/EPD22_model_checkpoint_ds_{ds}.pth')
                except:
                    pass

        return self.model

    def __pretrain_epidenoise_30(self, 
        num_epochs=25, mask_percentage=0.15, context_length=2000, batch_size=50, inner_epochs=5, arch="a", hook=False):
        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"EPD30{arch} # model_parameters: {count_parameters(self.model)}")
        logfile = open(f"models/EPD30{arch}_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }
        dsf_list = [1, 2, 4]#, 8]
        self.masker = DataMasker(token_dict["cloze_mask"], mask_percentage)

        if hook:
            register_hooks(self.model)
            
        val_eval = MONITOR_VALIDATION(self.dataset.base_path, context_length, batch_size, arch=arch)
        
        num_total_samples = len(self.dataset.m_regions) * len(self.dataset.navigation)
        for epoch in range(num_epochs):
            self.dataset.new_epoch()
            next_epoch = False

            last_lopr = -1
            while (next_epoch==False) and (self.dataset.current_loci_batch_pointer < self.dataset.num_regions or self.dataset.current_bios_batch_pointer < self.dataset.num_bios):
                t0 = datetime.now()
                # print("new batch")
                # Randomly choose two downsampling factors and assign them to dsf_X and dsf_Y based on their values
                dsf_X, dsf_Y = sorted(random.choices(dsf_list, k=2), reverse=True) # dsf_X is of equal or higher dsf
                dsf_X, dsf_Y = 1, 1

                _X_batch, _mX_batch, _avX_batch = self.dataset.get_batch(dsf_X)
                _Y_batch, _mY_batch, _avY_batch = self.dataset.get_batch(dsf_Y)

                if _X_batch.shape != _Y_batch.shape or _mX_batch.shape != _mY_batch.shape or _avX_batch.shape != _avY_batch.shape:
                    self.dataset.update_batch_pointers()
                    print("mismatch in shapes! skipped batch...")
                    continue
                
                batch_rec = {
                    "imp_loss":[], "ups_loss":[], "msk_loss":[],
                    "ups_r2":[], "imp_r2":[],
                    "ups_mse":[], "imp_mse":[],
                    "ups_pmf":[], "imp_pmf":[],
                    "ups_conf":[], "imp_conf":[]
                    }
                for _ in range(inner_epochs):
                    # print("new inner epoch")
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                    X_batch, mX_batch, avX_batch = _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone()
                    Y_batch, mY_batch, avY_batch = _Y_batch.clone(), _mY_batch.clone(), _avY_batch.clone()

                    if arch in ["a", "b"]:
                        X_batch, mX_batch, avX_batch = self.masker.mask_feature30(X_batch, mX_batch, avX_batch)

                        masked_map = (X_batch == token_dict["cloze_mask"])
                        observed_map = (X_batch != token_dict["missing_mask"]) & (X_batch != token_dict["cloze_mask"])
                        missing_map = (X_batch == token_dict["missing_mask"])
                        masked_map = masked_map.to(self.device) # imputation targets
                        observed_map = observed_map.to(self.device) # upsampling targets
                    
                    elif arch in ["c", "d"]:
                        observed_map = (X_batch != token_dict["missing_mask"])
                        observed_map = observed_map.to(self.device) # upsampling targets
                        
                    X_batch = X_batch.float().to(self.device).requires_grad_(True)
                    mX_batch = mX_batch.to(self.device)
                    avX_batch = avX_batch.to(self.device)
                    mY_batch = mY_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)

                    if arch in ["a", "b"]:
                        output_p, output_n, output_mp, output_mo = self.model(X_batch, mX_batch, mY_batch, avX_batch)
                        pred_loss, obs_loss, msk_p_loss, msk_o_loss = self.criterion(
                            output_p, output_n, output_mp, output_mo, Y_batch, masked_map, observed_map) 

                        if torch.isnan(pred_loss).any():
                            if len(batch_rec["imp_loss"]) > 0:
                                pred_loss = torch.Tensor(np.mean(batch_rec["imp_loss"]))
                            else:
                                pred_loss = torch.Tensor(1e5)

                        if torch.isnan(obs_loss).any():
                            if len(batch_rec["ups_loss"]) > 0:
                                obs_loss = torch.Tensor(np.mean(batch_rec["ups_loss"]))
                            else:
                                obs_loss = torch.Tensor(1e5)

                        # loss = (mask_percentage * obs_loss) + (pred_loss * (1 - mask_percentage)) #+ msk_p_loss + msk_o_loss
                        # loss = (mask_percentage * obs_loss) + (pred_loss * (1 - mask_percentage)) + msk_p_loss + msk_o_loss
                        loss = pred_loss

                    elif arch in ["c", "d"]:
                        output_p, output_n = self.model(X_batch, mX_batch, mY_batch, avX_batch)
                        obs_loss = self.criterion(output_p, output_n, Y_batch, observed_map) 
                        loss = obs_loss

                    if torch.isnan(loss).sum() > 0:
                        skipmessage = "Encountered nan loss! Skipping batch..."
                        log_strs.append(skipmessage)
                        del X_batch, mX_batch, mY_batch, avX_batch, output_p, output_n, Y_batch, observed_map, loss, obs_loss
                        print(skipmessage)
                        torch.cuda.empty_cache() 
                        continue
                    
                    loss.backward()  

                    if hook:
                        # Initialize variables to store maximum gradient norms and corresponding layer names
                        max_weight_grad_norm = 0
                        max_weight_grad_layer = None
                        max_bias_grad_norm = 0
                        max_bias_grad_layer = None

                        # Check and update maximum gradient norms
                        for name, module in self.model.named_modules():
                            if hasattr(module, 'weight') and module.weight is not None and hasattr(module.weight, 'grad_norm'):
                                if module.weight.grad_norm > max_weight_grad_norm:
                                    max_weight_grad_norm = module.weight.grad_norm
                                    max_weight_grad_layer = name

                            if hasattr(module, 'bias') and module.bias is not None and hasattr(module.bias, 'grad_norm') and module.bias.grad_norm is not None:
                                if module.bias.grad_norm > max_bias_grad_norm:
                                    max_bias_grad_norm = module.bias.grad_norm
                                    max_bias_grad_layer = name

                        if max_weight_grad_layer:
                            print(f"Max Weight Grad Layer: {max_weight_grad_layer}, Weight Grad Norm: {max_weight_grad_norm:.3f}, Ups_loss: {obs_loss.item():.2f}, Imp_loss: {pred_loss.item():.2f}, mask_losses: {msk_p_loss.item():.2f},{msk_o_loss.item():.2f}")

                    self.optimizer.step()

                    if arch in ["a", "b"]:
                        imp_pred = NegativeBinomial(
                            output_p[masked_map].cpu().detach(), 
                            output_n[masked_map].cpu().detach()
                            ).expect().cpu().detach().numpy()

                        imp_true = Y_batch[masked_map].cpu().detach().numpy()
                        imp_r2 = r2_score(imp_true, imp_pred)
                        imp_pmf = NegativeBinomial(
                            output_p[masked_map].cpu().detach(),  
                            output_n[masked_map].cpu().detach()).pmf(imp_true).mean()
                        imp_mse = ((imp_true - imp_pred)**2).mean()

                        imp_std = NegativeBinomial(
                            output_p[masked_map].cpu().detach(), 
                            output_n[masked_map].cpu().detach()
                            ).std().cpu().detach().numpy()
                        imp_abs_error = torch.abs(torch.Tensor(imp_true) - torch.Tensor(imp_pred)).cpu().detach().numpy()
                        imp_errstd = pearsonr(imp_std, imp_abs_error)

                        batch_rec["imp_loss"].append(pred_loss.item())
                        batch_rec["msk_loss"].append(msk_p_loss.item() + msk_o_loss.item())
                        batch_rec["imp_mse"].append(imp_mse)
                        batch_rec["imp_r2"].append(imp_r2)
                        batch_rec["imp_pmf"].append(imp_pmf)
                        batch_rec["imp_conf"].append(imp_errstd)

                    ups_pred = NegativeBinomial(
                        output_p[observed_map].cpu().detach(), 
                        output_n[observed_map].cpu().detach()
                        ).expect().cpu().detach().numpy()

                    ups_true = Y_batch[observed_map].cpu().detach().numpy()
                    ups_pmf = NegativeBinomial(
                        output_p[observed_map].cpu().detach(), 
                        output_n[observed_map].cpu().detach()).pmf(ups_true).mean()

                    ups_std = NegativeBinomial(
                            output_p[observed_map].cpu().detach(), 
                            output_n[observed_map].cpu().detach()
                            ).std().cpu().detach().numpy()
                    ups_abs_error = torch.abs(torch.Tensor(ups_true) - torch.Tensor(ups_pred)).cpu().detach().numpy()
                    ups_errstd = pearsonr(ups_std, ups_abs_error)

                    try:
                        ups_r2 = r2_score(ups_true, ups_pred)
                        ups_mse = ((ups_true - ups_pred)**2).mean()
                    except:
                        ups_r2 = np.nan
                        ups_mse = np.nan
                
                    batch_rec["ups_loss"].append(obs_loss.item())
                    batch_rec["ups_r2"].append(ups_r2)
                    batch_rec["ups_mse"].append(ups_mse)
                    batch_rec["ups_pmf"].append(ups_pmf)
                    batch_rec["ups_conf"].append(ups_errstd)

                lopr = int((self.dataset.current_loci_batch_pointer/self.dataset.num_regions) * 100)
                if lopr > 1 and lopr % 10 == 0 and lopr != last_lopr:
                    try:
                        torch.save(
                            self.model.state_dict(), 
                            f'models/EPD30{arch}_model_checkpoint_epoch{epoch}_LociProg{lopr}.pth')
                    except:
                        pass

                elapsed_time = datetime.now() - t0
                hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)

                if arch in ["a", "b"]:
                    logstr = [
                        f"Ep. {epoch}",
                        f"DSF{dsf_X}->{dsf_Y}",
                        f"Loci Prog. {self.dataset.current_loci_batch_pointer/self.dataset.num_regions:.2%}",
                        f"Bios Prog. {self.dataset.current_bios_batch_pointer/self.dataset.num_bios:.2%}",
                        f"Imp_Loss {np.mean(batch_rec['imp_loss']):.2f}",
                        f"Ups_Loss {np.mean(batch_rec['ups_loss']):.2f}",
                        f"Msk_Loss {np.mean(batch_rec['msk_loss']):.2f}",
                        f"Imp_R2 {np.mean(batch_rec['imp_r2']):.2f}",
                        f"Ups_R2 {np.mean(batch_rec['ups_r2']):.2f}",
                        f"Imp_pmf {np.mean(batch_rec['imp_pmf']):.2f}",
                        f"Ups_pmf {np.mean(batch_rec['ups_pmf']):.2f}",
                        f"Imp_MSE {np.mean(batch_rec['imp_mse']):.2f}",
                        f"Ups_MSE {np.mean(batch_rec['ups_mse']):.2f}",
                        f"Imp_Conf {np.mean(batch_rec['imp_conf']):.2f}",
                        f"Ups_Conf {np.mean(batch_rec['ups_conf']):.2f}",
                        f"took {int(minutes)}:{int(seconds)}"]

                elif arch in ["c", "d"]:
                    logstr = [
                        f"Ep. {epoch}",
                        f"DSF{dsf_X}->{dsf_Y}",
                        f"Loci Prog. {self.dataset.current_loci_batch_pointer/self.dataset.num_regions:.2%}",
                        f"Bios Prog. {self.dataset.current_bios_batch_pointer/self.dataset.num_bios:.2%}",
                        f"Ups_Loss {np.mean(batch_rec['ups_loss']):.2f}",
                        f"Ups_R2 {np.mean(batch_rec['ups_r2']):.2f}",
                        f"Ups_pmf {np.mean(batch_rec['ups_pmf']):.2f}",
                        f"Ups_Conf {np.mean(batch_rec['ups_conf']):.2f}",
                        f"Ups_MSE {np.mean(batch_rec['ups_mse']):.2f}",
                        f"took {int(minutes)}:{int(seconds)}"]
                
                logstr = " | ".join(logstr)
                log_strs.append(logstr)
                print(logstr)
                
                if lopr % 2 == 0 and lopr != last_lopr:
                    validation_set_eval = val_eval.get_validation(self.model)
                    
                    torch.cuda.empty_cache()
                    log_strs.append(validation_set_eval)
                    print(validation_set_eval)
                    log_resource_usage()
                    
                logfile = open(f"models/EPD30{arch}_log.txt", "w")
                logfile.write("\n".join(log_strs))
                logfile.close()

                last_lopr = lopr
                next_epoch = self.dataset.update_batch_pointers()
                
            if epoch%1==0:
                try:
                    torch.save(self.model.state_dict(), f'models/EPD30{arch}_model_checkpoint_epoch{epoch}.pth')
                except:
                    pass
                
        return self.model

    def pretrain_epidenoise_30(self, 
        num_epochs=25, mask_percentage=0.15, context_length=2000, batch_size=50, inner_epochs=5, arch="a", hook=False):
        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"EPD30{arch} # model_parameters: {count_parameters(self.model)}")
        logfile = open(f"models/EPD30{arch}_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        images = []
        gif_filename = f"models/EPD30{arch}_TrainProg.gif"

        token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }
        self.masker = DataMasker(token_dict["cloze_mask"], mask_percentage)

        if hook:
            register_hooks(self.model)
            
        val_eval = MONITOR_VALIDATION(self.dataset.base_path, context_length, batch_size, arch=arch, token_dict=token_dict)
        
        num_total_samples = len(self.dataset.m_regions) * len(self.dataset.navigation)
        for epoch in range(num_epochs):
            self.dataset.new_epoch()
            next_epoch = False

            last_lopr = -1
            while (next_epoch==False):
                t0 = datetime.now()

                _X_batch, _mX_batch, _avX_batch = self.dataset.get_batch(side="x")
                _Y_batch, _mY_batch, _avY_batch = self.dataset.get_batch(side="y")

                if _X_batch.shape != _Y_batch.shape or _mX_batch.shape != _mY_batch.shape or _avX_batch.shape != _avY_batch.shape:
                    self.dataset.update_batch_pointers()
                    print("mismatch in shapes! skipped batch...")
                    continue
                
                batch_rec = {
                    "imp_loss":[], "ups_loss":[], "msk_loss":[],
                    "ups_r2":[], "imp_r2":[],
                    "ups_mse":[], "imp_mse":[],
                    "ups_pmf":[], "imp_pmf":[],
                    "ups_conf":[], "imp_conf":[]
                    }
                for _ in range(inner_epochs):
                    # print("new inner epoch")
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                    X_batch, mX_batch, avX_batch = _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone()
                    Y_batch, mY_batch, avY_batch = _Y_batch.clone(), _mY_batch.clone(), _avY_batch.clone()

                    if arch in ["a", "b", "d"]:
                        # X_batch, mX_batch, avX_batch = self.masker.mask_feature30(X_batch, mX_batch, avX_batch)
                        X_batch, mX_batch, avX_batch = self.masker.mask_chunk_features_30(X_batch, mX_batch, avX_batch)

                        masked_map = (X_batch == token_dict["cloze_mask"])
                        observed_map = (X_batch != token_dict["missing_mask"]) & (X_batch != token_dict["cloze_mask"])
                        missing_map = (X_batch == token_dict["missing_mask"])
                        masked_map = masked_map.to(self.device) # imputation targets
                        observed_map = observed_map.to(self.device) # upsampling targets
                    
                    elif arch in ["c"]:
                        observed_map = (X_batch != token_dict["missing_mask"])
                        observed_map = observed_map.to(self.device) # upsampling targets
                        
                    X_batch = X_batch.float().to(self.device).requires_grad_(True)
                    mX_batch = mX_batch.to(self.device)
                    avX_batch = avX_batch.to(self.device)
                    mY_batch = mY_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)

                    if arch in ["a", "b", "d"]:
                        output_p, output_n, output_mp, output_mo = self.model(X_batch, mX_batch, mY_batch, avX_batch)
                        pred_loss, obs_loss, msk_p_loss, msk_o_loss = self.criterion(
                            output_p, output_n, output_mp, output_mo, Y_batch, masked_map, observed_map) 

                        if torch.isnan(pred_loss).any():
                            if len(batch_rec["imp_loss"]) > 0:
                                pred_loss = torch.tensor(np.mean(batch_rec["imp_loss"]))
                            else:
                                pred_loss = torch.tensor(1e5)

                        if torch.isnan(obs_loss).any():
                            if len(batch_rec["ups_loss"]) > 0:
                                obs_loss = torch.tensor(np.mean(batch_rec["ups_loss"]))
                            else:
                                obs_loss = torch.tensor(1e5)

                        loss = (mask_percentage * obs_loss) + (pred_loss * (1 - mask_percentage))
                        # loss.backward()  

                        del output_mp, output_mo, X_batch
                        torch.cuda.empty_cache()
                        X_batch = NegativeBinomial(output_p, output_n).expect()
                        del output_p, output_n

                        output_p, output_n, output_mp, output_mo = self.model(X_batch, mY_batch, mY_batch, avX_batch)
                        refine_pred_loss, refine_obs_loss, msk_p_loss, msk_o_loss = self.criterion(
                            output_p, output_n, output_mp, output_mo, Y_batch, masked_map, observed_map)

                        ref_loss = (mask_percentage * refine_obs_loss) + (refine_pred_loss * (1 - mask_percentage))
                        # ref_loss.backward()
                        loss += ref_loss


                    elif arch in ["c"]:
                        output_p, output_n = self.model(X_batch, mX_batch, mY_batch, avX_batch)
                        obs_loss = self.criterion(output_p, output_n, Y_batch, observed_map) 
                        loss = obs_loss

                    if torch.isnan(loss).sum() > 0:
                        skipmessage = "Encountered nan loss! Skipping batch..."
                        log_strs.append(skipmessage)
                        del X_batch, mX_batch, mY_batch, avX_batch, output_p, output_n, Y_batch, observed_map, loss, obs_loss
                        print(skipmessage)
                        torch.cuda.empty_cache() 
                        continue
                    
                    loss.backward()  

                    if hook:
                        # Initialize variables to store maximum gradient norms and corresponding layer names
                        max_weight_grad_norm = 0
                        max_weight_grad_layer = None
                        max_bias_grad_norm = 0
                        max_bias_grad_layer = None

                        # Check and update maximum gradient norms
                        for name, module in self.model.named_modules():
                            if hasattr(module, 'weight') and module.weight is not None and hasattr(module.weight, 'grad_norm'):
                                if module.weight.grad_norm > max_weight_grad_norm:
                                    max_weight_grad_norm = module.weight.grad_norm
                                    max_weight_grad_layer = name

                            if hasattr(module, 'bias') and module.bias is not None and hasattr(module.bias, 'grad_norm') and module.bias.grad_norm is not None:
                                if module.bias.grad_norm > max_bias_grad_norm:
                                    max_bias_grad_norm = module.bias.grad_norm
                                    max_bias_grad_layer = name

                        if max_weight_grad_layer:
                            print(f"Max Weight Grad Layer: {max_weight_grad_layer}, Weight Grad Norm: {max_weight_grad_norm:.3f}")

                    self.optimizer.step()

                    if arch in ["a", "b", "d"]:
                        imp_pred = NegativeBinomial(
                            output_p[masked_map].cpu().detach(), 
                            output_n[masked_map].cpu().detach()
                            ).expect().cpu().detach().numpy()

                        imp_true = Y_batch[masked_map].cpu().detach().numpy()
                        imp_r2 = r2_score(imp_true, imp_pred)
                        imp_pmf = NegativeBinomial(
                            output_p[masked_map].cpu().detach(),  
                            output_n[masked_map].cpu().detach()).pmf(imp_true).mean()
                        imp_mse = ((imp_true - imp_pred)**2).mean()

                        imp_std = NegativeBinomial(
                            output_p[masked_map].cpu().detach(), 
                            output_n[masked_map].cpu().detach()
                            ).std().cpu().detach().numpy()
                        imp_abs_error = torch.abs(torch.Tensor(imp_true) - torch.Tensor(imp_pred)).cpu().detach().numpy()
                        imp_errstd = pearsonr(imp_std, imp_abs_error)

                        batch_rec["imp_loss"].append(pred_loss.item())
                        # batch_rec["msk_loss"].append(msk_p_loss.item() + msk_o_loss.item())
                        batch_rec["imp_mse"].append(imp_mse)
                        batch_rec["imp_r2"].append(imp_r2)
                        batch_rec["imp_pmf"].append(imp_pmf)
                        batch_rec["imp_conf"].append(imp_errstd)

                    ups_pred = NegativeBinomial(
                        output_p[observed_map].cpu().detach(), 
                        output_n[observed_map].cpu().detach()
                        ).expect().cpu().detach().numpy()

                    ups_true = Y_batch[observed_map].cpu().detach().numpy()
                    ups_pmf = NegativeBinomial(
                        output_p[observed_map].cpu().detach(), 
                        output_n[observed_map].cpu().detach()).pmf(ups_true).mean()

                    ups_std = NegativeBinomial(
                            output_p[observed_map].cpu().detach(), 
                            output_n[observed_map].cpu().detach()
                            ).std().cpu().detach().numpy()
                    ups_abs_error = torch.abs(torch.Tensor(ups_true) - torch.Tensor(ups_pred)).cpu().detach().numpy()
                    ups_errstd = pearsonr(ups_std, ups_abs_error)

                    try:
                        ups_r2 = r2_score(ups_true, ups_pred)
                        ups_mse = ((ups_true - ups_pred)**2).mean()
                    except:
                        ups_r2 = np.nan
                        ups_mse = np.nan
                
                    batch_rec["ups_loss"].append(obs_loss.item())
                    batch_rec["ups_r2"].append(ups_r2)
                    batch_rec["ups_mse"].append(ups_mse)
                    batch_rec["ups_pmf"].append(ups_pmf)
                    batch_rec["ups_conf"].append(ups_errstd)

                elapsed_time = datetime.now() - t0
                hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)

                if arch in ["a", "b", "d"]:
                    logstr = [
                        f"Ep. {epoch}",
                        f"DSF{self.dataset.dsf_list[self.dataset.dsf_pointer]}->{1}",
                        f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]} Prog. {self.dataset.chr_loci_pointer/len(self.dataset.loci[list(self.dataset.loci.keys())[self.dataset.chr_pointer]]):.2%}",
                        f"Bios Prog. {self.dataset.bios_pointer/self.dataset.num_bios:.2%}",
                        f"Imp_Loss {np.mean(batch_rec['imp_loss']):.2f}",
                        f"Ups_Loss {np.mean(batch_rec['ups_loss']):.2f}",
                        # f"Msk_Loss {np.mean(batch_rec['msk_loss']):.2f}",
                        f"Imp_R2 {np.mean(batch_rec['imp_r2']):.2f}",
                        f"Ups_R2 {np.mean(batch_rec['ups_r2']):.2f}",
                        # f"Imp_pmf {np.mean(batch_rec['imp_pmf']):.2f}",
                        # f"Ups_pmf {np.mean(batch_rec['ups_pmf']):.2f}",
                        f"Imp_MSE {np.mean(batch_rec['imp_mse']):.2f}",
                        f"Ups_MSE {np.mean(batch_rec['ups_mse']):.2f}",
                        f"Imp_Conf {np.mean(batch_rec['imp_conf']):.2f}",
                        f"Ups_Conf {np.mean(batch_rec['ups_conf']):.2f}",
                        f"took {int(minutes)}:{int(seconds)}"]

                elif arch in ["c"]:
                    logstr = [
                        f"Ep. {epoch}",
                        f"DSF{self.dataset.dsf_list[self.dataset.dsf_pointer]}->{1}",
                        f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]} Prog. {self.dataset.chr_loci_pointer/len(self.dataset.loci[list(self.dataset.loci.keys())[self.dataset.chr_pointer]]):.2%}",
                        f"Bios Prog. {self.dataset.bios_pointer/self.dataset.num_bios:.2%}",
                        f"Ups_Loss {np.mean(batch_rec['ups_loss']):.2f}",
                        f"Ups_R2 {np.mean(batch_rec['ups_r2']):.2f}",
                        # f"Ups_pmf {np.mean(batch_rec['ups_pmf']):.2f}",
                        # f"Ups_Conf {np.mean(batch_rec['ups_conf']):.2f}",
                        f"Ups_MSE {np.mean(batch_rec['ups_mse']):.2f}",
                        f"took {int(minutes)}:{int(seconds)}"]
                
                logstr = " | ".join(logstr)
                log_strs.append(logstr)
                print(logstr)

                logfile = open(f"models/EPD30{arch}_log.txt", "w")
                logfile.write("\n".join(log_strs))
                logfile.close()
                
                chr0 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                dsf_pointer0 = self.dataset.dsf_pointer
                bios_pointer0 = self.dataset.bios_pointer

                next_epoch = self.dataset.update_batch_pointers()

                dsf_pointer1 = self.dataset.dsf_pointer
                chr1 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                bios_pointer1 = self.dataset.bios_pointer

                if dsf_pointer0 != dsf_pointer1 or chr0 != chr1 or bios_pointer0 != bios_pointer1:
                    # Generate and process the plot
                    fig_title = " | ".join([
                        f"Ep. {epoch}", f"DSF{self.dataset.dsf_list[dsf_pointer0]}->{1}",
                        f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]}"])
                    
                    plot_buf = val_eval.generate_training_gif_frame(self.model, fig_title)
                    images.append(imageio.imread(plot_buf))
                    plot_buf.close()
                    imageio.mimsave(gif_filename, images, duration=0.5 * len(images))

                if chr0 != chr1:
                    validation_set_eval = val_eval.get_validation(self.model)
                    torch.cuda.empty_cache()
                    log_strs.append(validation_set_eval)
                    print(validation_set_eval)
                    log_resource_usage()
            
            self.scheduler.step()
            print("learning rate scheduler step...")
            if epoch%1==0:
                try:
                    torch.save(self.model.state_dict(), f'models/EPD30{arch}_model_checkpoint_epoch{epoch}.pth')
                except:
                    pass
                
        return self.model

    def pretrain_epd30_synth(self, 
        num_epochs=25, mask_percentage=0.15, context_length=2000, batch_size=50, inner_epochs=5, arch="a", hook=False):

        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"EPD30{arch} # model_parameters: {count_parameters(self.model)}")
        logfile = open(f"models/Synth_EPD30{arch}_log.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }

        if hook:
            register_hooks(self.model)
        
        for epoch in range(num_epochs):
            next_epoch = False
            mask_percentage = (0.1 + 0.2)/2

            batch_rec = {
                "imp_loss":[], "ups_loss":[], "msk_loss":[],
                "ups_r2":[], "imp_r2":[],
                "ups_mse":[], "imp_mse":[],
                "ups_pmf":[], "imp_pmf":[],
                "ups_conf":[], "imp_conf":[],}

            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            t0 = datetime.now()

            X_batch, Y_batch, mX_batch, mY_batch, avX_batch, avY_batch = self.dataset.get_batch(batch_size, miss_perc_range=(0.7, 0.9), mask_perc_range=(0.1, 0.2))
            # X_batch, Y_batch, mX_batch, mY_batch, avX_batch, avY_batch = self.dataset.get_batch(batch_size, miss_perc_range=(0, 0), mask_perc_range=(0.3, 0.5))

            if arch in ["a", "b"]:
                masked_map = (X_batch == token_dict["cloze_mask"])
                observed_map = (X_batch != token_dict["missing_mask"]) & (X_batch != token_dict["cloze_mask"])
                # missing_map = (X_batch == token_dict["missing_mask"])
                masked_map = masked_map.to(self.device) # imputation targets
                observed_map = observed_map.to(self.device) # upsampling targets
            
            elif arch in ["c", "d"]:
                observed_map = (X_batch != token_dict["missing_mask"])
                observed_map = observed_map.to(self.device) # upsampling targets
                
            X_batch = X_batch.float().to(self.device)#.requires_grad_(True)
            mX_batch = mX_batch.to(self.device)
            avX_batch = avX_batch.to(self.device)

            Y_batch = Y_batch.float().to(self.device)
            mY_batch = mY_batch.to(self.device)
            avY_batch = avY_batch.to(self.device)

            if arch in ["a", "b"]:
                output_p, output_n, output_mp, output_mo = self.model(X_batch, mX_batch, mY_batch, avX_batch)
                pred_loss, obs_loss, msk_p_loss, msk_o_loss = self.criterion(
                    output_p, output_n, output_mp, output_mo, Y_batch, masked_map, observed_map) 

                if torch.isnan(pred_loss).any():
                    if len(batch_rec["imp_loss"]) > 0:
                        pred_loss = torch.Tensor(np.mean(batch_rec["imp_loss"]))
                    else:
                        pred_loss = torch.Tensor(1e5)

                if torch.isnan(obs_loss).any():
                    if len(batch_rec["ups_loss"]) > 0:
                        obs_loss = torch.Tensor(np.mean(batch_rec["ups_loss"]))
                    else:
                        obs_loss = torch.Tensor(1e5)

                loss = pred_loss
                # loss = (mask_percentage * obs_loss) + (pred_loss * (1 - mask_percentage))# + msk_p_loss + msk_o_loss
                # loss = (mask_percentage * obs_loss) + (pred_loss * (1 - mask_percentage)) + msk_p_loss + msk_o_loss

            elif arch in ["c", "d"]:
                output_p, output_n = self.model(X_batch, mX_batch, mY_batch, avX_batch)
                obs_loss = self.criterion(output_p, output_n, Y_batch, observed_map) 
                loss = obs_loss

            if torch.isnan(loss).sum() > 0:
                skipmessage = "Encountered nan loss! Skipping batch..."
                log_strs.append(skipmessage)
                del X_batch, mX_batch, mY_batch, avX_batch, output_p, output_n, Y_batch, observed_map, loss, obs_loss
                print(skipmessage)
                torch.cuda.empty_cache() 
                continue
            
            loss.backward()  

            if hook:
                # Initialize variables to store maximum gradient norms and corresponding layer names
                max_weight_grad_norm = 0
                max_weight_grad_layer = None
                max_bias_grad_norm = 0
                max_bias_grad_layer = None

                # Check and update maximum gradient norms
                for name, module in self.model.named_modules():
                    if hasattr(module, 'weight') and module.weight is not None and hasattr(module.weight, 'grad_norm'):
                        if module.weight.grad_norm > max_weight_grad_norm:
                            max_weight_grad_norm = module.weight.grad_norm
                            max_weight_grad_layer = name

                    if hasattr(module, 'bias') and module.bias is not None and hasattr(module.bias, 'grad_norm') and module.bias.grad_norm is not None:
                        if module.bias.grad_norm > max_bias_grad_norm:
                            max_bias_grad_norm = module.bias.grad_norm
                            max_bias_grad_layer = name

                if max_weight_grad_layer:
                    print(f"Max Weight Grad Layer: {max_weight_grad_layer}, Weight Grad Norm: {max_weight_grad_norm:.3f}, Ups_loss: {obs_loss.item():.2f}, Imp_loss: {pred_loss.item():.2f}, mask_losses: {msk_p_loss.item():.2f},{msk_o_loss.item():.2f}")

            self.optimizer.step()

            if arch in ["a", "b"]:
                imp_pred = NegativeBinomial(
                    output_p[masked_map].cpu().detach(), 
                    output_n[masked_map].cpu().detach()
                    ).expect().cpu().detach().numpy()

                imp_true = Y_batch[masked_map].cpu().detach().numpy()
                imp_r2 = r2_score(imp_true, imp_pred)
                imp_pmf = NegativeBinomial(
                    output_p[masked_map].cpu().detach(),  
                    output_n[masked_map].cpu().detach()).pmf(imp_true).mean()
                imp_mse = ((imp_true - imp_pred)**2).mean()

                imp_std = NegativeBinomial(
                    output_p[masked_map].cpu().detach(), 
                    output_n[masked_map].cpu().detach()
                    ).std().cpu().detach().numpy()
                imp_abs_error = torch.abs(torch.Tensor(imp_true) - torch.Tensor(imp_pred)).cpu().detach().numpy()
                imp_errstd = pearsonr(imp_std, imp_abs_error)

                batch_rec["imp_loss"].append(pred_loss.item())
                batch_rec["msk_loss"].append(msk_p_loss.item() + msk_o_loss.item())
                batch_rec["imp_mse"].append(imp_mse)
                batch_rec["imp_r2"].append(imp_r2)
                batch_rec["imp_pmf"].append(imp_pmf)
                batch_rec["imp_conf"].append(imp_errstd)

            ups_pred = NegativeBinomial(
                output_p[observed_map].cpu().detach(), 
                output_n[observed_map].cpu().detach()
                ).expect().cpu().detach().numpy()

            ups_true = Y_batch[observed_map].cpu().detach().numpy()
            ups_pmf = NegativeBinomial(
                output_p[observed_map].cpu().detach(), 
                output_n[observed_map].cpu().detach()).pmf(ups_true).mean()

            ups_std = NegativeBinomial(
                    output_p[observed_map].cpu().detach(), 
                    output_n[observed_map].cpu().detach()
                    ).std().cpu().detach().numpy()
            ups_abs_error = torch.abs(torch.Tensor(ups_true) - torch.Tensor(ups_pred)).cpu().detach().numpy()
            ups_errstd = pearsonr(ups_std, ups_abs_error)

            try:
                ups_r2 = r2_score(ups_true, ups_pred)
                ups_mse = ((ups_true - ups_pred)**2).mean()
            except:
                ups_r2 = np.nan
                ups_mse = np.nan
        
            batch_rec["ups_loss"].append(obs_loss.item())
            batch_rec["ups_r2"].append(ups_r2)
            batch_rec["ups_mse"].append(ups_mse)
            batch_rec["ups_pmf"].append(ups_pmf)
            batch_rec["ups_conf"].append(ups_errstd)

            elapsed_time = datetime.now() - t0
            hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)

            if arch in ["a", "b"]:
                logstr = [
                    f"Ep. {epoch}",
                    f"Imp_Loss {np.mean(batch_rec['imp_loss']):.2f}",
                    f"Ups_Loss {np.mean(batch_rec['ups_loss']):.2f}",
                    f"Msk_Loss {np.mean(batch_rec['msk_loss']):.2f}",
                    f"Imp_R2 {np.mean(batch_rec['imp_r2']):.2f}",
                    f"Ups_R2 {np.mean(batch_rec['ups_r2']):.2f}",
                    f"Imp_pmf {np.mean(batch_rec['imp_pmf']):.2f}",
                    f"Ups_pmf {np.mean(batch_rec['ups_pmf']):.2f}",
                    f"Imp_MSE {np.mean(batch_rec['imp_mse']):.2f}",
                    f"Ups_MSE {np.mean(batch_rec['ups_mse']):.2f}",
                    f"Imp_Conf {np.mean(batch_rec['imp_conf']):.2f}",
                    f"Ups_Conf {np.mean(batch_rec['ups_conf']):.2f}",
                    f"took {int(minutes)}:{int(seconds)}"]

            elif arch in ["c", "d"]:
                logstr = [
                    f"Ep. {epoch}",
                    f"Ups_Loss {np.mean(batch_rec['ups_loss']):.2f}",
                    f"Ups_R2 {np.mean(batch_rec['ups_r2']):.2f}",
                    f"Ups_pmf {np.mean(batch_rec['ups_pmf']):.2f}",
                    f"Ups_Conf {np.mean(batch_rec['ups_conf']):.2f}",
                    f"Ups_MSE {np.mean(batch_rec['ups_mse']):.2f}",
                    f"took {int(minutes)}:{int(seconds)}"]
            
            logstr = " | ".join(logstr)
            log_strs.append(logstr)
            print(logstr)
                
            logfile = open(f"models/Synth_EPD30{arch}_log.txt", "w")
            logfile.write("\n".join(log_strs))
            logfile.close()
                
        return self.model

    def pretrain_epidenoise_30_eic(self, 
        num_epochs=25, mask_percentage=0.15, context_length=2000, batch_size=50, inner_epochs=5, arch="d", hook=False):
        log_strs = []
        log_strs.append(str(self.device))
        log_strs.append(f"EPD30{arch}_eic # model_parameters: {count_parameters(self.model)}")
        logfile = open(f"models/EPD30{arch}_log_eic.txt", "w")
        logfile.write("\n".join(log_strs))
        logfile.close()

        images = []
        gif_filename = f"models/EPD30{arch}_eic_TrainProg.gif"

        token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
        }
        self.masker = DataMasker(token_dict["cloze_mask"], mask_percentage)

        if hook:
            register_hooks(self.model)
            
        val_eval = MONITOR_VALIDATION(self.dataset.base_path, context_length, batch_size, arch=arch, token_dict=token_dict, eic=True)
        num_total_samples = len(self.dataset.m_regions) * len(self.dataset.navigation)

        for epoch in range(num_epochs):
            self.dataset.new_epoch()
            next_epoch = False

            last_lopr = -1
            while (next_epoch==False):
                t0 = datetime.now()

                _X_batch, _mX_batch, _avX_batch = self.dataset.get_batch(side="x")
                _Y_batch, _mY_batch, _avY_batch = self.dataset.get_batch(side="y")

                if _X_batch.shape != _Y_batch.shape or _mX_batch.shape != _mY_batch.shape or _avX_batch.shape != _avY_batch.shape:
                    self.dataset.update_batch_pointers()
                    print("mismatch in shapes! skipped batch...")
                    continue
                
                batch_rec = {
                    "imp_loss":[], "ups_loss":[], "msk_loss":[],
                    "ups_r2":[], "imp_r2":[],
                    "ups_mse":[], "imp_mse":[],
                    "ups_pmf":[], "imp_pmf":[],
                    "ups_conf":[], "imp_conf":[]
                    }
                for _ in range(inner_epochs):
                    # print("new inner epoch")
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                    X_batch, mX_batch, avX_batch = _X_batch.clone(), _mX_batch.clone(), _avX_batch.clone()
                    Y_batch, mY_batch, avY_batch = _Y_batch.clone(), _mY_batch.clone(), _avY_batch.clone()

                    if arch in ["a", "b", "d"]:
                        # X_batch, mX_batch, avX_batch = self.masker.mask_feature30(X_batch, mX_batch, avX_batch)
                        X_batch, mX_batch, avX_batch = self.masker.mask_chunk_features_30(X_batch, mX_batch, avX_batch)

                        masked_map = (X_batch == token_dict["cloze_mask"])
                        observed_map = (X_batch != token_dict["missing_mask"]) & (X_batch != token_dict["cloze_mask"])
                        missing_map = (X_batch == token_dict["missing_mask"])
                        masked_map = masked_map.to(self.device) # imputation targets
                        observed_map = observed_map.to(self.device) # upsampling targets
                    
                    elif arch in ["c"]:
                        observed_map = (X_batch != token_dict["missing_mask"])
                        observed_map = observed_map.to(self.device) # upsampling targets
                        
                    X_batch = X_batch.float().to(self.device).requires_grad_(True)
                    mX_batch = mX_batch.to(self.device)
                    avX_batch = avX_batch.to(self.device)
                    mY_batch = mY_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)

                    if arch in ["a", "b", "d"]:
                        output_p, output_n, output_mp, output_mo = self.model(X_batch, mX_batch, mY_batch, avX_batch)
                        pred_loss, obs_loss, msk_p_loss, msk_o_loss = self.criterion(
                            output_p, output_n, output_mp, output_mo, Y_batch, masked_map, observed_map) 

                        if torch.isnan(pred_loss).any():
                            if len(batch_rec["imp_loss"]) > 0:
                                pred_loss = torch.tensor(np.mean(batch_rec["imp_loss"]))
                            else:
                                pred_loss = torch.tensor(1e5)

                        if torch.isnan(obs_loss).any():
                            if len(batch_rec["ups_loss"]) > 0:
                                obs_loss = torch.tensor(np.mean(batch_rec["ups_loss"]))
                            else:
                                obs_loss = torch.tensor(1e5)

                        loss = (mask_percentage * obs_loss) + (pred_loss * (1 - mask_percentage))# + msk_p_loss + msk_o_loss
                        # loss = pred_loss #+ msk_p_loss + msk_o_loss

                    elif arch in ["c"]:
                        output_p, output_n = self.model(X_batch, mX_batch, mY_batch, avX_batch)
                        obs_loss = self.criterion(output_p, output_n, Y_batch, observed_map) 
                        loss = obs_loss

                    if torch.isnan(loss).sum() > 0:
                        skipmessage = "Encountered nan loss! Skipping batch..."
                        log_strs.append(skipmessage)
                        del X_batch, mX_batch, mY_batch, avX_batch, output_p, output_n, Y_batch, observed_map, loss, obs_loss
                        print(skipmessage)
                        torch.cuda.empty_cache() 
                        continue
                    
                    loss.backward()  

                    if hook:
                        # Initialize variables to store maximum gradient norms and corresponding layer names
                        max_weight_grad_norm = 0
                        max_weight_grad_layer = None
                        max_bias_grad_norm = 0
                        max_bias_grad_layer = None

                        # Check and update maximum gradient norms
                        for name, module in self.model.named_modules():
                            if hasattr(module, 'weight') and module.weight is not None and hasattr(module.weight, 'grad_norm'):
                                if module.weight.grad_norm > max_weight_grad_norm:
                                    max_weight_grad_norm = module.weight.grad_norm
                                    max_weight_grad_layer = name

                            if hasattr(module, 'bias') and module.bias is not None and hasattr(module.bias, 'grad_norm') and module.bias.grad_norm is not None:
                                if module.bias.grad_norm > max_bias_grad_norm:
                                    max_bias_grad_norm = module.bias.grad_norm
                                    max_bias_grad_layer = name

                        if max_weight_grad_layer:
                            print(f"Max Weight Grad Layer: {max_weight_grad_layer}, Weight Grad Norm: {max_weight_grad_norm:.3f}")

                    self.optimizer.step()

                    if arch in ["a", "b", "d"]:
                        imp_pred = NegativeBinomial(
                            output_p[masked_map].cpu().detach(), 
                            output_n[masked_map].cpu().detach()
                            ).expect().cpu().detach().numpy()

                        imp_true = Y_batch[masked_map].cpu().detach().numpy()
                        imp_r2 = r2_score(imp_true, imp_pred)
                        imp_pmf = NegativeBinomial(
                            output_p[masked_map].cpu().detach(),  
                            output_n[masked_map].cpu().detach()).pmf(imp_true).mean()
                        imp_mse = ((imp_true - imp_pred)**2).mean()

                        imp_std = NegativeBinomial(
                            output_p[masked_map].cpu().detach(), 
                            output_n[masked_map].cpu().detach()
                            ).std().cpu().detach().numpy()
                        imp_abs_error = torch.abs(torch.Tensor(imp_true) - torch.Tensor(imp_pred)).cpu().detach().numpy()
                        imp_errstd = pearsonr(imp_std, imp_abs_error)

                        batch_rec["imp_loss"].append(pred_loss.item())
                        # batch_rec["msk_loss"].append(msk_p_loss.item() + msk_o_loss.item())
                        batch_rec["imp_mse"].append(imp_mse)
                        batch_rec["imp_r2"].append(imp_r2)
                        batch_rec["imp_pmf"].append(imp_pmf)
                        batch_rec["imp_conf"].append(imp_errstd)

                    ups_pred = NegativeBinomial(
                        output_p[observed_map].cpu().detach(), 
                        output_n[observed_map].cpu().detach()
                        ).expect().cpu().detach().numpy()

                    ups_true = Y_batch[observed_map].cpu().detach().numpy()
                    ups_pmf = NegativeBinomial(
                        output_p[observed_map].cpu().detach(), 
                        output_n[observed_map].cpu().detach()).pmf(ups_true).mean()

                    ups_std = NegativeBinomial(
                            output_p[observed_map].cpu().detach(), 
                            output_n[observed_map].cpu().detach()
                            ).std().cpu().detach().numpy()
                    ups_abs_error = torch.abs(torch.Tensor(ups_true) - torch.Tensor(ups_pred)).cpu().detach().numpy()
                    ups_errstd = pearsonr(ups_std, ups_abs_error)

                    try:
                        ups_r2 = r2_score(ups_true, ups_pred)
                        ups_mse = ((ups_true - ups_pred)**2).mean()
                    except:
                        ups_r2 = np.nan
                        ups_mse = np.nan
                
                    batch_rec["ups_loss"].append(obs_loss.item())
                    batch_rec["ups_r2"].append(ups_r2)
                    batch_rec["ups_mse"].append(ups_mse)
                    batch_rec["ups_pmf"].append(ups_pmf)
                    batch_rec["ups_conf"].append(ups_errstd)

                elapsed_time = datetime.now() - t0
                hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)

                if arch in ["a", "b", "d"]:
                    logstr = [
                        f"Ep. {epoch}",
                        f"DSF{self.dataset.dsf_list[self.dataset.dsf_pointer]}->{1}",
                        f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]} Prog. {self.dataset.chr_loci_pointer/len(self.dataset.loci[list(self.dataset.loci.keys())[self.dataset.chr_pointer]]):.2%}",
                        f"Bios Prog. {self.dataset.bios_pointer/self.dataset.num_bios:.2%}",
                        f"Imp_Loss {np.mean(batch_rec['imp_loss']):.2f}",
                        f"Ups_Loss {np.mean(batch_rec['ups_loss']):.2f}",
                        # f"Msk_Loss {np.mean(batch_rec['msk_loss']):.2f}",
                        f"Imp_R2 {np.mean(batch_rec['imp_r2']):.2f}",
                        f"Ups_R2 {np.mean(batch_rec['ups_r2']):.2f}",
                        # f"Imp_pmf {np.mean(batch_rec['imp_pmf']):.2f}",
                        # f"Ups_pmf {np.mean(batch_rec['ups_pmf']):.2f}",
                        f"Imp_MSE {np.mean(batch_rec['imp_mse']):.2f}",
                        f"Ups_MSE {np.mean(batch_rec['ups_mse']):.2f}",
                        f"Imp_Conf {np.mean(batch_rec['imp_conf']):.2f}",
                        f"Ups_Conf {np.mean(batch_rec['ups_conf']):.2f}",
                        f"took {int(minutes)}:{int(seconds)}"]

                elif arch in ["c"]:
                    logstr = [
                        f"Ep. {epoch}",
                        f"DSF{self.dataset.dsf_list[self.dataset.dsf_pointer]}->{1}",
                        f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]} Prog. {self.dataset.chr_loci_pointer/len(self.dataset.loci[list(self.dataset.loci.keys())[self.dataset.chr_pointer]]):.2%}",
                        f"Bios Prog. {self.dataset.bios_pointer/self.dataset.num_bios:.2%}",
                        f"Ups_Loss {np.mean(batch_rec['ups_loss']):.2f}",
                        f"Ups_R2 {np.mean(batch_rec['ups_r2']):.2f}",
                        # f"Ups_pmf {np.mean(batch_rec['ups_pmf']):.2f}",
                        # f"Ups_Conf {np.mean(batch_rec['ups_conf']):.2f}",
                        f"Ups_MSE {np.mean(batch_rec['ups_mse']):.2f}",
                        f"took {int(minutes)}:{int(seconds)}"]
                
                logstr = " | ".join(logstr)
                log_strs.append(logstr)
                print(logstr)

                logfile = open(f"models/EPD30{arch}_log_eic.txt", "w")
                logfile.write("\n".join(log_strs))
                logfile.close()
                
                chr0 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                dsf_pointer0 = self.dataset.dsf_pointer
                bios_pointer0 = self.dataset.bios_pointer

                next_epoch = self.dataset.update_batch_pointers()

                dsf_pointer1 = self.dataset.dsf_pointer
                chr1 = list(self.dataset.loci.keys())[self.dataset.chr_pointer]
                bios_pointer1 = self.dataset.bios_pointer

                # if dsf_pointer0 != dsf_pointer1 or chr0 != chr1 or bios_pointer0 != bios_pointer1:
                #     # Generate and process the plot
                #     fig_title = " | ".join([
                #         f"Ep. {epoch}", f"DSF{self.dataset.dsf_list[dsf_pointer0]}->{1}",
                #         f"{list(self.dataset.loci.keys())[self.dataset.chr_pointer]}"])
                    
                #     plot_buf = val_eval.generate_training_gif_frame(self.model, fig_title)
                #     images.append(imageio.imread(plot_buf))
                #     plot_buf.close()
                #     imageio.mimsave(gif_filename, images, duration=0.5 * len(images))

                if chr0 != chr1:
                    validation_set_eval = val_eval.get_validation(self.model)
                    torch.cuda.empty_cache()
                    log_strs.append(validation_set_eval)
                    print(validation_set_eval)
                    log_resource_usage()
            
            self.scheduler.step()
            print("learning rate scheduler step...")
            if epoch%1==0:
                try:
                    torch.save(self.model.state_dict(), f'models/EPD30{arch}_eic_model_checkpoint_epoch{epoch}.pth')
                except:
                    pass
                
        return self.model

#========================================================================================================#
#==========================================  Loader  ====================================================#
#========================================================================================================#

class MODEL_LOADER(object):
    def __init__(self, model_path, hyper_parameters):
        self.model_path = model_path
        self.hyper_parameters = hyper_parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_epidenoise(self, version= "16"):
        if version in ["10", "15", "16", "17", "18"]:
            input_dim = output_dim = self.hyper_parameters["input_dim"]
            dropout = self.hyper_parameters["dropout"]
            nhead = self.hyper_parameters["nhead"]
            d_model = self.hyper_parameters["d_model"]
            nlayers = self.hyper_parameters["nlayers"]
            context_length = self.hyper_parameters["context_length"]
        
        # Assuming model is an instance of the correct class
        if version == "10":
            model = EpiDenoise10(
                input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
                output_dim=output_dim, dropout=dropout, context_length=context_length)

        elif version == "15":
            model = EpiDenoise15(
                input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
                output_dim=output_dim, dropout=dropout, context_length=context_length)
        
        elif version == "16":
            model = EpiDenoise16(
                input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
                output_dim=output_dim, dropout=dropout, context_length=context_length)

        elif version == "17":
            model = EpiDenoise17(
                input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
                output_dim=output_dim, dropout=dropout, context_length=context_length)
        
        elif version == "18":
            model = EpiDenoise18(
                input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
                output_dim=output_dim, dropout=dropout, context_length=context_length)

        elif version == "22":
            input_dim = output_dim = self.hyper_parameters["input_dim"]
            dropout = self.hyper_parameters["dropout"]
            nhead = self.hyper_parameters["nhead"]
            n_enc_layers = self.hyper_parameters["n_enc_layers"]
            n_dec_layers = self.hyper_parameters["n_dec_layers"]
            context_length = self.hyper_parameters["context_length"]

            conv_out_channels = self.hyper_parameters["conv_out_channels"]
            d_model = conv_out_channels[-1]

            dilation = self.hyper_parameters["dilation"]
            kernel_size = self.hyper_parameters["kernel_size"]

            model = EpiDenoise22(
                input_dim, conv_out_channels, kernel_size, nhead, 
                d_model, n_enc_layers, n_dec_layers, output_dim, 
                dilation=dilation, dropout=dropout, context_length=context_length)
        
        elif version == "30a":
            input_dim = output_dim = self.hyper_parameters["input_dim"]
            dropout = self.hyper_parameters["dropout"]
            nhead = self.hyper_parameters["nhead"]
            d_model = self.hyper_parameters["d_model"]
            nlayers = self.hyper_parameters["nlayers"]
            metadata_embedding_dim = self.hyper_parameters["metadata_embedding_dim"]
            context_length = self.hyper_parameters["context_length"]

            model = EpiDenoise30a(
                input_dim, metadata_embedding_dim, nhead, d_model, nlayers, output_dim, 
                dropout=dropout, context_length=context_length, pos_enc="relative")
        
        elif version == "30b":          
            input_dim = output_dim = self.hyper_parameters["input_dim"]
            dropout = self.hyper_parameters["dropout"]
            nhead = self.hyper_parameters["nhead"]
            d_model = self.hyper_parameters["d_model"]
            nlayers = self.hyper_parameters["nlayers"]
            metadata_embedding_dim = self.hyper_parameters["metadata_embedding_dim"]
            context_length = self.hyper_parameters["context_length"]

            n_cnn_layers = self.hyper_parameters["n_cnn_layers"]
            conv_kernel_size = self.hyper_parameters["conv_kernel_size"]
            n_decoder_layers = self.hyper_parameters["n_decoder_layers"]

            model = EpiDenoise30b(input_dim, metadata_embedding_dim, conv_kernel_size, 
                n_cnn_layers, nhead, d_model, nlayers, output_dim, n_decoder_layers,
                dropout=dropout, context_length=context_length, pos_enc="relative")
        
        elif version == "30c":
            input_dim = output_dim = self.hyper_parameters["input_dim"]
            dropout = self.hyper_parameters["dropout"]
            nhead = self.hyper_parameters["nhead"]
            d_model = self.hyper_parameters["d_model"]
            nlayers = self.hyper_parameters["nlayers"]
            metadata_embedding_dim = self.hyper_parameters["metadata_embedding_dim"]
            context_length = self.hyper_parameters["context_length"]

            n_cnn_layers = self.hyper_parameters["n_cnn_layers"]
            conv_kernel_size = self.hyper_parameters["conv_kernel_size"]
            pool_size = self.hyper_parameters["pool_size"]

            model = EpiDenoise30c(input_dim, metadata_embedding_dim, conv_kernel_size, 
            n_cnn_layers, nhead, d_model, nlayers, output_dim, pool_size, 
            dropout=dropout, context_length=context_length)
        
        elif version == "30d":
            input_dim = output_dim = self.hyper_parameters["input_dim"]
            dropout = self.hyper_parameters["dropout"]
            nhead = self.hyper_parameters["nhead"]
            d_model = self.hyper_parameters["d_model"]
            nlayers = self.hyper_parameters["nlayers"]
            metadata_embedding_dim = self.hyper_parameters["metadata_embedding_dim"]
            context_length = self.hyper_parameters["context_length"]

            n_cnn_layers = self.hyper_parameters["n_cnn_layers"]
            conv_kernel_size = self.hyper_parameters["conv_kernel_size"]
            pool_size = self.hyper_parameters["pool_size"]

            model = EpiDenoise30d(input_dim, metadata_embedding_dim, conv_kernel_size, 
            n_cnn_layers, nhead, d_model, nlayers, output_dim, pool_size = pool_size,
            dropout=dropout, context_length=context_length)

        model.load_state_dict(torch.load(self.model_path, map_location=self.device)) 
        model = model.to(self.device)
        return model

#========================================================================================================#
#=========================================  Trainer  ====================================================#
#========================================================================================================#

def train_epidenoise10(hyper_parameters, checkpoint_path=None, start_ds=0):

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

    model = EpiDenoise10(
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

    model_name = f"EpiDenoise10_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters10_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    # criterion = WeightedMSELoss()
    criterion = nn.MSELoss()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_10(d_model=d_model, num_epochs=epochs, 
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

def train_epidenoise15(hyper_parameters, checkpoint_path=None, start_ds=0):

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
    alpha = hyper_parameters["alpha"]

    # one nucleosome is around 150bp -> 6bins
    # each chuck ~ 1 nucleosome

    n_chunks = (mask_percentage * context_length) // 6 

    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise15(
        input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
        output_dim=output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise15_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters15_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    # criterion = WeightedMSELoss()
    criterion = ComboLoss15(alpha=alpha)

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_15(d_model=d_model, num_epochs=epochs, 
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

def train_epidenoise16(hyper_parameters, checkpoint_path=None, start_ds=0):

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

    n_chunks = (mask_percentage * context_length) // 6 

    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise16(
        input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
        output_dim=output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=330, gamma=0.5)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise16_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters16_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = ComboLoss16()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_16(d_model=d_model, num_epochs=epochs, 
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

def train_epidenoise17(hyper_parameters, checkpoint_path=None, start_ds=0):

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

    n_chunks = (mask_percentage * context_length) // 6 

    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise17(
        input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
        output_dim=output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=330, gamma=0.5)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise17_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters17_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    # criterion = WeightedMSELoss()
    criterion = ComboLoss17()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_17(d_model=d_model, num_epochs=epochs, 
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

def train_epidenoise18(hyper_parameters, checkpoint_path=None, start_ds=0):
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

    n_chunks = (mask_percentage * context_length) // 6 

    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise18(
        input_dim=input_dim, nhead=nhead, d_model=d_model, nlayers=nlayers, 
        output_dim=output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=66, gamma=0.5)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise18_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters18_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = ComboLoss18()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_18(d_model=d_model, num_epochs=epochs, 
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

def train_epidenoise20(hyper_parameters, checkpoint_path=None, start_ds=0):
    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    epochs = hyper_parameters["epochs"]
    mask_percentage = hyper_parameters["mask_percentage"]
    context_length = hyper_parameters["context_length"]

    conv_out_channels = hyper_parameters["conv_out_channels"]
    dilation = hyper_parameters["dilation"]
    kernel_size = hyper_parameters["kernel_size"]

    # one nucleosome is around 150bp -> 6bins
    # each chuck ~ 1 nucleosome

    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise20(
        input_dim=input_dim, conv_out_channels=conv_out_channels, conv_kernel_sizes=kernel_size,
        dilation=dilation, nhead=nhead, d_model=d_model, n_encoder_layers=nlayers, 
        output_dim= output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=110, gamma=0.75)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise20_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters20_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = ComboLoss20()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_20(d_model=d_model, num_epochs=epochs, 
        mask_percentage=mask_percentage, context_length=context_length, 
        batch_size=batch_size, start_ds=start_ds)

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

def train_epidenoise21(hyper_parameters, checkpoint_path=None, start_ds=0):
    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    epochs = hyper_parameters["epochs"]
    context_length = hyper_parameters["context_length"]

    conv_out_channels = hyper_parameters["conv_out_channels"]
    dilation = hyper_parameters["dilation"]
    kernel_size = hyper_parameters["kernel_size"]

    # one nucleosome is around 150bp -> 6bins
    # each chuck ~ 1 nucleosome

    learning_rate = hyper_parameters["learning_rate"]
    # end of hyperparameters

    model = EpiDenoise21(
        input_dim, conv_out_channels, kernel_size, dilation, nhead, 
        d_model, nlayers, output_dim, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9326035)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise21_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters21_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    criterion = ComboLoss21()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_21(
        d_model=d_model, num_epochs=epochs,  
        context_length=context_length, start_ds=start_ds)

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

def train_epidenoise22(hyper_parameters, checkpoint_path=None, start_ds=0):
    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    n_enc_layers = hyper_parameters["n_enc_layers"]
    n_dec_layers = hyper_parameters["n_dec_layers"]
    epochs = hyper_parameters["epochs"]
    context_length = hyper_parameters["context_length"]

    conv_out_channels = hyper_parameters["conv_out_channels"]
    d_model = conv_out_channels[-1]

    dilation = hyper_parameters["dilation"]
    kernel_size = hyper_parameters["kernel_size"]

    learning_rate = hyper_parameters["learning_rate"]
    batch_size = hyper_parameters["batch_size"]
    mask_percentage = hyper_parameters["mask_percentage"]
    outer_loop_epochs = hyper_parameters["outer_loop_epochs"]
    # end of hyperparameters

    model = EpiDenoise22(
        input_dim, conv_out_channels, kernel_size, nhead, 
        d_model, n_enc_layers, n_dec_layers, output_dim, dilation=dilation, dropout=dropout, context_length=context_length)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.8, patience=epochs*5, threshold=1e-3)

    # Load from checkpoint if provided
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    # model = model.to(device)

    print(f"# model_parameters: {count_parameters(model)}")
    dataset = ENCODE_IMPUTATION_DATASET(data_path)

    model_name = f"EpiDenoise22_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters22_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    # criterion = ComboPoissonNLLloss()
    criterion = ComboLoss22(alpha=0.95)

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_22(
        d_model=d_model, num_epochs=epochs, mask_percentage=mask_percentage, outer_loop_epochs=outer_loop_epochs, 
        context_length=context_length, start_ds=start_ds, batch_size=batch_size)
        
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

def train_epidenoise30(hyper_parameters, checkpoint_path=None, arch="d"):
    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]

    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    metadata_embedding_dim = hyper_parameters["metadata_embedding_dim"]
    
    resolution = 25
    epochs = hyper_parameters["epochs"]
    num_training_loci = hyper_parameters["num_loci"]
    mask_percentage = hyper_parameters["mask_percentage"]
    context_length = hyper_parameters["context_length"]
    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    lr_halflife = hyper_parameters["lr_halflife"]
    min_avail = hyper_parameters["min_avail"]
    inner_epochs = hyper_parameters["inner_epochs"]

    # end of hyperparameters
    if arch == "a":
        model = EpiDenoise30a(input_dim, metadata_embedding_dim, nhead, d_model, nlayers, output_dim, 
            dropout=dropout, context_length=context_length, pos_enc="relative")
            
    elif arch == "b":
        n_cnn_layers = hyper_parameters["n_cnn_layers"]
        conv_kernel_size = hyper_parameters["conv_kernel_size"]
        n_decoder_layers = hyper_parameters["n_decoder_layers"]

        model = EpiDenoise30b(input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, output_dim, n_decoder_layers,
        nlayers=nlayers, dropout=dropout, context_length=context_length, pos_enc="relative")

    elif arch == "c":
        n_cnn_layers = hyper_parameters["n_cnn_layers"]
        conv_kernel_size = hyper_parameters["conv_kernel_size"]
        pool_size = hyper_parameters["pool_size"]

        model = EpiDenoise30c(input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, pool_size, 
        dropout=dropout, context_length=context_length)
    
    elif arch == "d":
        n_cnn_layers = hyper_parameters["n_cnn_layers"]
        conv_kernel_size = hyper_parameters["conv_kernel_size"]
        pool_size = hyper_parameters["pool_size"]

        model = EpiDenoise30d(input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, pool_size = pool_size,
        dropout=dropout, context_length=context_length, pos_enc="relative")

    optimizer = optim.Adamax(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_halflife, gamma=0.5)
    # scheduler = None

    if checkpoint_path is not None:
        print("loading pretrained model...")
        model.load_state_dict(torch.load(checkpoint_path))

    print(f"EPD30{arch} # model_parameters: {count_parameters(model)}")
    summary(model)

    dataset = ExtendedEncodeDataHandler(data_path)
    dataset.initialize_EED(
        m=num_training_loci, context_length=context_length*resolution, 
        bios_batchsize=batch_size, loci_batchsize=1, loci_gen="random",#["chr19", "chr20"], 
        bios_min_exp_avail_threshold=min_avail, check_completeness=True)
    
    model_name = f"EpiDenoise30{arch}_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters30{arch}_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    if arch in ["a", "b", "d"]:
        criterion = ComboLoss_NBNLL_msk()
    elif arch in ["c"]:
        criterion = MatrixFactor_NBLL()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_30(
        num_epochs=epochs, mask_percentage=mask_percentage, 
        context_length=context_length, batch_size=batch_size, inner_epochs=inner_epochs, arch=arch)

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

def train_epd30_synthdata(hyper_parameters, arch="a"):
    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]
    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    metadata_embedding_dim = hyper_parameters["metadata_embedding_dim"]
    
    resolution = 25
    epochs = hyper_parameters["epochs"]
    num_training_loci = hyper_parameters["num_loci"]
    mask_percentage = hyper_parameters["mask_percentage"]
    context_length = hyper_parameters["context_length"]
    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    lr_halflife = hyper_parameters["lr_halflife"]
    min_avail = hyper_parameters["min_avail"]
    inner_epochs = hyper_parameters["inner_epochs"]

    # end of hyperparameters
    if arch == "a":
        model = EpiDenoise30a(input_dim, metadata_embedding_dim, nhead, d_model, nlayers, output_dim, 
            dropout=dropout, context_length=context_length, pos_enc="relative")
            
    elif arch == "b":
        n_cnn_layers = hyper_parameters["n_cnn_layers"]
        conv_kernel_size = hyper_parameters["conv_kernel_size"]
        n_decoder_layers = hyper_parameters["n_decoder_layers"]

        model = EpiDenoise30b(input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, n_decoder_layers,
        dropout=dropout, context_length=context_length, pos_enc="relative")

    elif arch == "c":
        n_cnn_layers = hyper_parameters["n_cnn_layers"]
        conv_kernel_size = hyper_parameters["conv_kernel_size"]
        pool_size = hyper_parameters["pool_size"]

        model = EpiDenoise30c(input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, pool_size, 
        dropout=dropout, context_length=context_length)

    elif arch == "d":
        pass
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_halflife, gamma=1)
    scheduler = None

    print(f"EPD30{arch} # model_parameters: {count_parameters(model)}")

    print(f"initializing data synthesis...")
    dataset = SyntheticData(n=0.1485, p=0.0203, num_features=input_dim, sequence_length=context_length)

    if arch in ["a", "b"]:
        criterion = ComboLoss_NBNLL_msk()
    elif arch in ["c", "d"]:
        criterion = MatrixFactor_NBLL()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epd30_synth(
        num_epochs=epochs, mask_percentage=mask_percentage, 
        context_length=context_length, batch_size=batch_size, 
        inner_epochs=inner_epochs, arch=arch)

    end_time = time.time()
    print(f"took {end_time - start_time}")

    # # Save the trained model
    # model_dir = "models/"
    # os.makedirs(model_dir, exist_ok=True)
    # torch.save(model.state_dict(), os.path.join(model_dir, model_name))

    # # Write a description text file
    # description = {
    #     "hyper_parameters": hyper_parameters,
    #     "model_architecture": str(model),
    #     "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    #     "number_of_model_parameters": count_parameters(model),
    #     "training_duration": int(end_time - start_time)
    # }
    # with open(os.path.join(model_dir, model_name.replace(".pt", ".txt")), 'w') as f:
    #     f.write(json.dumps(description, indent=4))

    # return model

def train_epd30_eic(hyper_parameters, checkpoint_path=None, arch="d"):
    # Defining the hyperparameters
    data_path = hyper_parameters["data_path"]

    input_dim = output_dim = hyper_parameters["input_dim"]
    dropout = hyper_parameters["dropout"]
    nhead = hyper_parameters["nhead"]
    d_model = hyper_parameters["d_model"]
    nlayers = hyper_parameters["nlayers"]
    metadata_embedding_dim = hyper_parameters["metadata_embedding_dim"]
    
    resolution = 25
    epochs = hyper_parameters["epochs"]
    num_training_loci = hyper_parameters["num_loci"]
    mask_percentage = hyper_parameters["mask_percentage"]
    context_length = hyper_parameters["context_length"]
    batch_size = hyper_parameters["batch_size"]
    learning_rate = hyper_parameters["learning_rate"]
    lr_halflife = hyper_parameters["lr_halflife"]
    min_avail = hyper_parameters["min_avail"]
    inner_epochs = hyper_parameters["inner_epochs"]

    # end of hyperparameters
    if arch == "a":
        model = EpiDenoise30a(input_dim, metadata_embedding_dim, nhead, d_model, nlayers, output_dim, 
            dropout=dropout, context_length=context_length, pos_enc="relative")
            
    elif arch == "b":
        n_cnn_layers = hyper_parameters["n_cnn_layers"]
        conv_kernel_size = hyper_parameters["conv_kernel_size"]
        n_decoder_layers = hyper_parameters["n_decoder_layers"]

        model = EpiDenoise30b(input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, output_dim, n_decoder_layers,
        nlayers=nlayers, dropout=dropout, context_length=context_length, pos_enc="relative")

    elif arch == "c":
        n_cnn_layers = hyper_parameters["n_cnn_layers"]
        conv_kernel_size = hyper_parameters["conv_kernel_size"]
        pool_size = hyper_parameters["pool_size"]

        model = EpiDenoise30c(input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, pool_size, 
        dropout=dropout, context_length=context_length)
    
    elif arch == "d":
        n_cnn_layers = hyper_parameters["n_cnn_layers"]
        conv_kernel_size = hyper_parameters["conv_kernel_size"]
        pool_size = hyper_parameters["pool_size"]

        model = EpiDenoise30d(input_dim, metadata_embedding_dim, conv_kernel_size, 
        n_cnn_layers, nhead, d_model, nlayers, output_dim, pool_size = pool_size,
        dropout=dropout, context_length=context_length, pos_enc="relative")

    optimizer = optim.Adamax(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_halflife, gamma=0.5)
    # scheduler = None

    if checkpoint_path is not None:
        print("loading pretrained model...")
        model.load_state_dict(torch.load(checkpoint_path))

    print(f"EPD30{arch} # model_parameters: {count_parameters(model)}")
    summary(model)

    dataset = ExtendedEncodeDataHandler(data_path)
    dataset.initialize_EED(
        m=num_training_loci, context_length=context_length*resolution, 
        bios_batchsize=batch_size, loci_batchsize=1, loci_gen="random",
        bios_min_exp_avail_threshold=1, check_completeness=False, 
        eic=True)
    
    model_name = f"EpiDenoise30{arch}_eic_{datetime.now().strftime('%Y%m%d%H%M%S')}_params{count_parameters(model)}.pt"
    with open(f'models/hyper_parameters30{arch}_eic_{model_name.replace(".pt", ".pkl")}', 'wb') as f:
        pickle.dump(hyper_parameters, f)

    if arch in ["a", "b", "d"]:
        criterion = ComboLoss_NBNLL_msk()
    elif arch in ["c"]:
        criterion = MatrixFactor_NBLL()

    start_time = time.time()

    trainer = PRE_TRAINER(model, dataset, criterion, optimizer, scheduler)
    model = trainer.pretrain_epidenoise_30_eic(
        num_epochs=epochs, mask_percentage=mask_percentage, 
        context_length=context_length, batch_size=batch_size, inner_epochs=inner_epochs, arch=arch)

    end_time = time.time()

    # Save the trained model
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))

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

class CANDI_DNA_Encoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", expansion_factor=3):
        super(CANDI_DNA_Encoder, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        d_model = self.f2
        self.latent_dim = self.f2

        DNA_conv_channels = exponential_linspace_int(4, self.f2, n_cnn_layers+3)
        DNA_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers+2)]

        self.convEncDNA = nn.ModuleList(
            [ConvTower(
                DNA_conv_channels[i], DNA_conv_channels[i + 1],
                DNA_kernel_size[i], S=1, D=1,
                pool_type="max", residuals=True,
                groups=1, pool_size=5 if i >= n_cnn_layers else pool_size) for i in range(n_cnn_layers + 2)])

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [expansion_factor * x for x in conv_channels[::-1]]
        conv_kernel_size_list = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                conv_kernel_size_list[i], S=1, D=1,
                pool_type="avg", residuals=True,
                groups=self.f1,
                pool_size=pool_size) for i in range(n_cnn_layers)])
        
        self.xmd_emb = EmbedMetadata(self.f1, metadata_embedding_dim, non_linearity=True)
        self.xmd_fusion = nn.Sequential(
            nn.Linear(self.f3, self.f2),
            nn.LayerNorm(self.f2), 
            nn.ReLU())

        self.DNA_Epig_fusion = nn.Sequential(
            nn.Linear(2*self.f2, self.f2), 
            nn.LayerNorm(self.f2), 
            nn.ReLU())

        if self.pos_enc == "relative":
            self.transformer_encoder = nn.ModuleList([
                RelativeEncoderLayer(d_model=d_model, heads=nhead, feed_forward_hidden=expansion_factor*d_model, dropout=dropout) for _ in range(n_sab_layers)])
            
        else:
            self.posEnc = PositionalEncoding(d_model, dropout, self.l2)
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=expansion_factor*d_model, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_sab_layers)

    def forward(self, src, seq, x_metadata):
        if len(seq.shape) != len(src.shape):
            seq = seq.unsqueeze(0).expand(src.shape[0], -1, -1)

        seq = seq.permute(0, 2, 1)  # to N, 4, 25*L
        seq = seq.float()

        ### DNA CONV ENCODER ###
        for seq_conv in self.convEncDNA:
            seq = seq_conv(seq)
        seq = seq.permute(0, 2, 1)  # to N, L', F2

        ### SIGNAL CONV ENCODER ###
        src = src.permute(0, 2, 1) # to N, F1, L
        for conv in self.convEnc:
            src = conv(src)
        src = src.permute(0, 2, 1)  # to N, L', F2

        ### SIGNAL METADATA EMBEDDING ###
        xmd_embedding = self.xmd_emb(x_metadata)
        src = torch.cat([src, xmd_embedding.unsqueeze(1).expand(-1, self.l2, -1)], dim=-1)
        src = self.xmd_fusion(src)
        
        ### DNA EPIGENETIC SIGNAL FUSION ###
        src = torch.cat([src, seq], dim=-1)
        src = self.DNA_Epig_fusion(src)

        ### TRANSFORMER ENCODER ###
        if self.pos_enc != "relative":
            src = self.posEnc(src)
            src = self.transformer_encoder(src)
        else:
            for enc in self.transformer_encoder:
                src = enc(src)

        return src

#========================================================================================================#

class ENCODE_IMPUTATION_DATASET(object):
    def __init__(self, path):
        """
        each pkl.gz file is for one biosample and is a dictionary:
        d = {
            "assay1":[list of several pairs of ([chr, start, end], [signal_array]) ],
            "assay2":[list of several pairs of ([chr, start, end], [signal_array]) ],
            "assay3":[list of several pairs of ([chr, start, end], [signal_array]) ],
        }

        let's say we have A assays, and M sample ( len(d["assay1"])=M ).
        signal_arrays are of the same length and for all assays, signal_array[i] corresponds to the same genomic position. 
        if we have M pairs of ([chr, start, end], [signal_array]) for each assay, we will have M samples of size: (len(signal_array), number_of_assays)
        """

        self.path = path
        self.all_assays = ['M{:02d}'.format(i) for i in range(1, 36)]
        self.all_ct = ['C{:02d}'.format(i) for i in range(1, 52)]

        availability = {}
        for f in os.listdir(self.path):
            if ".bigwig" in f: 
                if f[:3] not in availability.keys():
                    availability[f[:3]] = 0
                availability[f[:3]] += 1
                

        self.biosamples = {}
        for f in os.listdir(self.path):
            if ".pkl.gz" in f: 
                self.biosamples[f[:3]] = f"{self.path}/{f}"
        
        # Sort the keys in availability in descending order
        sorted_keys = sorted(availability, key=availability.get, reverse=True)

        # Create a new dictionary with sorted keys
        self.biosamples = {key: self.biosamples[key] for key in sorted_keys if key in self.biosamples}

        self.preprocessed_datasets = []
        for f in os.listdir(self.path):
            if ".pt" in f and "mixed_dataset" in f: 
                self.preprocessed_datasets.append(f"{self.path}/{f}")
        
    def get_biosample_pkl(self, pkl_path):
        with gzip.open(pkl_path, 'rb') as f:
            loaded_file = pickle.load(f)

        bios_assays = loaded_file.keys()
        assay_availability = {ass: (True if ass in bios_assays else False) for ass in self.all_assays}

        M = len(loaded_file[list(loaded_file.keys())[0]])
        L = len(loaded_file[list(loaded_file.keys())[0]][0][1])
        D = len(self.all_assays)

        missing_f_i = []
        # Initialize an empty list to hold all samples
        all_samples = []
        
        # Iterate over all assays
        for i, assay in enumerate(self.all_assays):
            if assay_availability[assay]:
                # If assay is available, append its signal arrays to all_samples
                assay_samples = []
                for j in range(len(loaded_file[assay])):
                    assay_samples.append(loaded_file[assay][j][1])

            else:
                missing_f_i.append(i)
                # If assay is not available, append -1 of appropriate shape
                assay_samples = []
                for j in range(M):
                    assay_samples.append([-1 for _ in range(L)])
            
            all_samples.append(assay_samples)

        # Convert all_samples to a numpy array and transpose to get shape (M, L, D)
        all_samples_tensor = np.array(all_samples, dtype=np.float32).transpose(1, 2, 0)

        # Convert numpy array to PyTorch tensor
        all_samples_tensor = torch.from_numpy(all_samples_tensor)
        all_samples_tensor = all_samples_tensor.float() 

        # Create a mask tensor
        mask = (all_samples_tensor == -1)

        return all_samples_tensor, mask, missing_f_i
    
    def get_dataset_pt(self, pt_path):
        ds = torch.load(pt_path)
        mask = (ds == -1)
        mask_2 = (ds.sum(dim=1) < 0) # missing assay pattern per sample

        indices = [torch.nonzero(mask_2[i, :], as_tuple=True)[0].tolist() for i in range(mask_2.shape[0])]

        unique_indices = [list(x) for x in set(tuple(x) for x in indices)]
        pattern_dict = {tuple(pattern): [] for pattern in unique_indices}

        for i, pattern in enumerate(indices):
            pattern_dict[tuple(pattern)].append(i)

        return ds, mask, pattern_dict
       

       
class DataMasker:
    def __init__(self, mask_value, mask_percentage, chunk_size=5, prog=False):
        self.mask_value = mask_value
        self.mask_percentage = mask_percentage
        self.chunk_size = chunk_size

    def mask_chunks(self, data):
        data = data.clone()
        N, L, F = data.size()
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        num_masks = int((L * self.mask_percentage) / self.chunk_size)
        for _ in range(num_masks):
            start = random.randint(0, L - self.chunk_size)
            mask_indicator[:, start:start+self.chunk_size, :] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mask_features(self, data, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)

        if len(available_features) == 1:
            return data, mask_indicator
        else:
            num_features_to_mask = int(len(self.available_features) * self.mask_percentage) + 1

        features_to_mask = random.sample(self.available_features, num_features_to_mask)

        for feature in features_to_mask:
            mask_indicator[:, :, feature] = True

        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mask_feature30(self, data, metadata, availability):
        B, L, F = data.shape

        # Number of features to mask per sample in the batch
        num_to_mask = []
        num_available = availability.sum(dim=1)
        for b in range(B):
            if num_available[b] == 1:
                num_to_mask.append(0)
            else:
                num_to_mask.append(max(1, int(num_available[b] * self.mask_percentage)))

        # Prepare the new availability tensor
        new_A = availability.clone().float()
        new_md = metadata.clone().float()
        data = data.clone().float()

        # Mask indices generation and masking operation
        for b in range(B):
            if num_to_mask[b] > 0:
                available_indices = torch.where(availability[b] == 1)[0]  # Find indices where features are available
                mask_indices = torch.randperm(available_indices.size(0))[:num_to_mask[b]]  # Randomly select indices to mask
                actual_indices_to_mask = available_indices[mask_indices]  # Actual indices in the feature dimension

                data[b, :, actual_indices_to_mask] = self.mask_value  # Mask the features in X
                new_md[b, :, actual_indices_to_mask] = self.mask_value
                new_A[b, actual_indices_to_mask] = self.mask_value  # Update the availability tensor to indicate masked features

        return data, new_md, new_A

    def progressive(self, data, metadata, availability, num_mask):
        B, L, F = data.shape

        # Number of features to mask per sample in the batch
        num_to_mask = []
        num_available = availability.sum(dim=1)
        for b in range(B):

            if num_available[b] > num_mask:
                num_to_mask.append(num_mask)

            else:
                num_to_mask.append(num_available[b] - 1)

        # Prepare the new availability tensor
        new_A = availability.clone().float()
        new_md = metadata.clone().float()
        data = data.clone().float()

        # Mask indices generation and masking operation
        for b in range(B):
            if num_to_mask[b] > 0:
                available_indices = torch.where(availability[b] == 1)[0]  # Find indices where features are available
                mask_indices = torch.randperm(available_indices.size(0))[:num_to_mask[b]]  # Randomly select indices to mask
                actual_indices_to_mask = available_indices[mask_indices]  # Actual indices in the feature dimension

                data[b, :, actual_indices_to_mask] = self.mask_value  # Mask the features in X
                new_md[b, :, actual_indices_to_mask] = self.mask_value
                new_A[b, actual_indices_to_mask] = self.mask_value  # Update the availability tensor to indicate masked features

        return data, new_md, new_A
    
    def mask_chunk_features_30(self, data, metadata, availability):
        B, L, F = data.shape

        # Prepare the new availability tensor
        new_A = availability.clone().float()
        new_md = metadata.clone().float()
        data = data.clone().float()

        # Calculate the total number of signals and chunks to mask per batch sample
        num_all_signals = L * availability.sum(dim=1)
        num_masks = (num_all_signals * self.mask_percentage / self.chunk_size).int()

        # Masking operation for each sample in the batch
        for b in range(B):
            for _ in range(num_masks[b]):
                # Select a random chunk start and feature index
                length_start = random.randint(0, L - self.chunk_size)
                available_indices = torch.where(availability[b] == 1)[0]
                if len(available_indices) == 0:
                    continue
                feature_start = random.choice(available_indices)
                
                # Apply the mask to the data, metadata, and update availability
                data[b, length_start:length_start+self.chunk_size, feature_start] = self.mask_value
                # new_md[b, length_start:length_start+self.chunk_size, feature_start] = self.mask_value
                # new_A[b, feature_start] = 0  # Update the availability to indicate masked feature

        return data, new_md, new_A

    def mask_chunk_features(self, data, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        num_all_signals = L * len(self.available_features)
        num_masks = int((num_all_signals * self.mask_percentage) / self.chunk_size)
        for _ in range(num_masks):
            length_start = random.randint(0, L - self.chunk_size)
            feature_start = random.choice(self.available_features)
            mask_indicator[:, length_start:length_start+self.chunk_size, feature_start] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mid_slice_mask(self, data, available_features):
        data = data.clone()
        N, L, F = data.size()
        slice_length = int(L * self.mask_percentage)
        start = L // 2 - slice_length // 2
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        mask_indicator[:, start:start+slice_length, available_features] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mid_slice_mask_features(self, data, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()
        slice_length = int(L * self.mask_percentage)
        num_features_to_mask = int(len(self.available_features) * self.mask_percentage)
        features_to_mask = random.sample(self.available_features, num_features_to_mask)
        start = L // 2 - slice_length // 2
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)
        for feature in features_to_mask:
            mask_indicator[:, start:start+slice_length, feature] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator

    def mid_slice_focused_full_feature_mask(self, data, missing_mask_value, available_features):
        self.available_features = available_features

        data = data.clone()
        N, L, F = data.size()

        num_features_to_mask = int(len(self.available_features) * self.mask_percentage)
        if num_features_to_mask == 0:
            features_to_mask = available_features
        else:
            features_to_mask = random.sample(self.available_features, num_features_to_mask)
        mask_indicator = torch.zeros_like(data, dtype=torch.bool)

        # Mask features completely
        for feature in features_to_mask:
            data[:, :, feature] = missing_mask_value

        # Mark only the middle part of those masked features
        slice_length = int(L * self.mask_percentage)
        start = L // 2 - slice_length // 2
        mask_indicator[:, start:start+slice_length, features_to_mask] = True
        data[mask_indicator] = self.mask_value
        return data, mask_indicator


class EmbedMetadata(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_sequencing_platforms=10, num_runtypes=4, non_linearity=True):
        """
        Args:
            input_dim (int): Number of metadata features.
            embedding_dim (int): Final embedding dimension.
            num_sequencing_platforms (int): Number of sequencing platforms in the data.
            num_runtypes (int): Number of run types in the data.
            non_linearity (bool): Whether to apply ReLU at the end.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim 
        self.non_linearity = non_linearity
        # We divide the embedding_dim into 4 parts for continuous types.
        # (You can adjust the splitting scheme as needed.)
        self.continuous_size = embedding_dim // 4

        # For each feature (total input_dim features), create a separate linear transform.
        self.depth_transforms = nn.ModuleList(
            [nn.Linear(1, self.continuous_size) for _ in range(input_dim)]
        )
        # For sequencing platform, create separate embedding layers per feature.
        # Use dynamic size based on actual data
        self.sequencing_platform_embeddings = nn.ModuleList(
            [nn.Embedding(num_sequencing_platforms, self.continuous_size) for _ in range(input_dim)]
        )
        self.read_length_transforms = nn.ModuleList(
            [nn.Linear(1, self.continuous_size) for _ in range(input_dim)]
        )
        # For runtype, create separate embedding layers per feature.
        # Use dynamic size based on actual data
        self.runtype_embeddings = nn.ModuleList(
            [nn.Embedding(num_runtypes, self.continuous_size) for _ in range(input_dim)]
        )

        # Final projection: the concatenated vector for each feature will be of size 4*continuous_size.
        # For all features, that becomes input_dim * 4 * continuous_size.
        self.final_embedding = nn.Linear(input_dim * 4 * self.continuous_size, embedding_dim)
        self.final_emb_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, metadata):
        """
        Args:
            metadata: Tensor of shape (B, 4, input_dim)
                      where dimension 1 indexes the four metadata types in the order:
                      [depth, sequencing_platform, read_length, runtype]
        Returns:
            embeddings: Tensor of shape (B, embedding_dim)
        """
        B = metadata.size(0)
        # Lists to collect per-feature embeddings.
        per_feature_embeds = []
        for i in range(self.input_dim):
            # Extract each metadata type for feature i.
            depth = metadata[:, 0, i].unsqueeze(-1).float() 
            sequencing_platform = metadata[:, 1, i].long() 
            read_length = metadata[:, 2, i].unsqueeze(-1).float() 
            runtype = metadata[:, 3, i].long() 
            
            # For runtype, map -1 -> 2 (missing) and -2 -> 3 (cloze_masked)
            runtype = torch.where(runtype == -1, torch.tensor(2, device=runtype.device), runtype)
            runtype = torch.where(runtype == -2, torch.tensor(3, device=runtype.device), runtype)
            
            # For sequencing platform, map -1 -> 2 (missing) and -2 -> 3 (cloze_masked)
            sequencing_platform = torch.where(sequencing_platform == -1, torch.tensor(2, device=sequencing_platform.device), sequencing_platform)
            sequencing_platform = torch.where(sequencing_platform == -2, torch.tensor(3, device=sequencing_platform.device), sequencing_platform)
            
            # Apply the separate transforms/embeddings for feature i.
            depth_embed = self.depth_transforms[i](depth)              # (B, continuous_size)
            sequencing_platform_embed = self.sequencing_platform_embeddings[i](sequencing_platform)  # (B, continuous_size)
            read_length_embed = self.read_length_transforms[i](read_length)  # (B, continuous_size)
            runtype_embed = self.runtype_embeddings[i](runtype)           # (B, continuous_size)
            
            # Concatenate the four embeddings along the last dimension.
            feature_embed = torch.cat([depth_embed, sequencing_platform_embed, read_length_embed, runtype_embed], dim=-1)  # (B, 4*continuous_size)
            per_feature_embeds.append(feature_embed)
        
        # Now stack along a new dimension for features -> shape (B, input_dim, 4*continuous_size)
        embeddings = torch.stack(per_feature_embeds, dim=1)
        # Flatten feature dimension: (B, input_dim * 4*continuous_size)
        embeddings = embeddings.view(B, -1)
        # Project to final embedding dimension.
        embeddings = self.final_embedding(embeddings)
        embeddings = self.final_emb_layer_norm(embeddings)
        
        if self.non_linearity:
            embeddings = F.relu(embeddings)
        
        return embeddings
