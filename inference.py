
# sequence clustering
class CANDIPredictor:
    def __init__(self, model, hyper_parameters_path, number_of_states, 
        split="test", DNA=False, eic=True, chr="chr21", resolution=25, context_length=1600,
        savedir="models/output/", data_path="/project/compbio-lab/encode_data/"):

        self.model = model
        self.number_of_states = number_of_states
        self.chr = chr
        self.resolution = resolution
        self.savedir = savedir
        self.DNA = DNA
        self.context_length = context_length
        self.data_path = data_path
        self.eic = eic
        self.split = split

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        
        self.dataset.init_eval(
            self.context_length, check_completeness=True, split=split, bios_min_exp_avail_threshold=5, eic=eic)

        if isinstance(self.model, str):
            with open(hyper_parameters_path, 'rb') as f:
                self.hyper_parameters = pickle.load(f)
                self.hyper_parameters["signal_dim"] = self.dataset.signal_dim
                self.hyper_parameters["metadata_embedding_dim"] = self.dataset.signal_dim
                self.context_length = self.hyper_parameters["context_length"]

            loader = CANDI_LOADER(model, self.hyper_parameters, DNA=self.DNA)
            self.model = loader.load_CANDI()

        self.model = self.model.to(self.device)
        self.model.eval()

        self.chr_sizes = {}
        with open("data/hg38.chrom.sizes", 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name == self.chr:
                    self.chr_sizes[chr_name] = int(chr_size)
                    break

        self.context_length = self.hyper_parameters["context_length"]
        self.batch_size = self.hyper_parameters["batch_size"]
        self.token_dict = {
            "missing_mask": -1, 
            "cloze_mask": -2,
            "pad": -3
            }

    def load_bios(self, bios_name, x_dsf, y_dsf=1, fill_in_y_prompt=True):
        # Load biosample data
        
        print(f"getting bios vals for {bios_name}")

        if self.eic:
            if self.split == "test":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("B_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            elif self.split == "val":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("V_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            
            # print(temp_x.keys(), temp_mx.keys())
            X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
            del temp_x, temp_mx
            
            temp_y, temp_my = self.dataset.load_bios(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
            if fill_in_y_prompt:
                mY = self.dataset.fill_in_y_prompt(mY)
            del temp_y, temp_my

            temp_py = self.dataset.load_bios_BW(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            if self.split == "test":
                temp_px = self.dataset.load_bios_BW(bios_name.replace("B_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            elif self.split == "val":
                temp_px = self.dataset.load_bios_BW(bios_name.replace("V_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)

            temp_p = {**temp_py, **temp_px}
            P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
            del temp_py, temp_px, temp_p

        else:
            temp_x, temp_mx = self.dataset.load_bios(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            X, mX, avX = self.dataset.make_bios_tensor(temp_x, temp_mx)
            del temp_x, temp_mx
            
            temp_y, temp_my = self.dataset.load_bios(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            Y, mY, avY = self.dataset.make_bios_tensor(temp_y, temp_my)
            if fill_in_y_prompt:
                mY = self.dataset.fill_in_y_prompt(mY)
            del temp_y, temp_my

            temp_p = self.dataset.load_bios_BW(bios_name, [self.chr, 0, self.chr_sizes[self.chr]], y_dsf)
            P, avlP = self.dataset.make_bios_tensor_BW(temp_p)
            assert (avlP == avY).all(), "avlP and avY do not match"
            del temp_p

        if self.DNA:
            seq = dna_to_onehot(get_DNA_sequence(self.chr, 0, self.chr_sizes[self.chr]))

        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y, P = X[:num_rows, :], Y[:num_rows, :], P[:num_rows, :]

        if self.DNA:
            seq = seq[:num_rows*self.resolution, :]
            
        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])
        P = P.view(-1, self.context_length, P.shape[-1])

        if self.DNA:
            seq = seq.view(-1, self.context_length*self.resolution, seq.shape[-1])

        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)

        if self.DNA:
            return X, Y, P, seq, mX, mY, avX, avY
        else:
            return X, Y, P, mX, mY, avX, avY

    def pred(self, X, mX, mY, avail, imp_target=[], seq=None):
        n = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        mu = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        var = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        Z = torch.empty((X.shape[0], self.model.l2, self.model.latent_dim), device="cpu", dtype=torch.float32)

        for i in range(0, len(X), self.batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+ self.batch_size]
            mX_batch = mX[i:i+ self.batch_size]
            mY_batch = mY[i:i+ self.batch_size]
            avail_batch = avail[i:i+ self.batch_size]

            if self.DNA:
                seq_batch = seq[i:i + self.batch_size]

            with torch.no_grad():
                x_batch = x_batch.clone()
                avail_batch = avail_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()

                x_batch_missing_vals = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing_vals = (mX_batch == self.token_dict["missing_mask"])
                avail_batch_missing_vals = (avail_batch == 0)

                x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing_vals] = self.token_dict["cloze_mask"]

                if len(imp_target)>0:
                    x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    avail_batch[:, imp_target] = 0

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                if self.DNA:
                    seq_batch = seq_batch.to(self.device)
                    outputs_p, outputs_n, outputs_mu, outputs_var, latent = self.model(x_batch.float(), seq_batch, mX_batch, mY_batch, avail_batch, return_z=True)
                else:
                    outputs_p, outputs_n, outputs_mu, outputs_var, latent = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch, return_z=True)

            n[i:i+outputs_n.shape[0], :, :] = outputs_n.cpu()
            p[i:i+outputs_p.shape[0], :, :] = outputs_p.cpu()
            mu[i:i+outputs_mu.shape[0], :, :] = outputs_mu.cpu()
            var[i:i+outputs_var.shape[0], :, :] = outputs_var.cpu()
            Z[i:i+latent.shape[0], :, :] = latent.cpu()

            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n, outputs_mu, outputs_var, latent
            torch.cuda.empty_cache()
            
        n = n.view(n.shape[0] * n.shape[1], n.shape[-1])
        p = p.view(p.shape[0] * p.shape[1], p.shape[-1])
        mu = mu.view(mu.shape[0] * mu.shape[1], mu.shape[-1])
        var = var.view(var.shape[0] * var.shape[1], var.shape[-1])
        Z = Z.view(Z.shape[0] * Z.shape[1], Z.shape[-1])
        return n, p, mu, var, Z

    def pred_cropped(self, X, mX, mY, avail, imp_target=[], seq=None):
        """
        given 
        """
        pass

    def get_latent_representations(self, X, mX, mY, avX, seq=None):
        if self.DNA:
            _, _, _, _, Z = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            _, _, _, _, Z = self.pred(X, mX, mY, avX, seq=None, imp_target=[])
        return Z

    def get_decoded_signal(self, X, mX, mY, avX, seq=None):
        if self.DNA:
            p, n, mu, var, _ = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            p, n, mu, var, _ = self.pred(X, mX, mY, avX, seq=None, imp_target=[])

        count_dist = NegativeBinomial(p, n)
        pval_dist = Gaussian(mu, var)
        
        return count_dist.mean(), pval_dist.mean()

    def latent_position_dependency_experiment(self, bios_name, n_positions=10):
        # Load the data
        print("Loading data...")
        if self.DNA:
            X, Y, P, seq, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf=1)
        else:
            X, Y, P, mX, mY, avX, avY = self.load_bios(bios_name, x_dsf=1)
        
        # Flatten only X and seq
        X_flat = X.reshape(-1, X.shape[-1])
        if self.DNA:
            seq_flat = seq.reshape(-1, seq.shape[-1])
        total_length = X_flat.shape[0]

        # Print shapes
        print("Shapes of tensors:")
        print(f"X shape: {X.shape}")
        print(f"X_flat shape: {X_flat.shape}")
        print(f"mX shape: {mX.shape}")
        print(f"mY shape: {mY.shape}")
        print(f"avX shape: {avX.shape}")
        if self.DNA:
            print(f"seq shape: {seq.shape}")
            print(f"seq_flat shape: {seq_flat.shape}")

        # Randomly select positions avoiding edges
        positions = np.random.randint(self.context_length//2, total_length - self.context_length//2, size=n_positions)
        offsets = np.arange(-self.context_length//2, self.context_length//2 + 1, (self.model.l1 // self.model.l2)*5)

        # Initialize arrays to store distances
        cosine_distances = np.zeros((n_positions, len(offsets)))
        euclidean_distances = np.zeros((n_positions, len(offsets)))
        manhattan_distances = np.zeros((n_positions, len(offsets)))

        for idx, pos in enumerate(tqdm(positions, desc='Processing positions')):
            # Reference context window centered at pos
            start = pos - self.context_length//2
            end = pos + self.context_length//2
            X_ref = X_flat[start:end]
            mX_ref = mX[0].unsqueeze(0)
            mY_ref = mY[0].unsqueeze(0)
            avX_ref = avX[0].unsqueeze(0)
            if self.DNA:
                seq_ref = seq_flat[(start*self.resolution):(end*self.resolution)]
            
            # Expand dims for X_ref
            X_ref = X_ref.unsqueeze(0)
            if self.DNA:
                seq_ref = seq_ref.unsqueeze(0)
            
            # Get reference latent representation
            if self.DNA:
                Z_ref = self.get_latent_representations(X_ref, mX_ref, mY_ref, avX_ref, seq=seq_ref)
            else:
                Z_ref = self.get_latent_representations(X_ref, mX_ref, mY_ref, avX_ref, seq=None)

            # Position of pos within context window
            pos_in_window = int((pos - start) * (self.model.l2 / self.model.l1))
            # print(f"ref position in window {pos_in_window}")
            Z_ref_pos = Z_ref[pos_in_window].cpu().numpy()

            # Iterate over offsets
            for i, offset in enumerate(offsets):
                new_pos = pos + offset
                start = new_pos - self.context_length//2
                end = new_pos + self.context_length//2
                if start < 0 or end > total_length:
                    cosine_distances[idx, i] = np.nan
                    euclidean_distances[idx, i] = np.nan
                    manhattan_distances[idx, i] = np.nan
                    continue
                X_window = X_flat[start:end]
                if self.DNA:
                    seq_window = seq_flat[(start*self.resolution):(end*self.resolution)]
                
                X_window = X_window.unsqueeze(0)
                if self.DNA:
                    seq_window = seq_window.unsqueeze(0)
                
                if self.DNA:
                    Z = self.get_latent_representations(X_window, mX_ref, mY_ref, avX_ref, seq=seq_window)
                else:
                    Z = self.get_latent_representations(X_window, mX_ref, mY_ref, avX_ref, seq=None)
                
                # Position of pos within the shifted context window
                pos_in_window_shifted = int((pos - start) * (self.model.l2 / self.model.l1))
                if pos_in_window_shifted == self.model.l2:
                    pos_in_window_shifted -= 1
                # print(f"shifted position in window {pos_in_window_shifted}")
                Z_pos = Z[pos_in_window_shifted].cpu().numpy()
                
                # Compute distances
                cosine_distances[idx, i] = cosine(Z_ref_pos, Z_pos)
                euclidean_distances[idx, i] = euclidean(Z_ref_pos, Z_pos)
                manhattan_distances[idx, i] = cityblock(Z_ref_pos, Z_pos)

        # Plotting
        # Line plots for each position
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        for idx in range(n_positions):
            axes[0].plot(offsets, cosine_distances[idx], label=f'Pos {idx+1}')
            axes[1].plot(offsets, euclidean_distances[idx], label=f'Pos {idx+1}')
            axes[2].plot(offsets, manhattan_distances[idx], label=f'Pos {idx+1}')
        axes[0].set_title('Cosine Distance vs. Position in Context Window')
        axes[0].set_xlabel('Offset from Center')
        axes[0].set_ylabel('Cosine Distance')
        axes[1].set_title('Euclidean Distance vs. Position in Context Window')
        axes[1].set_xlabel('Offset from Center')
        axes[1].set_ylabel('Euclidean Distance')
        axes[2].set_title('Manhattan Distance vs. Position in Context Window')
        axes[2].set_xlabel('Offset from Center')
        axes[2].set_ylabel('Manhattan Distance')
        # for ax in axes:
        #     ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.savedir, 'latent_position_dependency_lineplots.png'))
        plt.close()

        # Heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        sns.heatmap(cosine_distances, ax=axes[0], cmap='viridis', xticklabels=offsets.astype(str))
        axes[0].set_title('Cosine Distance Heatmap')
        axes[0].set_xlabel('Offset from Center')
        axes[0].set_ylabel('Position Index')
        sns.heatmap(euclidean_distances, ax=axes[1], cmap='viridis', xticklabels=offsets.astype(str))
        axes[1].set_title('Euclidean Distance Heatmap')
        axes[1].set_xlabel('Offset from Center')
        axes[1].set_ylabel('Position Index')
        sns.heatmap(manhattan_distances, ax=axes[2], cmap='viridis', xticklabels=offsets.astype(str))
        axes[2].set_title('Manhattan Distance Heatmap')
        axes[2].set_xlabel('Offset from Center')
        axes[2].set_ylabel('Position Index')
        plt.tight_layout()
        plt.savefig(os.path.join(self.savedir, 'latent_position_dependency_heatmaps.png'))
        plt.close()

        # Average distances with error bars
        cosine_mean = np.nanmean(cosine_distances, axis=0)
        cosine_std = np.nanstd(cosine_distances, axis=0)
        euclidean_mean = np.nanmean(euclidean_distances, axis=0)
        euclidean_std = np.nanstd(euclidean_distances, axis=0)
        manhattan_mean = np.nanmean(manhattan_distances, axis=0)
        manhattan_std = np.nanstd(manhattan_distances, axis=0)
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        axes[0].errorbar(offsets, cosine_mean, yerr=cosine_std, fmt='-o')
        axes[0].set_title('Average Cosine Distance vs. Position in Context Window')
        axes[0].set_xlabel('Offset from Center')
        axes[0].set_ylabel('Cosine Distance')
        axes[1].errorbar(offsets, euclidean_mean, yerr=euclidean_std, fmt='-o')
        axes[1].set_title('Average Euclidean Distance vs. Position in Context Window')
        axes[1].set_xlabel('Offset from Center')
        axes[1].set_ylabel('Euclidean Distance')
        axes[2].errorbar(offsets, manhattan_mean, yerr=manhattan_std, fmt='-o')
        axes[2].set_title('Average Manhattan Distance vs. Position in Context Window')
        axes[2].set_xlabel('Offset from Center')
        axes[2].set_ylabel('Manhattan Distance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.savedir, 'latent_position_dependency_average.png'))
        plt.close()

if __name__ == "__main__":
    model_path = "models/CANDIeic_DNA_random_mask_oct17-expan2_model_checkpoint_epoch5.pth"
    hyper_parameters_path = "models/hyper_parameters_eic_DNA_random_mask_oct17-expan2_CANDIeic_DNA_random_mask_oct17-expan2_20241017130209_params14059878.pkl"
    dataset_path = "/project/compbio-lab/encode_data/"
    output_dir = "output"
    number_of_states = 10
    DNA = True

    CANDIP = CANDIPredictor(
        model_path, hyper_parameters_path, number_of_states, data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25)
    
    os.makedirs(output_dir, exist_ok=True)

    if DNA:
        X, Y, P, seq, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
    else:
        X, Y, P, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
        seq = None
        
    print(X.shape, Y.shape, P.shape)
    print(mX.shape, mY.shape, avX.shape, avY.shape)
