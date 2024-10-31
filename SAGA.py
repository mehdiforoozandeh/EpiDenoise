# Standard library imports
import os
import sys
from tqdm import tqdm
from scipy.spatial.distance import cosine, euclidean, cityblock
import seaborn as sns


# Third-party imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap
from intervaltree import IntervalTree
from hmmlearn import hmm

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Local imports
from CANDI import *

# SAGA means segmentation and genome annotation -- similar to ChromHMM or Segway

def write_bed(data, chromosome, start_position, resolution, output_file, is_posterior=False, track_name="Custom Track", track_description="Clustering Results", visibility="dense"):
    """
    Write clustering results to a BED file compatible with UCSC Genome Browser, including a header.

    Parameters:
    data (numpy.ndarray): L*k matrix of posterior probabilities or cluster assignments
    chromosome (str): Chromosome name (e.g., 'chr1')
    start_position (int): Start position of the first segment
    resolution (int): Resolution of each segment (e.g., 25 bp)
    output_file (str): Path to the output BED file
    is_posterior (bool): If True, data contains posterior probabilities. If False, contains cluster assignments.
    track_name (str): Name of the track for UCSC Genome Browser
    track_description (str): Description of the track
    visibility (str): Visibility setting for the track ('hide', 'dense', 'pack', 'squish', 'full')
    """
    import numpy as np
    import colorsys

    L = data.shape[0]

    # Collect unique clusters/states
    if is_posterior:
        clusters = np.arange(1, data.shape[1] + 1)  # States are 1-indexed
    else:
        clusters = np.unique(data)
    clusters = sorted(clusters)

    # Generate colors for each cluster/state
    def generate_colors(n):
        colors = []
        for i in range(n):
            hue = i / n  # Distribute hues evenly
            lightness = 0.5  # Medium lightness
            saturation = 0.9  # High saturation
            r_float, g_float, b_float = colorsys.hls_to_rgb(hue, lightness, saturation)
            r = int(r_float * 255)
            g = int(g_float * 255)
            b = int(b_float * 255)
            colors.append(f"{r},{g},{b}")
        return colors

    num_clusters = len(clusters)
    colors = generate_colors(num_clusters)
    cluster_color_map = dict(zip(clusters, colors))

    with open(output_file, 'w') as f:
        # Write the UCSC Genome Browser track header
        header = f'track name="{track_name}" description="{track_description}" visibility={visibility} itemRgb="On"\n'
        f.write(header)

        for i in range(L):
            start = start_position + i * resolution
            end = start + resolution

            if is_posterior:
                # For posterior probabilities, write the most likely state
                state = np.argmax(data[i]) + 1  # +1 to make states 1-indexed
            else:
                # For cluster assignments, use the assigned state directly
                state = data[i]

            # Prepare BED fields
            chrom = chromosome
            chromStart = start
            chromEnd = end
            name = str(state)
            score = '0'
            strand = '.'
            thickStart = chromStart
            thickEnd = chromEnd
            itemRgb = cluster_color_map[state]

            f.write(f"{chrom}\t{chromStart}\t{chromEnd}\t{name}\t{score}\t{strand}\t{thickStart}\t{thickEnd}\t{itemRgb}\n")

    print(f"BED file written to {output_file}")

class SequenceClustering(object):
    def __init__(self):
        self.models = {}

    def GMM(self, sequence_embedding, n_components=18, covariance_type='full', random_state=42):
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
        labels = gmm.fit_predict(sequence_embedding)
        posteriors = gmm.predict_proba(sequence_embedding)
        self.models['GMM'] = gmm
        return labels, posteriors

    def kmeans(self, sequence_embedding, n_clusters=18, random_state=42):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(sequence_embedding)
        self.models['KMeans'] = kmeans
        return labels

    def dbscan(self, sequence_embedding, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(sequence_embedding)
        self.models['DBSCAN'] = dbscan
        return labels

    def hierarchical(self, sequence_embedding, n_clusters=18, linkage='ward'):
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hierarchical.fit_predict(sequence_embedding)
        self.models['Hierarchical'] = hierarchical
        return labels
    
    def tsne_clustering(self, sequence_embedding, n_components=2, perplexity=30, n_iter=1000, random_state=42):
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
        tsne_embedding = tsne.fit_transform(sequence_embedding)
        
        # Apply KMeans on the t-SNE embedding
        kmeans = KMeans(n_clusters=18, random_state=random_state)
        labels = kmeans.fit_predict(tsne_embedding)
        
        self.models['TSNE'] = (tsne, kmeans)
        return labels

    def umap_clustering(self, sequence_embedding, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
        umap_reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
        umap_embedding = umap_reducer.fit_transform(sequence_embedding)
        
        # Apply KMeans on the UMAP embedding
        kmeans = KMeans(n_clusters=18, random_state=random_state)
        labels = kmeans.fit_predict(umap_embedding)
        
        self.models['UMAP'] = (umap_reducer, kmeans)
        return labels
    
    def HMM(self, sequence_embedding, n_components=18, random_state=42, transition_exponent=1.0):
        # Ensure the input is a 2D numpy array
        sequence_embedding = np.array(sequence_embedding)
        if len(sequence_embedding.shape) == 1:
            sequence_embedding = sequence_embedding.reshape(-1, 1)

        # Initialize the HMM model
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", random_state=random_state, verbose=True)

        # Fit the model to the data
        model.fit(sequence_embedding)

        # Modify the transition probability matrix
        model.transmat_ = np.power(model.transmat_, transition_exponent)
        model.transmat_ /= model.transmat_.sum(axis=1)[:, np.newaxis]  # Normalize rows

        # Predict the hidden states
        hidden_states = model.predict(sequence_embedding)
        posteriors = model.predict_proba(sequence_embedding)

        self.models['HMM'] = model
        return posteriors, hidden_states

    def get_model(self, model_name):
        return self.models.get(model_name, None)

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

    def pred_crop(self, X, mX, mY, avail, imp_target=[], seq=None, crop_percent = 0.1):
        """
        Predicts over the input data X, mX, mY, avail, possibly seq.
        Handles overlapping windows to ensure all positions are predicted in the center (non-edge) of some window.
        Assembles the predictions into final outputs with the same format as _OLD_pred.
        """
        # Parameters
         # You can adjust this value if needed
        crop_len = int(crop_percent * self.context_length)
        valid_len = self.context_length - 2 * crop_len
        stride = valid_len

        total_samples, context_length, feature_dim = X.shape
        latent_dim = self.model.latent_dim

        # Initialize full tensors
        n_full = torch.empty((total_samples, context_length, feature_dim), device="cpu", dtype=torch.float32)
        p_full = torch.empty((total_samples, context_length, feature_dim), device="cpu", dtype=torch.float32)
        mu_full = torch.empty((total_samples, context_length, feature_dim), device="cpu", dtype=torch.float32)
        var_full = torch.empty((total_samples, context_length, feature_dim), device="cpu", dtype=torch.float32)
        Z_full = torch.empty((total_samples, self.model.l2, latent_dim), device="cpu", dtype=torch.float32)

        for sample_idx in range(0, total_samples):
            X_sample = X[sample_idx]
            mX_sample = mX[sample_idx]
            mY_sample = mY[sample_idx]
            avail_sample = avail[sample_idx]
            if self.DNA:
                seq_sample = seq[sample_idx]

            # Generate start indices for windows
            sample_length = X_sample.shape[0]
            starts = list(range(0, sample_length - self.context_length + 1, stride))
            if starts[-1] + self.context_length < sample_length:
                starts.append(sample_length - self.context_length)
            num_windows = len(starts)

            # Initialize tensors for the sample
            n_sample = torch.full((sample_length, feature_dim), float('nan'), device="cpu", dtype=torch.float32)
            p_sample = torch.full((sample_length, feature_dim), float('nan'), device="cpu", dtype=torch.float32)
            mu_sample = torch.full((sample_length, feature_dim), float('nan'), device="cpu", dtype=torch.float32)
            var_sample = torch.full((sample_length, feature_dim), float('nan'), device="cpu", dtype=torch.float32)
            Z_sample = torch.full((sample_length, latent_dim), float('nan'), device="cpu", dtype=torch.float32)

            for start in starts:
                end = start + self.context_length
                if end > sample_length:
                    end = sample_length
                X_window = X_sample[start:end]
                mX_window = mX_sample[start:end]
                mY_window = mY_sample[start:end]
                avail_window = avail_sample[start:end]
                if self.DNA:
                    seq_window = seq_sample[start * self.resolution:end * self.resolution]

                # Determine valid positions
                window_length = end - start
                if start == 0:
                    valid_start = 0
                    valid_end = window_length - crop_len
                elif end >= sample_length:
                    valid_start = crop_len
                    valid_end = window_length
                else:
                    valid_start = crop_len
                    valid_end = window_length - crop_len

                valid_pos_in_window = torch.arange(valid_start, valid_end)
                valid_pos_in_seq = torch.arange(start + valid_start, start + valid_end)

                # Prepare batch tensors
                x_batch = X_window.unsqueeze(0)
                mX_batch = mX_window.unsqueeze(0)
                mY_batch = mY_window.unsqueeze(0)
                avail_batch = avail_window.unsqueeze(0)
                if self.DNA:
                    seq_batch = seq_window.unsqueeze(0)

                with torch.no_grad():
                    x_batch = x_batch.clone()
                    mX_batch = mX_batch.clone()
                    mY_batch = mY_batch.clone()
                    avail_batch = avail_batch.clone()

                    x_batch_missing_vals = (x_batch == self.token_dict["missing_mask"])
                    mX_batch_missing_vals = (mX_batch == self.token_dict["missing_mask"])

                    x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"]
                    mX_batch[mX_batch_missing_vals] = self.token_dict["cloze_mask"]

                    if len(imp_target) > 0:
                        x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                        mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                        avail_batch[:, :, imp_target] = 0

                    x_batch = x_batch.to(self.device)
                    mX_batch = mX_batch.to(self.device)
                    mY_batch = mY_batch.to(self.device)
                    avail_batch = avail_batch.to(self.device)

                    if self.DNA:
                        seq_batch = seq_batch.to(self.device)
                        outputs_p, outputs_n, outputs_mu, outputs_var, latent = self.model(
                            x_batch.float(), seq_batch, mX_batch, mY_batch, avail_batch, return_z=True)
                    else:
                        outputs_p, outputs_n, outputs_mu, outputs_var, latent = self.model(
                            x_batch.float(), mX_batch, mY_batch, avail_batch, return_z=True)

                output_seq_len = outputs_n.shape[1]
                input_seq_len = x_batch.shape[1]
                sequence_length_ratio = output_seq_len / input_seq_len

                adjusted_valid_pos_in_window = (valid_pos_in_window.float() * sequence_length_ratio).long()
                adjusted_valid_pos_in_window = adjusted_valid_pos_in_window.clamp(0, output_seq_len - 1)

                try:
                    n_valid = outputs_n[0, adjusted_valid_pos_in_window].cpu()
                    p_valid = outputs_p[0, adjusted_valid_pos_in_window].cpu()
                    mu_valid = outputs_mu[0, adjusted_valid_pos_in_window].cpu()
                    var_valid = outputs_var[0, adjusted_valid_pos_in_window].cpu()

                    # For latent representation
                    latent_seq_len = latent.shape[1]
                    latent_indices = (valid_pos_in_window.float() * (latent_seq_len / input_seq_len)).long()
                    latent_indices = latent_indices.clamp(0, latent_seq_len - 1)
                    Z_valid = latent[0, latent_indices].cpu()

                    # Assign predictions to sample tensors
                    n_sample[valid_pos_in_seq] = n_valid
                    p_sample[valid_pos_in_seq] = p_valid
                    mu_sample[valid_pos_in_seq] = mu_valid
                    var_sample[valid_pos_in_seq] = var_valid
                    Z_sample[valid_pos_in_seq] = Z_valid
                except IndexError as e:
                    print(f"IndexError at sample {sample_idx}, window starting at {start}: {str(e)}")
                    continue

                del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n, outputs_mu, outputs_var, latent
                torch.cuda.empty_cache()

            # Reshape Z_sample to (self.model.l2, latent_dim) by averaging over overlapping positions
            Z_sample = Z_sample.view(-1, latent_dim)
            valid_Z_positions = ~torch.isnan(Z_sample[:, 0])
            if valid_Z_positions.any():
                Z_valid = Z_sample[valid_Z_positions]
                # Assuming that each position corresponds to a latent representation
                Z_sample_final = torch.mean(Z_valid, dim=0)
            else:
                Z_sample_final = torch.zeros(latent_dim)

            Z_full[sample_idx] = Z_sample_final
            n_full[sample_idx] = n_sample
            p_full[sample_idx] = p_sample
            mu_full[sample_idx] = mu_sample
            var_full[sample_idx] = var_sample

        return n_full, p_full, mu_full, var_full, Z_full

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

#########################################################

def save_latent_representations(Z, output_file):
    torch.save(Z, output_file, _use_new_zipfile_serialization=True)
    print(f"Latent representations saved to {output_file} in compressed format")

def save_chromatin_state_bedgraph(labels, output_file, chromosome="chr21", start_position=0, resolution=400):
    write_bed(labels, chromosome, start_position, resolution, output_file)
    print(f"BedGraph file written to {output_file}")

def cluster(latent_representations, algorithm='HMM', pca_components=None, **kwargs):
    sequence_clustering = SequenceClustering()
    
    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        latent_representations = pca.fit_transform(latent_representations)
    
    if algorithm == 'GMM':
        labels, posteriors = sequence_clustering.GMM(latent_representations, **kwargs)
    elif algorithm == 'kmeans':
        labels = sequence_clustering.kmeans(latent_representations, **kwargs)
    elif algorithm == 'dbscan':
        labels = sequence_clustering.dbscan(latent_representations, **kwargs)
    elif algorithm == 'hierarchical':
        labels = sequence_clustering.hierarchical(latent_representations, **kwargs)
    elif algorithm == 'tsne':
        labels = sequence_clustering.tsne_clustering(latent_representations, **kwargs)
    elif algorithm == 'HMM':
        posteriors, labels = sequence_clustering.HMM(latent_representations, **kwargs)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    return labels

def generate_and_save_latent_representations(bios_name, dsf=1,
    model_path="models/CANDIeic_DNA_random_mask_oct17-expan2_model_checkpoint_epoch5.pth",
    hyper_parameters_path="models/hyper_parameters_eic_DNA_random_mask_oct17-expan2_CANDIeic_DNA_random_mask_oct17-expan2_20241017130209_params14059878.pkl",
    dataset_path="/project/compbio-lab/encode_data/",
    output_dir="output",
    number_of_states=10,
    DNA=True):

    CANDIP = CANDIPredictor(
        model_path, hyper_parameters_path, number_of_states, data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25)
    
    os.makedirs(output_dir, exist_ok=True)
    latent_file = f"{output_dir}/{bios_name}_latent.pt"


    if DNA:
        X, Y, P, seq, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
    else:
        X, Y, P, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
        seq = None
        
    Z = CANDIP.get_latent_representations(X, mX, mY, avX, seq=seq if DNA else None)
    save_latent_representations(Z, latent_file)

def visualize_latent(latent_file, annotation_bed_file=None, output_dir="output", visualize_method="all"):
    # Load the latent representations
    try:
        Z = torch.load(latent_file)
        print(f"Latent representations loaded from {latent_file}")
        print(f"Shape of loaded tensor: {Z.shape}")
    except FileNotFoundError:
        print(f"Error: File {latent_file} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Convert to numpy array if it's a torch tensor
    if isinstance(Z, torch.Tensor):
        Z = Z.cpu().numpy()

    # Derive bios_name from the latent file name
    bios_name = latent_file.split('/')[-1].split('_latent.pt')[0]

    # Initialize labels
    labels = None

    # Read the annotation BED file and build an interval tree if provided
    if annotation_bed_file:
        chromatin_states = {}
        try:
            with open(annotation_bed_file, 'r') as bed_file:
                for line in bed_file:
                    if line.startswith('track') or line.startswith('#'):
                        continue
                    fields = line.strip().split('\t')
                    if len(fields) < 4:
                        continue
                    chrom, start, end, state = fields[0], int(fields[1]), int(fields[2]), fields[3]
                    if chrom not in chromatin_states:
                        chromatin_states[chrom] = IntervalTree()
                    chromatin_states[chrom][start:end] = state

            print("Chromatin state annotations loaded.")

            # Define the genomic coordinates for each latent representation
            chromosome = 'chr21'
            start_position = 0
            resolution = 400

            L = Z.shape[0]  # Number of positions

            labels = []
            for i in range(L):
                start = start_position + i * resolution
                end = start + resolution
                intervals = chromatin_states.get(chromosome, IntervalTree())
                overlapping = intervals[start:end]
                if overlapping:
                    label = sorted(overlapping)[0].data
                else:
                    label = 'Unknown'
                labels.append(label)

            print("Labels assigned to latent representations.")
        except FileNotFoundError:
            print(f"Warning: Annotation file {annotation_bed_file} not found. Proceeding without labels.")
            labels = None
        except Exception as e:
            print(f"Warning: Error reading annotation file: {e}. Proceeding without labels.")
            labels = None

    def plot_and_save(embedding, labels, title, filename):
        plt.figure(figsize=(10, 8))
        if labels:
            unique_labels = sorted(set(labels))
            label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
            colors = plt.cm.get_cmap('tab20', len(unique_labels))
            ints = [label_to_int[label] for label in labels]
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=ints, cmap=colors, alpha=0.3, s=5)
            handles = [mpatches.Patch(color=colors(idx), label=label) for label, idx in label_to_int.items()]
            plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.3, s=5)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"{title} saved as {filename}")

    # Create output directory if it doesn't exist
    output_dir = 'models/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate visualizations based on the specified method
    if visualize_method == "all" or visualize_method == "pca":
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(Z)
        plot_and_save(pca_result, labels, 'PCA of Latent Representations', f'{output_dir}/{bios_name}_pca.png')

    if visualize_method == "all" or visualize_method == "umap":
        umap_reducer = umap.UMAP(random_state=42)
        umap_result = umap_reducer.fit_transform(Z)
        plot_and_save(umap_result, labels, 'UMAP of Latent Representations', f'{output_dir}/{bios_name}_umap.png')

    # if visualize_method == "all" or visualize_method == "tsne":
        # tsne = TSNE(n_components=2, random_state=42)
        # tsne_result = tsne.fit_transform(Z)
        # plot_and_save(tsne_result, labels, 't-SNE of Latent Representations', f'{output_dir}/{bios_name}_tsne.png')

def linear_probe_evaluation(latent_file, annotation_bed_file, k_folds=5):
    # Load the latent representations
    try:
        Z = torch.load(latent_file)
        print(f"Latent representations loaded from {latent_file}")
        print(f"Shape of loaded tensor: {Z.shape}")
    except FileNotFoundError:
        print(f"Error: File {latent_file} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Convert to numpy array if it's a torch tensor
    if isinstance(Z, torch.Tensor):
        Z = Z.cpu().numpy()

    # Read the annotation BED file and build an interval tree
    chromatin_states = {}
    with open(annotation_bed_file, 'r') as bed_file:
        for line in bed_file:
            if line.startswith('track') or line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 4:
                continue
            chrom, start, end, state = fields[0], int(fields[1]), int(fields[2]), fields[3]
            if chrom not in chromatin_states:
                chromatin_states[chrom] = IntervalTree()
            chromatin_states[chrom][start:end] = state

    print("Chromatin state annotations loaded.")

    # Define the genomic coordinates for each latent representation
    chromosome = 'chr21'
    start_position = 0
    resolution = 400

    # Assign labels to latent representations
    labels = []
    for i in range(Z.shape[0]):
        start = start_position + i * resolution
        end = start + resolution
        intervals = chromatin_states.get(chromosome, IntervalTree())
        overlapping = intervals[start:end]
        if overlapping:
            label = sorted(overlapping)[0].data
        else:
            label = 'Unknown'
        labels.append(label)

    print("Labels assigned to latent representations.")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Prepare for k-fold cross-validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Initialize metrics storage
    all_accuracies = []
    all_reports = []

    # Perform k-fold cross-validation
    for fold, (train_index, test_index) in enumerate(skf.split(Z, y), 1):
        X_train, X_test = Z[train_index], Z[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train logistic regression
        clf = LogisticRegression(multi_class='ovr', max_iter=100)
        clf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

        all_accuracies.append(accuracy)
        all_reports.append(report)

        print(f"Fold {fold} Accuracy: {accuracy:.4f}")

    # Calculate and print average metrics
    print("\nAverage Accuracy:", np.mean(all_accuracies))

    # Aggregate classification reports
    avg_report = {}
    for label in label_encoder.classes_:
        avg_report[label] = {
            'precision': np.mean([report[label]['precision'] for report in all_reports]),
            'recall': np.mean([report[label]['recall'] for report in all_reports]),
            'f1-score': np.mean([report[label]['f1-score'] for report in all_reports]),
            'support': np.mean([report[label]['support'] for report in all_reports])
        }

    print("\nAverage Classification Report:")
    for label, metrics in avg_report.items():
        print(f"{label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")
        print(f"  Support: {metrics['support']:.0f}")
        print()

def cluster_and_visualize_latent(latent_file, number_of_states=6, transition_exponent=1.0, pca_components=20):
    # Load the latent representations
    try:
        Z = torch.load(latent_file)
        print(f"Latent representations loaded from {latent_file}")
        print(f"Shape of loaded tensor: {Z.shape}")
    except FileNotFoundError:
        print(f"Error: File {latent_file} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Convert to numpy array if it's a torch tensor
    if isinstance(Z, torch.Tensor):
        Z = Z.cpu().numpy()

    # Derive bios_name from the latent file name
    bios_name = latent_file.split('/')[-1].split('_latent.pt')[0]

    # Create output directory if it doesn't exist
    output_dir = 'models/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def plot_and_save(embedding, labels, title, filename):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', alpha=0.3, s=5)
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"{title} saved as {filename}")

    # Perform clustering
    clustering_methods = ['HMM', 'GMM', 'kmeans']
    for method in clustering_methods:
        print(f"\nPerforming {method} clustering...")
        if method == 'HMM':
            labels = cluster(Z, algorithm=method, n_components=number_of_states, pca_components=pca_components, transition_exponent=transition_exponent)
        elif method == 'GMM':
            labels = cluster(Z, algorithm=method, n_components=number_of_states, pca_components=pca_components)
        else:
            labels = cluster(Z, algorithm=method, n_clusters=number_of_states)

        # Save clustering results as BED file
        write_bed(labels, "chr21", 0, 400, f'{output_dir}/{bios_name}_{method.lower()}.bed', 
                  is_posterior=False, 
                  track_name=f"{method} Clustering", 
                  track_description=f"{method} Clustering Results", 
                  visibility="dense")
        print(f"{method} clustering results saved as {output_dir}/{bios_name}_{method.lower()}.bed")

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(Z)
        plot_and_save(pca_result, labels, f'PCA of Latent Representations ({method} clusters)', 
                      f'{output_dir}/{bios_name}_pca_{method.lower()}.png')

        # UMAP
        umap_reducer = umap.UMAP(random_state=42)
        umap_result = umap_reducer.fit_transform(Z)
        plot_and_save(umap_result, labels, f'UMAP of Latent Representations ({method} clusters)', 
                      f'{output_dir}/{bios_name}_umap_{method.lower()}.png')

        # t-SNE
        # tsne = TSNE(n_components=2, random_state=42)
        # tsne_result = tsne.fit_transform(Z)
        # plot_and_save(tsne_result, labels, f't-SNE of Latent Representations ({method} clusters)', 
        #               f'{output_dir}/{bios_name}_tsne_{method.lower()}.png')

    print("\nClustering and visualization complete.")

def annotate_latent(latent_file, output_dir="output", number_of_states=6, transition_exponent=1.0, pca_components=20):
    # Load the latent representations
    try:
        Z = torch.load(latent_file)
        print(f"Latent representations loaded from {latent_file}")
        print(f"Shape of loaded tensor: {Z.shape}")
    except FileNotFoundError:
        print(f"Error: File {latent_file} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Convert to numpy array if it's a torch tensor
    if isinstance(Z, torch.Tensor):
        Z = Z.cpu().numpy()

    # Derive bios_name from the latent file name
    bios_name = latent_file.split('/')[-1].split('_latent.pt')[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    labels = cluster(Z, algorithm="HMM", n_components=number_of_states, pca_components=pca_components, transition_exponent=transition_exponent)

    # Save clustering results as BED file
    write_bed(labels, "chr21", 0, 400, f'{output_dir}/{bios_name}_latent_{pca_components}PC_hmm.bed', 
                is_posterior=False, 
                track_name=f"HMM Clustering", 
                track_description=f"HMM Clustering Results", 
                visibility="dense")
    print(f"HMM clustering results saved as {output_dir}/{bios_name}_latent_{pca_components}PC_hmm.bed")

def annotate_decoded_data(bios_name, decoded_resolution=25, annotation_resolution=200, annotate_based_on="pval", dsf=1, transition_exponent=1.0,
    model_path="models/CANDIeic_DNA_random_mask_oct17-expan2_model_checkpoint_epoch5.pth",
    hyper_parameters_path="models/hyper_parameters_eic_DNA_random_mask_oct17-expan2_CANDIeic_DNA_random_mask_oct17-expan2_20241017130209_params14059878.pkl",
    dataset_path="/project/compbio-lab/encode_data/",
    output_dir="output",
    number_of_states=10,
    DNA=True):

    CANDIP = CANDIPredictor(
        model_path, hyper_parameters_path, number_of_states, data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25)
    
    os.makedirs(output_dir, exist_ok=True)

    if DNA:
        X, Y, P, seq, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
    else:
        X, Y, P, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
        seq = None
        
    count, pval = CANDIP.get_decoded_signal(X, mX, mY, avX, seq=seq if DNA else None)

    # Convert decoded resolution to annotation resolution
    decoded_bins = count.shape[0]
    annotation_bins = decoded_bins * decoded_resolution // annotation_resolution
    
    print(f"Decoded bins: {decoded_bins}, Annotation bins: {annotation_bins}")
    print(f"Original count shape: {count.shape}")
    
    # Reshape and average for count
    count_reshaped = count.reshape(annotation_bins, -1).mean(axis=1)
    
    # Reshape and average for pval
    pval_reshaped = pval.reshape(annotation_bins, -1).mean(axis=1)

    print(f"Reshaped count shape: {count_reshaped.shape}")
    print(f"Reshaped pval shape: {pval_reshaped.shape}")

    # Update count and pval with the new resolution
    count = count_reshaped
    pval = pval_reshaped

    if annotate_based_on == "count":
        labels = cluster(count, algorithm="HMM", n_components=number_of_states, transition_exponent=transition_exponent)
    else:
        labels = cluster(pval, algorithm="HMM", n_components=number_of_states, transition_exponent=transition_exponent)

    # Save clustering results as BED file
    write_bed(labels, "chr21", 0, annotation_resolution, f'{output_dir}/{bios_name}_decoded_hmm_{annotate_based_on}.bed', 
        is_posterior=False, 
        track_name=f"HMM Clustering",     
        track_description=f"HMM Clustering Results", 
        visibility="dense")
    print(f"HMM clustering results saved as {output_dir}/{bios_name}_decoded_hmm_{annotate_based_on}.bed")

def latent_position_dependency_experiment(
    bios_name, 
    model_path="models/CANDIeic_DNA_random_mask_oct17-expan2_model_checkpoint_epoch5.pth",
    hyper_parameters_path="models/hyper_parameters_eic_DNA_random_mask_oct17-expan2_CANDIeic_DNA_random_mask_oct17-expan2_20241017130209_params14059878.pkl",
    dataset_path="/project/compbio-lab/encode_data/",
    output_dir="output", number_of_states=10, DNA=True, n_positions=10):
    predictor = CANDIPredictor(
        model_path, hyper_parameters_path, number_of_states, data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25)
    predictor.latent_position_dependency_experiment(bios_name, n_positions=n_positions)

def compare_cropped_noncropped(bios_name, dsf=1,
    model_path="models/CANDIeic_DNA_random_mask_oct17-expan2_model_checkpoint_epoch5.pth",
    hyper_parameters_path="models/hyper_parameters_eic_DNA_random_mask_oct17-expan2_CANDIeic_DNA_random_mask_oct17-expan2_20241017130209_params14059878.pkl",
    dataset_path="/project/compbio-lab/encode_data/",
    output_dir="output",
    number_of_states=10,
    DNA=True,
    n_bins=20):
    """
    Compare predictions from cropped and non-cropped approaches using binned positional analysis.
    """
    CANDIP = CANDIPredictor(
        model_path, hyper_parameters_path, number_of_states, 
        data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25)
    
    os.makedirs(output_dir, exist_ok=True)

    # Load data and get predictions
    if DNA:
        X, Y, P, seq, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
        p, n, mu, var, Z = CANDIP.pred(X, mX, mY, avX, seq=seq)
        p_crop, n_crop, mu_crop, var_crop, Z_crop = CANDIP.pred_crop(X, mX, mY, avX, seq=seq)
    else:
        X, Y, P, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
        p, n, mu, var, Z = CANDIP.pred(X, mX, mY, avX)
        p_crop, n_crop, mu_crop, var_crop, Z_crop = CANDIP.pred_crop(X, mX, mY, avX)

    # Flatten first two dimensions of cropped tensors
    p_crop = p_crop.reshape(-1, p_crop.shape[-1])  # [1167*1600, 35]
    n_crop = n_crop.reshape(-1, n_crop.shape[-1])  # [1167*1600, 35] 
    mu_crop = mu_crop.reshape(-1, mu_crop.shape[-1])  # [1167*1600, 35]
    var_crop = var_crop.reshape(-1, var_crop.shape[-1])  # [1167*1600, 35]
    Z_crop = Z_crop.reshape(-1, Z_crop.shape[-1])  # [1167*100, 560]

    # Print shapes of original and cropped predictions
    print("\nShape Analysis of Predictions:")
    print(f"Original predictions:")
    print(f"p shape: {p.shape}")
    print(f"n shape: {n.shape}") 
    print(f"mu shape: {mu.shape}")
    print(f"var shape: {var.shape}")
    print(f"Z shape: {Z.shape}")
    
    print(f"\nCropped predictions:")
    print(f"p_crop shape: {p_crop.shape}")
    print(f"n_crop shape: {n_crop.shape}")
    print(f"mu_crop shape: {mu_crop.shape}")
    print(f"var_crop shape: {var_crop.shape}")
    print(f"Z_crop shape: {Z_crop.shape}")

    # Convert to numpy arrays
    def to_numpy(tensor):
        return tensor.cpu().numpy()

    # Reshape signal outputs to [batch, window, feature]
    window_size = CANDIP.context_length
    feature_dim = p.shape[-1]
    
    def reshape_signal(arr):
        arr = to_numpy(arr)
        total_windows = arr.shape[0] // window_size
        return arr[:total_windows * window_size].reshape(total_windows, window_size, feature_dim)

    # Convert p, n to NegativeBinomial mean
    def get_nbinom_mean(p, n):
        return (n * (1 - p)) / p

    # Process signal outputs (p, n, mu, var)
    p, p_crop = reshape_signal(p), reshape_signal(p_crop)
    n, n_crop = reshape_signal(n), reshape_signal(n_crop)
    mu, mu_crop = reshape_signal(mu), reshape_signal(mu_crop)
    
    # Calculate means
    nbinom_mean = get_nbinom_mean(p, n)
    nbinom_mean_crop = get_nbinom_mean(p_crop, n_crop)
    
    # Process latent representations
    Z, Z_crop = to_numpy(Z), to_numpy(Z_crop)
    latent_window_size = CANDIP.model.l2
    latent_dim = Z.shape[-1]
    Z = Z.reshape(-1, latent_window_size, latent_dim)
    Z_crop = Z_crop.reshape(-1, latent_window_size, latent_dim)

    # Print shapes of original and cropped predictions
    print("\nShape Analysis of Predictions:")
    print(f"Original predictions:")
    print(f"p shape: {p.shape}")
    print(f"n shape: {n.shape}") 
    print(f"mu shape: {mu.shape}")
    print(f"var shape: {var.shape}")
    print(f"Z shape: {Z.shape}")
    
    print(f"\nCropped predictions:")
    print(f"p_crop shape: {p_crop.shape}")
    print(f"n_crop shape: {n_crop.shape}")
    print(f"mu_crop shape: {mu_crop.shape}")
    print(f"var_crop shape: {var_crop.shape}")
    print(f"Z_crop shape: {Z_crop.shape}")

    exit()

    # Function to compute binned differences for a single feature
    def compute_binned_differences_single_feature(arr1, arr2, feature_idx, n_bins=n_bins):
        """Compute differences between two arrays for a single feature across position bins."""
        _, seq_len, _ = arr1.shape
        bin_size = seq_len // n_bins
        
        mean_diffs = np.zeros(n_bins)
        std_diffs = np.zeros(n_bins)
        
        for bin_idx in range(n_bins):
            start_idx = bin_idx * bin_size
            end_idx = start_idx + bin_size if bin_idx < n_bins - 1 else seq_len
            
            bin_diff = arr1[:, start_idx:end_idx, feature_idx] - arr2[:, start_idx:end_idx, feature_idx]
            mean_diffs[bin_idx] = np.mean(bin_diff)
            std_diffs[bin_idx] = np.std(bin_diff)
            
        return mean_diffs, std_diffs

    # Function to compute latent differences
    def compute_latent_differences(Z1, Z2, n_bins=n_bins):
        """Compute euclidean and cosine distances for latent representations across bins."""
        _, seq_len, n_features = Z1.shape
        bin_size = seq_len // n_bins
        
        euclidean_dists = np.zeros(n_bins)
        cosine_dists = np.zeros(n_bins)
        
        for bin_idx in range(n_bins):
            start_idx = bin_idx * bin_size
            end_idx = start_idx + bin_size if bin_idx < n_bins - 1 else seq_len
            
            Z1_bin = Z1[:, start_idx:end_idx, :].reshape(-1, n_features)
            Z2_bin = Z2[:, start_idx:end_idx, :].reshape(-1, n_features)
            
            euclidean_dists[bin_idx] = np.mean([euclidean(z1, z2) for z1, z2 in zip(Z1_bin, Z2_bin)])
            cosine_dists[bin_idx] = np.mean([cosine(z1, z2) for z1, z2 in zip(Z1_bin, Z2_bin)])
            
        return euclidean_dists, cosine_dists

    # Compute latent differences
    latent_euclidean, latent_cosine = compute_latent_differences(Z, Z_crop)

    # Create subplots for each feature
    n_features = feature_dim
    fig, axes = plt.subplots(n_features + 1, 2, figsize=(20, 5*(n_features + 1)))
    
    # Plot differences for each feature
    for feature_idx in range(n_features):
        # NegativeBinomial mean differences
        nbinom_mean_diffs, nbinom_std_diffs = compute_binned_differences_single_feature(
            nbinom_mean, nbinom_mean_crop, feature_idx)
        
        # Gaussian mean differences
        mu_mean_diffs, mu_std_diffs = compute_binned_differences_single_feature(
            mu, mu_crop, feature_idx)
        
        # Plot NegativeBinomial differences
        axes[feature_idx, 0].errorbar(range(n_bins), nbinom_mean_diffs, 
                                    yerr=nbinom_std_diffs, 
                                    label=f'Feature {feature_idx+1}')
        axes[feature_idx, 0].set_title(f'NegativeBinomial Mean Differences - Feature {feature_idx+1}')
        axes[feature_idx, 0].set_xlabel('Position Bin')
        axes[feature_idx, 0].set_ylabel('Mean Difference')
        
        # Plot Gaussian differences
        axes[feature_idx, 1].errorbar(range(n_bins), mu_mean_diffs, 
                                    yerr=mu_std_diffs, 
                                    label=f'Feature {feature_idx+1}')
        axes[feature_idx, 1].set_title(f'Gaussian Mean Differences - Feature {feature_idx+1}')
        axes[feature_idx, 1].set_xlabel('Position Bin')
        axes[feature_idx, 1].set_ylabel('Mean Difference')
    
    # Plot Latent distances in the last row
    axes[-1, 0].plot(range(n_bins), latent_euclidean, label='Euclidean Distance')
    axes[-1, 0].set_title('Latent Space Euclidean Distance')
    axes[-1, 0].set_xlabel('Position Bin')
    axes[-1, 0].set_ylabel('Distance')
    axes[-1, 0].legend()
    
    axes[-1, 1].plot(range(n_bins), latent_cosine, label='Cosine Distance')
    axes[-1, 1].set_title('Latent Space Cosine Distance')
    axes[-1, 1].set_xlabel('Position Bin')
    axes[-1, 1].set_ylabel('Distance')
    axes[-1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{bios_name}_binned_comparison_by_feature.png')
    plt.close()

    # Print summary statistics for each feature
    print("\nSummary Statistics by Feature:")
    for feature_idx in range(n_features):
        nbinom_mean_diffs, _ = compute_binned_differences_single_feature(
            nbinom_mean, nbinom_mean_crop, feature_idx)
        mu_mean_diffs, _ = compute_binned_differences_single_feature(
            mu, mu_crop, feature_idx)
        
        print(f"\nFeature {feature_idx+1}:")
        print(f"NegativeBinomial Mean Differences:")
        print(f"  Mean: {np.mean(nbinom_mean_diffs):.4f}")
        print(f"  Std: {np.std(nbinom_mean_diffs):.4f}")
        print(f"Gaussian Mean Differences:")
        print(f"  Mean: {np.mean(mu_mean_diffs):.4f}")
        print(f"  Std: {np.std(mu_mean_diffs):.4f}")
    
    print("\nLatent Space Distances:")
    print(f"Mean Euclidean: {np.mean(latent_euclidean):.4f}")
    print(f"Mean Cosine: {np.mean(latent_cosine):.4f}")

def compare_decoded_outputs(bios_name, dsf=1,
    model_path="models/CANDIeic_DNA_random_mask_oct17-expan2_model_checkpoint_epoch5.pth",
    hyper_parameters_path="models/hyper_parameters_eic_DNA_random_mask_oct17-expan2_CANDIeic_DNA_random_mask_oct17-expan2_20241017130209_params14059878.pkl",
    dataset_path="/project/compbio-lab/encode_data/",
    output_dir="output",
    number_of_states=10,
    DNA=True):

    CANDIP = CANDIPredictor(
        model_path, hyper_parameters_path, number_of_states, data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25)
    
    os.makedirs(output_dir, exist_ok=True)

    # Function to get decoded signals and availability information
    def get_decoded_signals(fill_in_y_prompt):
        if DNA:
            X, Y, P, seq, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf, fill_in_y_prompt=fill_in_y_prompt)
        else:
            X, Y, P, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf, fill_in_y_prompt=fill_in_y_prompt)
            seq = None
        
        count, pval = CANDIP.get_decoded_signal(X, mX, mY, avX, seq=seq if DNA else None)
        return count, pval, avX, avY
    
    # Get decoded signals for both cases
    count_true, pval_true, avX_true, avY_true = get_decoded_signals(fill_in_y_prompt=True)
    count_false, pval_false, avX_false, avY_false = get_decoded_signals(fill_in_y_prompt=False)

    # Convert PyTorch tensors to NumPy arrays
    count_true = count_true.cpu().numpy()
    count_false = count_false.cpu().numpy()
    pval_true = pval_true.cpu().numpy()
    pval_false = pval_false.cpu().numpy()
    avX_true = avX_true.cpu().numpy()
    avY_true = avY_true.cpu().numpy()

    # Print available features
    print("Available features in avX:")
    for i, available in enumerate(avX_true[0]):
        print(f"Feature {i+1}: {'Available' if available else 'Not Available'}")
    
    print("\nAvailable features in avY:")
    for i, available in enumerate(avY_true[0]):
        print(f"Feature {i+1}: {'Available' if available else 'Not Available'}")

    # Compare the results
    count_diff = count_true - count_false
    pval_diff = pval_true - pval_false

    # Calculate statistics
    count_mean_diff = np.mean(count_diff)
    count_std_diff = np.std(count_diff)
    pval_mean_diff = np.mean(pval_diff)
    pval_std_diff = np.std(pval_diff)

    # Create histograms for each feature
    num_features = count_diff.shape[1]
    fig, axes = plt.subplots(num_features, 2, figsize=(20, 5*num_features))
    
    for i in range(num_features):
        # Count difference histogram
        axes[i, 0].hist(count_diff[:, i], bins=100, edgecolor='black')
        axes[i, 0].set_title(f'Count Difference - Feature {i+1}\n' +
                             f'({"Available" if avX_true[0, i] else "Not Available"} in avX, ' +
                             f'{"Available" if avY_true[0, i] else "Not Available"} in avY)')
        axes[i, 0].set_xlabel('Difference')
        axes[i, 0].set_ylabel('Frequency')
        
        # P-value difference histogram
        axes[i, 1].hist(pval_diff[:, i], bins=100, edgecolor='black')
        axes[i, 1].set_title(f'P-value Difference - Feature {i+1}\n' +
                             f'({"Available" if avX_true[0, i] else "Not Available"} in avX, ' +
                             f'{"Available" if avY_true[0, i] else "Not Available"} in avY)')
        axes[i, 1].set_xlabel('Difference')
        axes[i, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{bios_name}_decoded_comparison_histograms.png')
    plt.close()

    # Print statistics
    print(f"Count prediction differences:")
    print(f"  Mean difference: {count_mean_diff}")
    print(f"  Standard deviation of difference: {count_std_diff}")
    print(f"P-value prediction differences:")
    print(f"  Mean difference: {pval_mean_diff}")
    print(f"  Standard deviation of difference: {pval_std_diff}")

    print(f"Comparison results saved to {output_dir}/{bios_name}_decoded_comparison_histograms.png")

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <bios_name> <function_name> [additional_args]")
        print("Available functions: generate_latent, visualize_latent, annotate_latent, annotate_decoded, position_dependency, compare_decoded, compare_cropped")
        sys.exit(1)

    bios_name = sys.argv[1]
    function_name = sys.argv[2]

    # Default parameters
    output_dir = "models/output"
    number_of_states = 10
    DNA = True
    dsf = 1
    pca_components = 20
    transition_exponent = 10.0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if function_name == "generate_latent":
        print(f"Generating and saving latent representations for {bios_name}...")
        generate_and_save_latent_representations(
            bios_name, dsf=dsf, output_dir=output_dir, 
            number_of_states=number_of_states, DNA=DNA
        )

    elif function_name == "visualize_latent":
        print(f"Visualizing latent representations for {bios_name}...")
        latent_file = f"{output_dir}/{bios_name}_latent.pt"
        if not os.path.exists(latent_file):
            print(f"Error: File {latent_file} not found.")
            print(f"Generating and saving latent representations for {bios_name}...")
            generate_and_save_latent_representations(
                bios_name, dsf=dsf, output_dir=output_dir, 
                number_of_states=number_of_states, DNA=DNA
            )
        visualize_latent(latent_file, output_dir=output_dir)

    elif function_name == "annotate_latent":
        print(f"Annotating latent representations for {bios_name}...")
        latent_file = f"{output_dir}/{bios_name}_latent.pt"
        annotate_latent(
            latent_file, pca_components=pca_components,
            transition_exponent=transition_exponent, output_dir=output_dir,
            number_of_states=number_of_states
        )

    elif function_name == "annotate_decoded":
        print(f"Annotating decoded data for {bios_name}...")
        annotate_decoded_data(
            bios_name, annotate_based_on="pval", dsf=dsf, 
            transition_exponent=transition_exponent, 
            output_dir=output_dir, number_of_states=number_of_states, DNA=DNA
        )
    
    elif function_name == "position_dependency":
        print(f"Performing latent position dependency experiment for {bios_name}...")
        latent_position_dependency_experiment(
            bios_name, n_positions=100)

    elif function_name == "compare_decoded":
        print(f"Comparing decoded outputs for {bios_name}...")
        compare_decoded_outputs(
            bios_name, dsf=dsf, output_dir=output_dir, 
            number_of_states=number_of_states, DNA=DNA
        )

    elif function_name == "compare_cropped":
        print(f"Comparing cropped and non-cropped predictions for {bios_name}...")
        compare_cropped_noncropped(
            bios_name, dsf=dsf, output_dir=output_dir,
            number_of_states=number_of_states, DNA=DNA
        )

    else:
        print(f"Unknown function: {function_name}")
        sys.exit(1)
    
    

    print("Operation completed successfully.")

if __name__ == "__main__":
    main()




