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

    def pred(self, X, mX, mY, avail, imp_target=[], seq=None, crop_percent=0.1):
        """
        Predicts over the input data X, mX, mY, avail, possibly seq.
        Returns flattened results matching the shape of the old pred function.
        """
        # Calculate crop length and valid length
        crop_len = int(crop_percent * self.context_length)
        valid_len = self.context_length - 2 * crop_len
        stride = valid_len

        total_length = X.shape[0] * X.shape[1]  # Total number of positions
        feature_dim = X.shape[-1]
        
        # Flatten input tensors
        X_flat = X.view(-1, feature_dim)
        mX_flat = mX.view(-1, feature_dim)
        mY_flat = mY.view(-1, feature_dim)
        avail_flat = avail.view(-1, avail.shape[-1])

        if self.DNA:
            seq_flat = seq.view(-1, seq.shape[-1])

        # Generate start indices for windows
        starts = list(range(0, total_length - self.context_length + 1, stride))
        if starts[-1] + self.context_length < total_length:
            starts.append(total_length - self.context_length)

        # Prepare windows and tracking arrays
        X_windows = []
        mX_windows = []
        mY_windows = []
        avail_windows = []
        if self.DNA:
            seq_windows = []
        valid_positions_in_window = []
        valid_positions_in_seq = []

        for start in starts:
            end = start + self.context_length
            X_window = X_flat[start:end]
            mX_window = mX_flat[start:end]
            mY_window = mY_flat[start:end]
            avail_window = avail_flat[start:end]
            if self.DNA:
                seq_window = seq_flat[start*self.resolution:end*self.resolution]
                seq_windows.append(seq_window)

            X_windows.append(X_window)
            mX_windows.append(mX_window)
            mY_windows.append(mY_window)
            avail_windows.append(avail_window)

            # Determine valid positions
            if start == 0:
                # First window
                valid_start = 0
                valid_end = self.context_length - crop_len
            elif end >= total_length:
                # Last window
                valid_start = crop_len
                valid_end = self.context_length
            else:
                # Middle windows
                valid_start = crop_len
                valid_end = self.context_length - crop_len

            # Store valid positions
            valid_pos_in_window = torch.arange(valid_start, valid_end)
            valid_pos_in_seq = torch.arange(start + valid_start, start + valid_end)
            valid_positions_in_window.append(valid_pos_in_window)
            valid_positions_in_seq.append(valid_pos_in_seq)

        # Stack windows
        X_stacked = torch.stack(X_windows)
        mX_stacked = torch.stack(mX_windows)
        mY_stacked = torch.stack(mY_windows)
        avail_stacked = torch.stack(avail_windows)
        if self.DNA:
            seq_stacked = torch.stack(seq_windows)

        # Initialize output tensors - now keeping them flattened
        n_full = torch.full((total_length, feature_dim), float('nan'), device="cpu", dtype=torch.float32)
        p_full = torch.full((total_length, feature_dim), float('nan'), device="cpu", dtype=torch.float32)
        mu_full = torch.full((total_length, feature_dim), float('nan'), device="cpu", dtype=torch.float32)
        var_full = torch.full((total_length, feature_dim), float('nan'), device="cpu", dtype=torch.float32)
        Z_full = torch.full((total_length // (self.context_length // self.model.l2), self.model.latent_dim), 
                            float('nan'), device="cpu", dtype=torch.float32)

        num_windows = X_stacked.shape[0]

        for idx in range(0, num_windows, self.batch_size):
            torch.cuda.empty_cache()

            x_batch = X_stacked[idx:idx + self.batch_size]
            mX_batch = mX_stacked[idx:idx + self.batch_size]
            mY_batch = mY_stacked[idx:idx + self.batch_size]
            avail_batch = avail_stacked[idx:idx + self.batch_size]

            valid_positions_in_window_batch = valid_positions_in_window[idx:idx + self.batch_size]
            valid_positions_in_seq_batch = valid_positions_in_seq[idx:idx + self.batch_size]

            if self.DNA:
                seq_batch = seq_stacked[idx:idx + self.batch_size]

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

                if len(imp_target) > 0:
                    x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    avail_batch[:, imp_target] = 0

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

            batch_size_actual = x_batch.shape[0]
            for b in range(batch_size_actual):
                valid_pos_in_window_b = valid_positions_in_window_batch[b]
                valid_pos_in_seq_b = valid_positions_in_seq_batch[b]

                try:
                    n_valid = outputs_n[b, valid_pos_in_window_b].cpu()
                    p_valid = outputs_p[b, valid_pos_in_window_b].cpu()
                    mu_valid = outputs_mu[b, valid_pos_in_window_b].cpu()
                    var_valid = outputs_var[b, valid_pos_in_window_b].cpu()
                    Z_valid = latent[b].cpu()  # Take the entire latent sequence

                    # Assign to flattened tensors
                    n_full[valid_pos_in_seq_b] = n_valid
                    p_full[valid_pos_in_seq_b] = p_valid
                    mu_full[valid_pos_in_seq_b] = mu_valid
                    var_full[valid_pos_in_seq_b] = var_valid
                    
                    # Calculate correct indices for latent representation
                    z_start = (valid_pos_in_seq_b[0] // (self.context_length // self.model.l2))
                    z_end = (valid_pos_in_seq_b[-1] // (self.context_length // self.model.l2)) + 1
                    Z_full[z_start:z_end] = Z_valid

                except IndexError as e:
                    print(f"  IndexError in window {b}: {str(e)}")
                    continue

            # Clean up GPU memory
            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n, outputs_mu, outputs_var, latent
            torch.cuda.empty_cache()

        # Return flattened tensors (no reshaping)
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
        predictor = CANDIPredictor(
            model_path, hyper_parameters_path, number_of_states, data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25)
        predictor.latent_position_dependency_experiment(bios_name, n_positions=n_positions)

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
        print("Available functions: generate_latent, visualize_latent, annotate_latent, annotate_decoded, position_dependency, compare_decoded")
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

    else:
        print(f"Unknown function: {function_name}")
        sys.exit(1)

    print("Operation completed successfully.")

if __name__ == "__main__":
    main()




