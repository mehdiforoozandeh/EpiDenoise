from re import T
from CANDI import *
import sys
import os
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, AffinityPropagation
from sklearn.manifold import TSNE
import numpy as np
# import umap
# from minisom import MiniSom
# import hdbscan
# from sklearn.cluster import KernelKMeans
from hmmlearn import hmm
import numpy as np

# SAGA means segmentation and genome annotation -- similar to ChromHMM or Segway

def write_bedgraph(data, chromosome, start_position, resolution, output_file, is_posterior=False):
    """
    Write clustering results to a BedGraph file.

    Parameters:
    data (numpy.ndarray): L*k matrix of posterior probabilities or cluster assignments
    chromosome (str): Chromosome name (e.g., 'chr1')
    start_position (int): Start position of the first segment
    resolution (int): Resolution of each segment (e.g., 25 bp)
    output_file (str): Path to the output BedGraph file
    is_posterior (bool): If True, data contains posterior probabilities. If False, contains cluster assignments.

    """
    L = data.shape[0]
    
    with open(output_file, 'w') as f:
        for i in range(L):
            start = start_position + i * resolution
            end = start + resolution
            
            if is_posterior:
                # For posterior probabilities, write the most likely state
                state = np.argmax(data[i]) + 1  # +1 to make states 1-indexed
            else:
                # For cluster assignments, use the assigned state directly
                state = data[i]
            
            f.write(f"{chromosome}\t{start}\t{end}\t{state}\n")

    print(f"BedGraph file written to {output_file}")

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
    
    def HMM(self, sequence_embedding, n_components=18, random_state=42):
        # Ensure the input is a 2D numpy array
        sequence_embedding = np.array(sequence_embedding)
        if len(sequence_embedding.shape) == 1:
            sequence_embedding = sequence_embedding.reshape(-1, 1)

        # Initialize the HMM model
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", random_state=random_state, verbose=True)

        # Fit the model to the data
        model.fit(sequence_embedding)

        # Predict the hidden states
        hidden_states = model.predict(sequence_embedding)
        posteriors = model.predict_proba(sequence_embedding)

        self.models['HMM'] = model
        return posteriors, hidden_states

    def get_model(self, model_name):
        return self.models.get(model_name, None)

class SAGA(object):
    def __init__(
        self, model, hyper_parameters_path, number_of_states, 
        split="test", DNA=False, eic=True, chr="chr21", resolution=25, context_length=1600,
        savedir="models/evals/", data_path="/project/compbio-lab/encode_data/"):

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

    def load_bios(self, bios_name, x_dsf, y_dsf=1, fill_in_y_prompt=False):
        print(f"getting bios vals for {bios_name}")

        if self.eic:
            if self.split == "test":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("B_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            elif self.split == "val":
                temp_x, temp_mx = self.dataset.load_bios(bios_name.replace("V_", "T_"), [self.chr, 0, self.chr_sizes[self.chr]], x_dsf)
            
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

    def get_latent_representations(self, X, mX, mY, avX, seq=None):
        if self.DNA:
            _, _, _, _, Z = self.pred(X, mX, mY, avX, seq=seq, imp_target=[])
        else:
            _, _, _, _, Z = self.pred(X, mX, mY, avX, seq=None, imp_target=[])
        return Z

    def save_latent_representations(self, Z, output_file):
        torch.save(Z, output_file, _use_new_zipfile_serialization=True)
        print(f"Latent representations saved to {output_file} in compressed format")

    def cluster(self, latent_representations, algorithm='GMM', pca_components=None, **kwargs):
        sequence_clustering = SequenceClustering()
        
        if pca_components is not None:
            from sklearn.decomposition import PCA
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

    def save_chromatin_state_bedgraph(self, labels, chromosome, start_position, output_file):
        write_bedgraph(labels, chromosome, start_position, self.resolution*(self.model.l1//self.model.l2), output_file)
        print(f"BedGraph file written to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python SAGA.py <bios_name>")
        sys.exit(1)

    bios_name = sys.argv[1]

    # Initialize SAGA
    model_path = "models/CANDIeic_DNA_random_mask_oct17-expan2_model_checkpoint_epoch0.pth"  # Update this with your model path
    hyper_parameters_path = "models/hyper_parameters_eic_DNA_random_mask_oct17-expan2_CANDIeic_DNA_random_mask_oct17-expan2_20241017130209_params14059878.pkl"
    dataset_path = "/project/compbio-lab/encode_data/"
    number_of_states = 16
    DNA = True
    saga = SAGA(model_path, hyper_parameters_path, number_of_states, data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25)

    # Load biosample data
    if DNA:
        X, Y, P, seq, mX, mY, avX, avY= saga.load_bios(bios_name, x_dsf=1)
    else:
        X, Y, P, mX, mY, avX, avY = saga.load_bios(bios_name, x_dsf=1)

    # Get latent representations
    Z = saga.get_latent_representations(X, mX, mY, avX, seq=seq)

    # L = Z.shape[0]
    # start = L // 2 - L // 6  # 1/6 of L is 15% of the total length
    # end = L // 2 + L // 6    # This gives us the middle 30%
    # Z = Z[start:end, :]
    
    # Save latent representations
    os.makedirs("output", exist_ok=True)
    latent_file = f"output/{bios_name}_latent.pt"
    saga.save_latent_representations(Z, latent_file)
    
    # Perform clustering
    labels = saga.cluster(Z, algorithm='HMM', n_components=number_of_states, pca_components=10)
    # labels = saga.cluster(Z, algorithm='GMM', n_components=number_of_states, pca_components=10)
    # labels = saga.cluster(Z, algorithm='kmeans', n_clusters=number_of_states, pca_components=10)

    unique_labels, counts = np.unique(labels, return_counts=True)
    total_length = len(labels)
    print(f"Number of unique labels: {len(unique_labels)}")
    for label, count in zip(unique_labels, counts):
        fraction = count / total_length
        print(f"Label {label}: {count} occurrences, covering {fraction:.2%} of the sequence")
    
    # Save chromatin state bedgraph
    bedgraph_file = f"output/{bios_name}_chromatin_states.bedgraph"
    saga.save_chromatin_state_bedgraph(labels, saga.chr, 0, bedgraph_file)

if __name__ == "__main__":
    # T_adrenal_gland
    main()
