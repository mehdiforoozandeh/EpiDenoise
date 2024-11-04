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
from inference import *

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
    output_dir="output",
    number_of_states=10,
    DNA=True,
    n_positions=10):
    
    # Create CANDIPredictor instance
    predictor = CANDIPredictor(
        model_path, hyper_parameters_path, number_of_states, 
        data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print("Loading data...")
    if DNA:
        X, Y, P, seq, mX, mY, avX, avY = predictor.load_bios(bios_name, x_dsf=1)
    else:
        X, Y, P, mX, mY, avX, avY = predictor.load_bios(bios_name, x_dsf=1)
    
    # Flatten only X and seq
    X_flat = X.reshape(-1, X.shape[-1])
    if DNA:
        seq_flat = seq.reshape(-1, seq.shape[-1])
    total_length = X_flat.shape[0]

    # Print shapes
    print("Shapes of tensors:")
    print(f"X shape: {X.shape}")
    print(f"X_flat shape: {X_flat.shape}")
    print(f"mX shape: {mX.shape}")
    print(f"mY shape: {mY.shape}")
    print(f"avX shape: {avX.shape}")
    if DNA:
        print(f"seq shape: {seq.shape}")
        print(f"seq_flat shape: {seq_flat.shape}")

    # Randomly select positions avoiding edges
    positions = np.random.randint(predictor.context_length//2, total_length - predictor.context_length//2, size=n_positions)
    offsets = np.arange(-predictor.context_length//2, predictor.context_length//2 + 1, (predictor.model.l1 // predictor.model.l2)*5)

    # Initialize arrays to store distances
    cosine_distances = np.zeros((n_positions, len(offsets)))
    euclidean_distances = np.zeros((n_positions, len(offsets)))
    manhattan_distances = np.zeros((n_positions, len(offsets)))

    for idx, pos in enumerate(tqdm(positions, desc='Processing positions')):
        # Reference context window centered at pos
        start = pos - predictor.context_length//2
        end = pos + predictor.context_length//2
        X_ref = X_flat[start:end]
        mX_ref = mX[0].unsqueeze(0)
        mY_ref = mY[0].unsqueeze(0)
        avX_ref = avX[0].unsqueeze(0)
        if DNA:
            seq_ref = seq_flat[(start*predictor.resolution):(end*predictor.resolution)]
        
        # Expand dims for X_ref
        X_ref = X_ref.unsqueeze(0)
        if DNA:
            seq_ref = seq_ref.unsqueeze(0)
        
        # Get reference latent representation
        if DNA:
            Z_ref = predictor.get_latent_representations(X_ref, mX_ref, mY_ref, avX_ref, seq=seq_ref)
        else:
            Z_ref = predictor.get_latent_representations(X_ref, mX_ref, mY_ref, avX_ref, seq=None)

        # Position of pos within context window
        pos_in_window = int((pos - start) * (predictor.model.l2 / predictor.model.l1))
        Z_ref_pos = Z_ref[pos_in_window].cpu().numpy()

        # Iterate over offsets
        for i, offset in enumerate(offsets):
            new_pos = pos + offset
            start = new_pos - predictor.context_length//2
            end = new_pos + predictor.context_length//2
            if start < 0 or end > total_length:
                cosine_distances[idx, i] = np.nan
                euclidean_distances[idx, i] = np.nan
                manhattan_distances[idx, i] = np.nan
                continue
            X_window = X_flat[start:end]
            if DNA:
                seq_window = seq_flat[(start*predictor.resolution):(end*predictor.resolution)]
            
            X_window = X_window.unsqueeze(0)
            if DNA:
                seq_window = seq_window.unsqueeze(0)
            
            if DNA:
                Z = predictor.get_latent_representations(X_window, mX_ref, mY_ref, avX_ref, seq=seq_window)
            else:
                Z = predictor.get_latent_representations(X_window, mX_ref, mY_ref, avX_ref, seq=None)
            
            # Position of pos within the shifted context window
            pos_in_window_shifted = int((pos - start) * (predictor.model.l2 / predictor.model.l1))
            if pos_in_window_shifted == predictor.model.l2:
                pos_in_window_shifted -= 1
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
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_position_dependency_lineplots.png'))
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
    plt.savefig(os.path.join(output_dir, 'latent_position_dependency_heatmaps.png'))
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
    plt.savefig(os.path.join(output_dir, 'latent_position_dependency_average.png'))
    plt.close()

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




