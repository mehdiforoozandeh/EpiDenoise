from model import *
from data import *
from _utils import *
from scipy.stats import pearsonr, spearmanr, poisson, rankdata
from sklearn.metrics import mean_squared_error, r2_score, auc
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.svm import SVR

import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import pyBigWig

import scipy.stats as stats
from scipy.optimize import minimize

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
PROC_GENE_BED_FPATH = "data/gene_bodies.bed"
PROC_PROM_BED_PATH = "data/tss.bed"

def auc_rec(y_true, y_pred):
    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)
    
    # Sort the errors
    sorted_errors = np.sort(errors)
    
    # Calculate cumulative proportion of points within error tolerance
    cumulative_proportion = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    # Calculate AUC-REC
    auc_rec = auc(sorted_errors, cumulative_proportion)

    # Determine the maximum possible area under the curve
    max_error = sorted_errors[-1]  # Maximum error tolerance observed
    max_area = max_error * 1  # Since the cumulative proportion ranges from 0 to 1
    
    # Normalize AUC-REC
    normalized_auc_rec = auc_rec / max_area if max_area > 0 else 0

    return normalized_auc_rec

def k_fold_cross_validation(data, k=10, target='TPM', logscale=True, model_type='linear'):
    """
    Perform k-fold cross-validation for linear regression on the provided data.
    
    Args:
        data (pd.DataFrame): The DataFrame containing features and labels.
        k (int): The number of folds for cross-validation.
        target (str): The label to predict ('TPM' or 'FPKM'). Default is 'TPM'.
        
    Returns:
        avg_mse (float): The average Mean Squared Error across all folds.
        avg_r2 (float): The average R-squared value across all folds.
    """
    # Get unique gene IDs
    unique_gene_ids = data["geneID"].unique()
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    if logscale:
        data["TPM"] = np.log(data["TPM"]+1)
    
    mse_scores = []
    r2_scores = []
    auc_recs = []
    pearsonrs = []
    spearmanrs = []

    all_errors = []

    # Perform K-Fold Cross Validation
    for train_index, test_index in kf.split(unique_gene_ids):
        train_gene_ids = unique_gene_ids[train_index]
        test_gene_ids = unique_gene_ids[test_index]

        # Split the data into training and testing sets
        train_data = data[data["geneID"].isin(train_gene_ids)]
        test_data = data[data["geneID"].isin(test_gene_ids)]
    
        # Extract features and labels
        X_train = train_data[["promoter_signal", "gene_body_signal", "TES_signal"]]
        y_train = train_data[target]
        
        X_test = test_data[["promoter_signal", "gene_body_signal", "TES_signal"]]
        y_test = test_data[target]

        # Initialize the model based on the model type
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        elif model_type == 'svr':
            model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        elif model_type == 'loess':
            # LOESS (Local Regression) model
            lowess_predictions = lowess(y_train, X_train.mean(axis=1), frac=0.3)
            # For LOESS, we use an averaging of predictions as there is no direct fit/predict method
            y_pred = lowess_predictions[:, 1]
        
        # Collect errors from this fold
        errors = np.abs(y_test - y_pred)
        all_errors.extend(errors)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        aucrec = auc_rec(y_test, y_pred)
        pcc = pearsonr(y_pred, y_test)[0]
        scrr = spearmanr(y_pred, y_test)[0]
        
        mse_scores.append(mse)
        r2_scores.append(r2)
        auc_recs.append(aucrec)
        pearsonrs.append(pcc)
        spearmanrs.append(scrr)

    # Compute average metrics
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    avg_aucrec = np.mean(auc_recs)
    avg_pcc = np.mean(pearsonrs)
    avg_srcc = np.mean(spearmanrs)

    all_errors = np.array(all_errors)

    return {
        'avg_mse': avg_mse,
        'avg_r2': avg_r2,
        'avg_aucrec': avg_aucrec,
        'avg_pcc': avg_pcc,
        'avg_srcc': avg_srcc,
        "errors": all_errors
    }

def binarize_nbinom(data, threshold=0.0001):
    """
    Fits a Negative Binomial distribution to the data, binarizes the data based on a threshold.

    Parameters:
    data (numpy array): The input data array to analyze.
    threshold (float): The probability threshold for binarization. Default is 0.0001.
    """
    # Ensure the data is a NumPy array
    data = np.asarray(data)

    # Step 1: Fit Negative Binomial Distribution
    def negbinom_loglik(params, data):
        r, p = params
        return -np.sum(stats.nbinom.logpmf(data, r, p))

    initial_params = [1, 0.5]  # Initial guess for r and p
    result_nbinom = minimize(negbinom_loglik, initial_params, args=(data), bounds=[(1e-5, None), (1e-5, 1-1e-5)])
    r, p = result_nbinom.x

    # Step 2: Calculate Probabilities under the Negative Binomial distribution
    probabilities = stats.nbinom.pmf(data, r, p)

    # Step 3: Binarize the Data
    binarized_data = (probabilities < threshold).astype(int)

    return binarized_data

class METRICS(object):
    def __init__(self, chrom='chr21', bin_size=25):
        self.prom_df = self.get_prom_positions(chrom, bin_size)
        self.gene_df = self.get_gene_positions(chrom, bin_size)

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
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in df.iterrows()])
        valid_indices = indices[indices < len(array)]

        signals = array[valid_indices]
        return signals

    ################################################################################

    def get_gene_signals(self, y_true, y_pred, bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return gt_vals, pred_vals
    
    def get_prom_signals(self, y_true, y_pred, bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return gt_vals, pred_vals
    
    def get_1obs_signals(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return y_true[perc_99_pos], y_pred[perc_99_pos]

    def get_1imp_signals(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return y_true[perc_99_pos], y_pred[perc_99_pos]
    
    ################################################################################
    def r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def r2_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return self.r2(y_true=gt_vals, y_pred=pred_vals)

    def r2_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return self.r2(y_true=gt_vals, y_pred=pred_vals)

    def r2_1obs(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return self.pearson(y_true[perc_99_pos], y_pred[perc_99_pos])

    def r2_1imp(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return self.r2(y_true[perc_99_pos], y_pred[perc_99_pos])

    def mse(self, y_true, y_pred):
        """
        Calculate the genome-wide Mean Squared Error (MSE). This is a measure of the average squared difference 
        between the true and predicted values across the entire genome at a resolution of 25bp.
        """
        return np.mean((np.array(y_true) - np.array(y_pred))**2)
    
    def mse_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return self.mse(y_true=gt_vals, y_pred=pred_vals)

    def pearson_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return self.pearson(y_true=gt_vals, y_pred=pred_vals)

    def spearman_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

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
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return self.mse(y_true=gt_vals, y_pred=pred_vals)

    def pearson_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return self.pearson(y_true=gt_vals, y_pred=pred_vals)

    def spearman_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

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
        if p == 0:
            return 0

        elif p == 1:
            return 1

        top_p_percent = int(p * len(y_true))

        # Get the indices of the top p percent of the observed (true) values
        top_p_percent_obs_i = np.argsort(y_true)[-top_p_percent:]
        
        # Get the indices of the top p percent of the predicted values
        top_p_percent_pred_i = np.argsort(y_pred)[-top_p_percent:]

        # Calculate the overlap
        overlap = len(np.intersect1d(top_p_percent_obs_i, top_p_percent_pred_i))

        # Calculate the percentage of overlap
        overlap_percent = overlap / top_p_percent 

        return overlap_percent

    def correspondence_curve(self, y_true, y_pred):
        curve = []
        derivatives = []
        steps = [float(p / 100) for p in range(0, 101, 1)]

        obs_rank = np.argsort(y_true)
        pred_rank = np.argsort(y_pred)

        for p in steps:
            if p == 0 or p == 1:
                overlap_percent = p
            else:
                top_p_percent = int(p * len(y_true))
                top_p_percent_obs_i = obs_rank[-top_p_percent:]
                top_p_percent_pred_i = pred_rank[-top_p_percent:]

                overlap = len(np.intersect1d(top_p_percent_obs_i, top_p_percent_pred_i))
                overlap_percent = overlap / len(y_true)

            curve.append((p, overlap_percent))

        # Calculate derivatives using finite differences
        for i in range(1, len(curve)):
            dp = curve[i][0] - curve[i-1][0]
            d_overlap_percent = curve[i][1] - curve[i-1][1]
            derivative = d_overlap_percent / dp
            derivatives.append((curve[i][0], derivative))

        return curve, derivatives

    def confidence_quantile(self, nbinom_p, nbinom_n, y_true):
        nbinom_dist = NegativeBinomial(nbinom_p, nbinom_n)
        return nbinom_dist.cdf(y_true)

    def foreground_vs_background(self, nbinom_p, nbinom_n, y_true):
        """
        inputs: 1) nbinom_p, nbinom_n -> two arrays with length L -> negative binomial dist parameters for each position
                2) y_true -> one array of true observed signal

        task:

            - NEW 2: peak vs. background comparison
            - binarize each observed experiment according to some threshold.
            - measure the two following
                - what fraction of positions outside peaks have overlap (confidence interval) with zero
                - for peaks, for what fraction, the confidence interval overlap with 0. this should ideally be low since otherwise the model is not sure about the peak
            - intervals 90 95 99 percent
        """

        nbinom_dist = NegativeBinomial(nbinom_p, nbinom_n)
        binarized_y = binarize_nbinom(y_true)

        pmf_zero = (nbinom_dist.pmf(0))

        analysis = {}

        background_pmf_zero = pmf_zero[binarized_y == 0].mean() 
        peak_pmf_zero = pmf_zero[binarized_y == 1].mean() 

        analysis["p0_bg"] = background_pmf_zero.item()
        analysis["p0_fg"] = peak_pmf_zero.item()

        return analysis


    ################################################################################

    """
    to do -- eval:
        1. binary classification eval (aucPR, aucROC) on peak called signals (imp vs obs)
        2. SAGA on a subset of tracks + SAGAconf (imp vs obs)
        3. sum(abs(log(derivative of correspondence curve))) --> near zero is better 
    """
 
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
            ranges = [(0, x_90), (0, x_99), (0, error.max())]

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

    def BIOS_signal_scatter_with_marginals(self, eval_res, share_axes=True):
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

                bin_range = np.linspace(min(np.concatenate([xs, ys])), max(np.concatenate([xs, ys])), 50)
                ax_histx = ax.inset_axes([0, 1.05, 1, 0.2])
                ax_histy = ax.inset_axes([1.05, 0, 0.2, 1])
                
                ax_histx.hist(xs, bins=bin_range, alpha=0.9, color="#f25a64")
                ax_histy.hist(ys, bins=bin_range, orientation='horizontal', alpha=0.9, color="#f25a64")
                
                ax_histx.set_xticklabels([])
                ax_histx.set_yticklabels([])
                ax_histy.set_xticklabels([])
                ax_histy.set_yticklabels([])

                # Set title, labels, and range if share_axes is True
                ax.set_title(f"{result['feature']}_{c}_{result['comparison']}_{pcc}")
                ax.set_xlabel("Observed")
                ax.set_ylabel("Predicted")

                if share_axes:
                    common_range = [min(np.concatenate([xs, ys])), max(np.concatenate([xs, ys]))]
                    ax.set_xlim(common_range)
                    ax.set_ylim(common_range)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_scatters_with_marginals.png", dpi=150)

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

class EVAL_CANDI(object):
    def __init__(
        self, model, data_path, context_length, batch_size, hyper_parameters_path="",
        train_log={}, chr_sizes_file="data/hg38.chrom.sizes", resolution=25, 
        savedir="models/evals/", mode="eval", split="test", eic=False, DNA=False):

        self.savedir = savedir
        if os.path.exists(self.savedir) == False:
            os.mkdir(self.savedir)

        self.data_path = data_path
        self.version = version
        self.context_length = context_length
        self.batch_size = batch_size
        self.resolution = resolution
        
        self.eic = eic
        self.DNA = DNA


        self.model = model
        self.dataset = ExtendedEncodeDataHandler(self.data_path, resolution=self.resolution)
        self.dataset.init_eval(
            self.context_length, check_completeness=True, split=split, bios_min_exp_avail_threshold=5, eic=eic)

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

    def pred(self, X, mX, mY, avail, imp_target=[], seq=None):
        # Initialize a tensor to store all predictions
        n = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        p = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        mu = torch.empty_like(X, device="cpu", dtype=torch.float32) 
        var = torch.empty_like(X, device="cpu", dtype=torch.float32) 

        # make predictions in batches
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
                # mY_batch_missing_vals = (mY_batch == self.token_dict["missing_mask"])
                avail_batch_missing_vals = (avail_batch == 0)

                x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing_vals] = self.token_dict["cloze_mask"]
                # mY_batch[mY_batch_missing_vals] = self.token_dict["cloze_mask"]

                if len(imp_target)>0:
                    x_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    mX_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    # mY_batch[:, :, imp_target] = self.token_dict["cloze_mask"]
                    avail_batch[:, imp_target] = 0

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                if self.DNA:
                    seq_batch = seq_batch.to(self.device)
                    outputs_p, outputs_n, outputs_mu, outputs_var = self.model(x_batch.float(), seq_batch, mX_batch, mY_batch, avail_batch)
                else:
                    outputs_p, outputs_n, outputs_mu, outputs_var = self.model(x_batch.float(), mX_batch, mY_batch, avail_batch)


            # Store the predictions in the large tensor
            n[i:i+outputs_n.shape[0], :, :] = outputs_n.cpu()
            p[i:i+outputs_p.shape[0], :, :] = outputs_p.cpu()
            mu[i:i+outputs_mu.shape[0], :, :] = outputs_mu.cpu()
            var[i:i+outputs_var.shape[0], :, :] = outputs_var.cpu()

            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n, outputs_mu, outputs_var  # Free up memory
            torch.cuda.empty_cache()  # Free up GPU memory

        return n, p, mu, var

    def get_metrics(self, imp_count_dist, ups_count_dist, imp_pval_dist, ups_pval_dist, Y, P, bios_name, availability):
        imp_count_mean = imp_count_dist.expect()
        ups_count_mean = ups_count_dist.expect()

        imp_count_std = imp_count_dist.std()
        ups_count_std = ups_count_dist.std()

        imp_pval_mean = imp_pval_dist.mean()
        ups_pval_mean = ups_pval_dist.mean()

        imp_pval_std = imp_pval_dist.std()
        ups_pval_std = ups_pval_dist.std()

        if self.dataset.has_rnaseq(bios_name):
            print("got rna-seq data")
            rnaseq_res = self.eval_rnaseq(bios_name, ups_mean, Y, availability, k_fold=10, plot_REC=True)

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
    
 



if __name__=="__main__":

    e = EVAL_EED(
        model="/project/compbio-lab/EPD/pretrained/EPD30d_model_checkpoint_Jul8th.pth", 
        data_path="/project/compbio-lab/encode_data/", 
        context_length=3200, batch_size=50, 
        hyper_parameters_path="/project/compbio-lab/EPD/pretrained/hyper_parameters30d_EpiDenoise30d_20240710133714_params237654660.pkl",
        train_log={}, chr_sizes_file="data/hg38.chrom.sizes", 
        version="30d", resolution=25, savedir="/project/compbio-lab/EPD/eval_30d/", mode="eval")
    
    print(e.bios_pipeline("ENCBS343AKO", x_dsf=1))
    # e.viz_all()

    exit()

    # e = EVAL_EED(
    #     model="models/pretrained/EPD30a_model_checkpoint_epoch0_LociProg60.pth", 
    #     data_path="/project/compbio-lab/encode_data/", 
    #     context_length=400, batch_size=200, 
    #     hyper_parameters_path="models/pretrained/hyper_parameters30a_EpiDenoise30a_20240529134015_params2182872.pkl",
    #     train_log={}, chr_sizes_file="data/hg38.chrom.sizes", 
    #     version="30a", resolution=25, 
    #     savedir="models/evals/eval_30a/", mode="eval"
    # )
    # evres = e.bios_pipeline("ENCBS899TTJ", 1)
    # for i in range(len(evres)):
    #     print(evres[i])
    
    # exit()

    # e.viz_bios(evres)
    # try:
    #     evres = pd.DataFrame(evres)
    #     evres.to_csv("models/eval_30a/res.csv")
    # except:
    #     pass

    e = EVAL_EED(
        model="models/pretrained/EpiDenoise30b_20240529133959_params5969560.pt", 
        data_path="/project/compbio-lab/encode_data/", 
        context_length=1600, batch_size=50, 
        hyper_parameters_path="models/pretrained/hyper_parameters30b_EpiDenoise30b_20240529133959_params5969560.pkl",
        train_log={}, chr_sizes_file="data/hg38.chrom.sizes", 
        version="30b", resolution=25, savedir="models/eval_30b/", mode="eval")

    evres = e.bios_pipeline("ENCBS373AAA", 1)
    print("plotting error vs. std hexbin")
    try:
        e.viz.BIOS_error_std_hexbin(evres)
        e.viz.clear_pallete()
    except Exception as e:
        print(f"Failed to plot error vs. std hexbin: {e}")
    # e.viz_all()


    exit()
    for i in range(len(evres)):
        print(evres[i])

    

    evres = e.bios_pipeline("ENCBS596CTT", 1)
    for i in range(len(evres)):
        print(evres[i])

    e.viz_bios(evres)
    try:
        evres = pd.DataFrame(evres)
        evres.to_csv("models/eval_30b/res.csv")
    except:
        pass


    # e = EVAL_EED(
    #     model="models/EPD30c_model_checkpoint_epoch0_LociProg20.pth", 
    #     data_path="/project/compbio-lab/encode_data/", 
    #     context_length=1536, batch_size=50, 
    #     hyper_parameters_path="models/hyper_parameters30c_EpiDenoise30c_20240603115718_params15099287.pkl",
    #     train_log={}, chr_sizes_file="data/hg38.chrom.sizes", 
    #     version="30c", resolution=25, 
    #     savedir="models/eval_30c/", mode="eval"
    # )
    # evres = e.bios_pipeline("ENCBS596CTT", 1)
    # for i in range(len(evres)):
    #     print(evres[i])

    # e.viz_bios(evres)
    # try:
    #     evres = pd.DataFrame(evres)
    #     evres.to_csv("models/eval_30a/res.csv")
    # except:
    #     pass
    

    

    
    

