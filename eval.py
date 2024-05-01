from model import *
from data import *
from scipy.stats import pearsonr, spearmanr, poisson, rankdata
from sklearn.metrics import mean_squared_error
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import pyBigWig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
PROC_GENE_BED_FPATH = "data/gene_bodies.bed"
PROC_PROM_BED_PATH = "data/tss.bed"

class METRICS(object):
    def __init__(self):
        pass

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

    def get_gene_signals(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=gene_df)
        pred_vals = self.get_signals(array=y_pred, df=gene_df)

        return gt_vals, pred_vals
    
    def get_prom_signals(self, y_true, y_pred, chrom='chr21', bin_size=25):
        assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=prom_df)
        pred_vals = self.get_signals(array=y_pred, df=prom_df)

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
                observed_values = eval_res[j]["obs"][gene_coord[0]:gene_coord[1]]
                imputed_values = eval_res[j]["imp"][gene_coord[0]:gene_coord[1]]

                # Plot the lines
                ax.plot(x_values, observed_values, color="blue", alpha=0.7, label="Observed", linewidth=0.1)
                ax.plot(x_values, imputed_values, "--", color="red", alpha=0.5, label="Imputed", linewidth=0.1)

                # Shade under the curves
                ax.fill_between(x_values, 0, observed_values, alpha=0.7, color="blue")
                ax.fill_between(x_values, 0, imputed_values, color="red", alpha=0.5)

                start_coord = gene_coord[0] * self.resolution
                end_coord = gene_coord[1] * self.resolution
                # Set title and labels for the top row and first column to avoid clutter
                ax.set_title(f"{eval_res[j]['feature']}_{eval_res[j]['comparison']}")
                ax.set_ylabel("arcsinh(-log10(pval))")

                ax.set_xlabel(f"chr21 {start_coord} : {end_coord}")
                ax.set_xticklabels([])

                custom_lines = [mlines.Line2D([], [], color='blue', label='Observed'),
                                mlines.Line2D([], [], color='red',  label='Imputed')]
                ax.legend(handles=custom_lines)

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_tracks.png", dpi=200)

    def BIOS_signal_scatter(self, eval_res, share_axes=True):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
            # Loop over each gene
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
                ax.set_xlabel("Obs | arcsinh(-log10(pval))")
                ax.set_ylabel("Imp | arcsinh(-log10(pval))")

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
                ax.set_xlabel("Obs")
                ax.set_ylabel("Imp")

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
                ax.set_xlabel("Obs | arcsinh(-log10(pval))")
                ax.set_ylabel("Imp | arcsinh(-log10(pval))")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_heatmaps.png", dpi=150)
        
    def BIOS_signal_scatter_rank(self, eval_res):
        if os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")==False:
            os.mkdir(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
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
                ax.set_xlabel("Obs | rank")
                ax.set_ylabel("Imp | rank")

        plt.tight_layout()
        plt.savefig(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/signal_rank_scatters.png", dpi=150)
    
    def BIOS_signal_rank_heatmap(self, eval_res, share_axes=True, bins=50):
        if not os.path.exists(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/"):
            os.makedirs(f"{self.savedir}/{eval_res[0]['bios']}_{eval_res[0]['available assays']}/")

        cols = ["GW", "gene", "TSS", "1obs", "1imp"]

        # Define the size of the figure
        plt.figure(figsize=(5 * len(cols), len(eval_res) * 5))

        for j in range(len(eval_res)):
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
                ax.set_xlabel("Obs | rank")
                ax.set_ylabel("Imp | rank")

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
        savedir="models/evals/", mode="eval"):

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
        self.dataset.init_eval(self.context_length, check_completeness=True)

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

    def get_metrics(self, P_imp, P_ups, Y, bios_name, availability):
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
        
        for j in availability:  # for each feature i.e. assay
            j = j.item()
            for comparison in ['imputed', 'upsampled']:
                if comparison == "imputed":
                    pred = P_imp[:, j].numpy()
                elif comparison == "upsampled":
                    pred = P_ups[:, j].numpy()

                target = Y[:, j].numpy()
                metrics_list = []

                # if np.var(pred) == 0: 
                #     print(f'skipped {self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"]} due to constant pred.')
                #     continue

                # corresp, corresp_deriv = self.metrics.correspondence_curve(target, pred)
                metrics = {
                    'bios':bios_name,
                    'feature': self.mark_dict[f"M{str(j+1).zfill(len(str(len(self.mark_dict))))}"],
                    'comparison': comparison,
                    'available assays': len(availability),

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
        P = torch.empty_like(X, device="cpu") 

        # make predictions in batches
        for i in range(0, len(X), self.batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+ self.batch_size]
            mX_batch = mX[i:i+ self.batch_size]
            mY_batch = mY[i:i+ self.batch_size]
            avail_batch = avail[i:i+ self.batch_size]

            with torch.no_grad():
                x_batch = x_batch.to(self.device)

                if len(imp_target)>0:
                    x_batch = x_batch.clone()
                    x_batch[:,:,imp_target] = self.token_dict["cloze_mask"]
                    x_batch_missing_vals = (x_batch == -1)
                    x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"] 

                    avail_batch = avail_batch.clone()
                    avail_batch[:, imp_target] = self.token_dict["cloze_mask"]
                    avail_batch_missing_vals = (avail_batch == 0)
                    avail_batch[avail_batch_missing_vals] = self.token_dict["cloze_mask"]

                    mX_batch = mX_batch.clone()
                    mX_batch[:,:,imp_target] = self.token_dict["missing_mask"]
                
                else:
                    # change missing token to cloze token to force prediction
                    x_batch_missing_vals = (x_batch == -1)
                    x_batch = x_batch.clone()
                    x_batch[x_batch_missing_vals] = self.token_dict["cloze_mask"] 

                    avail_batch_missing_vals = (avail_batch == 0)
                    avail_batch = avail_batch.clone()
                    avail_batch[avail_batch_missing_vals] = self.token_dict["cloze_mask"]

                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)

                outputs_p, outputs_n = self.model(x_batch, mX_batch, mY_batch, avail_batch)
                outputs = NegativeBinomial(outputs_p.cpu(), outputs_n.cpu()).expect(stat="median")

            # Store the predictions in the large tensor
            P[i:i+outputs.shape[0], :, :] = outputs
            print("one batch completed")

        return P

    def bios_pipeline(self, bios_name, x_dsf):
        X, mX, avX, Y, mY, avY = self.load_bios(bios_name, x_dsf)  

        available_indices = torch.where(avX[0, :] == 1)[0]

        P_imp = torch.empty_like(X, device="cpu") 
        for leave_one_out in available_indices:
            P_imp[:, :, leave_one_out] = self.pred(X, mX, mY, avX, imp_target=[leave_one_out])[:, :, leave_one_out]
        
        P_ups = self.pred(X, mX, mY, avX, imp_target=[])

        P_imp = P_imp.view((P_imp.shape[0] * P_imp.shape[1]), P_imp.shape[-1]) # imp_preds
        P_ups = P_ups.view((P_ups.shape[0] * P_ups.shape[1]), P_ups.shape[-1]) # ups_preds

        Y = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1]) 
        X = X.view((X.shape[0] * X.shape[1]), X.shape[-1]) 

        eval_res = self.get_metrics(P_imp, P_ups, Y, bios_name, available_indices)
        return eval_res

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

        # try: 
        # print("plotting signal tracks")
        # self.viz.BIOS_signal_track(eval_res)
        # self.viz.clear_pallete()
        # except:
        #     print("faild to plot signal tracks")

        # try:
        # print("plotting context_specific performance")
        # self.viz.BIOS_context_length_specific_performance(eval_res, self.context_length, bins=10)
        # self.viz.clear_pallete()
        # except:
        #     print("faild to plot context_specific performance")
            
        # try:
        print("plotting signal scatter")
        self.viz.BIOS_signal_scatter(eval_res)
        self.viz.clear_pallete()
        # except:
        #     print("faild to plot  signal scatter")

        # try:
        print("plotting signal scatter with marginals")
        self.viz.BIOS_signal_scatter_with_marginals(eval_res)
        self.viz.clear_pallete()
        # except:
        #     print("faild to plot scatter with marginals")

        # try:
        print("plotting signal heatmap")
        self.viz.BIOS_signal_heatmap(eval_res)
        self.viz.clear_pallete()
        # except:
        #     print("faild to plot  signal heatmap")

        # try:
        print("plotting signal rank heatmap")
        self.viz.BIOS_signal_rank_heatmap(eval_res)
        self.viz.clear_pallete()
        # except:
        #     print("faild to plot  signal rank heatmap")

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
        for bios in self.dataset.test_bios:
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

if __name__=="__main__":
    e = EVAL_EED(
        model="models/EPD30a_model_checkpoint_epoch0.pth", 
        data_path="/project/compbio-lab/encode_data/", 
        context_length=400, batch_size=100, 
        hyper_parameters_path="models/hyper_parameters30a_EpiDenoise30a_20240428232813_params9408914.pkl",
        train_log={}, chr_sizes_file="data/hg38.chrom.sizes", 
        version="30a", resolution=25, 
        savedir="models/eval_30a/", mode="eval"
    )
    evres = e.bios_pipeline("ENCBS708DHS", 2)
    print(evres)
    exit()
    # df.to_csv("models/eval_30a/eval.csv")
    # e.viz_bios(evres)

    exit()

    e = EVAL(
        model= "models/EpiDenoise22_20240309024015_params1382403.pt", 
        hyper_parameters_path="models/hyper_parameters22_EpiDenoise22_20240309024015_params1382403.pkl",
        traindata_path="/project/compbio-lab/EIC/training_data/", 
        evaldata_path="/project/compbio-lab/EIC/validation_data/", 
        context_length=200, batch_size=300, is_arcsin=True,
        version="22", savedir="models/epd22_evals/")

    e.viz_all()
    exit()

    e = EVAL(
        model= "models/EpiDenoise22_20240301194644_params1048755.pt", 
        hyper_parameters_path="models/hyper_parameters22_EpiDenoise22_20240301194644_params1048755.pkl",
        traindata_path="/project/compbio-lab/EIC/training_data/", 
        evaldata_path="/project/compbio-lab/EIC/validation_data/", 
        context_length=200, batch_size=300,
        version="22", savedir="models/testevals/", mode="dev")
    
    e.bios_test()
    exit()

    
    



   