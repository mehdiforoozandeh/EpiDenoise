from model import *
from scipy.stats import pearsonr, spearmanr, poisson
from sklearn.metrics import mean_squared_error
import scipy.stats
import pyBigWig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
PROC_GENE_BED_FPATH = "data/gene_bodies.bed"
PROC_PROM_BED_PATH = "data/tss.bed"

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
        context_length, batch_size = self.hyper_parameters["context_length"], self.hyper_parameters["batch_size"]
        X, missing_x_i = self.load_biosample(bios_name, mode="train")
        Y, missing_y_i = self.load_biosample(bios_name, mode="eval")

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

    def evaluate_model(self, outdir):
        for bios in self.eval_data.keys():
            print("evaluating ", bios)
            self.evaluate_biosample(bios)

        self.results = pd.DataFrame(self.results)
        self.results.to_csv(outdir, index=False)

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

    def peak_overlap(self, y_true, y_pred, threshold=0.01):
        # Calculate the p-value of the Poisson distribution for both y_true and y_pred
        pvalue_true = 1 - poisson.cdf(y_true, np.mean(y_true))
        pvalue_pred = 1 - poisson.cdf(y_pred, np.mean(y_pred))

        # Binarize y_true and y_pred according to the pvalue and the threshold
        y_true_bin = np.where(pvalue_true < threshold, 1, 0)
        y_pred_bin = np.where(pvalue_pred < threshold, 1, 0)

        # Calculate and return the ratio of 1s in y_true for which the corresponding y_pred is also 1
        overlap = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        total = np.sum(y_true_bin == 1)

        return overlap / total if total > 0 else 0

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
    
if __name__=="__main__":
    # e = Evaluation(
    #     model_path= "models/EpiDenoise17_20240102154151_params3977253.pt", 
    #     hyper_parameters_path= "models/hyper_parameters17_EpiDenoise17_20240102154151_params3977253.pkl", 
    #     traindata_path="/project/compbio-lab/EIC/training_data/", 
    #     evaldata_path="/project/compbio-lab/EIC/validation_data/", 
    #     is_arcsin=True,  version="17"
    # )
    # e.evaluate_model("eval_EPD17.csv")

    e = Evaluation(
        model_path= "models/EpiDenoise16_20240105145712_params157128.pt", 
        hyper_parameters_path= "models/hyper_parameters16_EpiDenoise16_20240105145712_params157128.pkl", 
        traindata_path="/project/compbio-lab/EIC/training_data/", 
        evaldata_path="/project/compbio-lab/EIC/validation_data/", 
        is_arcsin=True, version="16"
    )
    e.evaluate_model("eval_EPD16.csv")

    exit()
    e = Evaluation(
        model_path= "models/EpiDenoise_20231210014829_params154531.pt", 
        hyper_parameters_path= "models/hyper_parameters_EpiDenoise_20231210014829_params154531.pkl", 
        traindata_path="/project/compbio-lab/EIC/training_data/", 
        evaldata_path="/project/compbio-lab/EIC/validation_data/", 
        is_arcsin=True
    )
    e.evaluate_model("eval_small_model_L400.csv")

    e = Evaluation(
        model_path= "models/EpiDenoise_20231212191632_params154531.pt", 
        hyper_parameters_path= "models/hyper_parameters_EpiDenoise_20231212191632_params154531.pkl", 
        traindata_path="/project/compbio-lab/EIC/training_data/", 
        evaldata_path="/project/compbio-lab/EIC/validation_data/",
        is_arcsin=True
    )
    e.evaluate_model("eval_small_model_L800.csv")

    e = Evaluation(
        model_path= "models/EpiDenoise_20231210014829_params154531.pt", 
        hyper_parameters_path= "models/hyper_parameters_EpiDenoise_20231210014829_params154531.pkl", 
        traindata_path="/project/compbio-lab/EIC/training_data/", 
        evaldata_path="/project/compbio-lab/EIC/blind_data/", 
        is_arcsin=True
    )
    e.evaluate_model("eval_small_model_L400_blind.csv")

    e = Evaluation(
        model_path= "models/EpiDenoise_20231212191632_params154531.pt", 
        hyper_parameters_path= "models/hyper_parameters_EpiDenoise_20231212191632_params154531.pkl", 
        traindata_path="/project/compbio-lab/EIC/training_data/", 
        evaldata_path="/project/compbio-lab/EIC/blind_data/",
        is_arcsin=True
    )
    e.evaluate_model("eval_small_model_L800_blind.csv")