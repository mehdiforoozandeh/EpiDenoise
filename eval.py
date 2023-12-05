from model import *
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import scipy.stats
import pyBigWig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

class Evaluation: # on chr21
    def __init__(
        self, model_path, hyper_parameters_path, 
        traindata_path, evaldata_path, 
        resolution=25, chr_sizes_file="data/hg38.chrom.sizes"):

        self.traindata_path = traindata_path
        self.evaldata_path = evaldata_path

        with open(hyper_parameters_path, 'rb') as f:
            self.hyper_parameters = pickle.load(f)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = load_epidenoise(model_path, self.hyper_parameters)
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

        X = X.view(-1, context_length, X.shape[-1])
        Y = Y.view(-1, context_length, Y.shape[-1])

        d_model = X.shape[-1]
        fmask = torch.ones(d_model, self.hyper_parameters["hidden_dim"])

        for i in missing_x_i: # input fmask
            fmask[i,:] = 0
        
        fmask = fmask.to(self.device)
        # Initialize a tensor to store all predictions
        P = torch.empty_like(X, device="cpu")

        # make predictions in batches
        for i in range(0, len(X), batch_size):
            torch.cuda.empty_cache()
            
            x_batch = X[i:i+batch_size]

            with torch.no_grad():
                # all one pmask (no position is masked)
                pmask = torch.ones((x_batch.shape[0], 1, 1, x_batch.shape[1]), device=self.device)
                x_batch = x_batch.to(self.device)
                outputs = self.model(x_batch, pmask, fmask)

            # Store the predictions in the large tensor
            P[i:i+batch_size, :, :] = outputs.cpu()
        
        P = P.view((P.shape[0] * context_length), P.shape[-1]) # preds
        Y = Y.view((Y.shape[0] * context_length), Y.shape[-1]) # eval data
        X = X.view((X.shape[0] * context_length), X.shape[-1]) # train data

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
                'PCC_mean': self.pearson_correlation(target, pred),
                'spearman_rho_mean': self.spearman_correlation(target, pred),
                'MSE_mean': self.mse(target, pred),
                'MSE1obs_mean': self.mse1obs(target, pred),
                'MSE1imp_mean': self.mse1imp(target, pred), 
                'pred_mean_min_max': (pred.mean(), pred.max(), pred.min()),
                'target_mean_min_max': (target.mean(), target.max(), target.min())
            }
            # print(metrics)
            # Add the results to the DataFrame
            self.results.append(metrics)

    def evaluate_model(self, outdir):
        for bios in self.eval_data.keys():
            print("evaluating ", bios)
            self.evaluate_biosample(bios)

        self.results.to_csv(outdir, index=False)

    def mse(self, y_true, y_pred):
        """
        Calculate the genome-wide Mean Squared Error (MSE). This is a measure of the average squared difference 
        between the true and predicted values across the entire genome at a resolution of 25bp.
        """
        return mean_squared_error(y_pred, y_true)

    def pearson_correlation(self, y_true, y_pred):
        """
        Calculate the genome-wide Pearson Correlation. This measures the linear relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        return pearsonr(y_pred, y_true)[0]

    def spearman_correlation(self, y_true, y_pred):
        """
        Calculate the genome-wide Spearman Correlation. This measures the monotonic relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        return spearmanr(y_pred, y_true)[0]

    def mse_prom(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error in promoter regions (MSEProm). This is a measure of the average squared 
        difference between the true and predicted values in promoter regions defined as ±2kb from the start of GENCODEv38 annotated genes.
        """
        pass

    def mse_gene(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error in gene bodies (MSEGene). This is a measure of the average squared difference
         between the true and predicted values in gene bodies from GENCODEv38 annotated genes.
        """
        pass

    def mse_enh(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error in enhancer regions (MSEEnh). This is a measure of the average squared difference
        between the true and predicted values in enhancer regions as defined by FANTOM5 annotated permissive enhancers.
        """
        pass

    def weighted_mse(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error weighted at each position by the variance of the experimental signal (Weighted MSE).
        This is a measure of the average squared difference between the true and predicted values, where each position is 
        weighted by its variance across experiments.
        """
        pass

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

def evaluate_epidenoise(model_path, hyper_parameters_path, traindata_path, evaldata_path, outdir, batch_size=20, context_length=1600):  
    with open(hyper_parameters_path, 'rb') as f:
        hyper_parameters = pickle.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_epidenoise(model_path, hyper_parameters)
    print(f"# model_parameters: {count_parameters(model)}")

    train_data = ENCODE_IMPUTATION_DATASET(traindata_path)
    eval_data = ENCODE_IMPUTATION_DATASET(evaldata_path)
    all_assays = train_data.all_assays
    model.eval()  # set the model to evaluation mode

    # Initialize a DataFrame to store the results
    results = []

    for b_e in eval_data.biosamples.keys():
        for b_t in train_data.biosamples.keys():
            if b_e == b_t: # id biosamples that are present at both training and eval files

                # file path
                f_t = train_data.biosamples[b_t]
                f_e = eval_data.biosamples[b_e]
                
                x_t, missing_mask, missing_f_ind_t = train_data.get_biosample(f_t)
                y_e, missing_mask, missing_f_ind_e = eval_data.get_biosample(f_e)

                del missing_mask

                d_model = x_t.shape[2]
                fmask = torch.ones(d_model, d_model)

                for i in missing_f_ind_t: # input fmask
                    fmask[i,:] = 0
                
                fmask = fmask.to(device)

                if context_length < 8000:
                    x_t = x_t[:, :context_length, :]
                    y_e = y_e[:, :context_length, :]
                
                print(f"shape of inputs {x_t.shape}, shape of targets {y_e.shape}")

                # Initialize a tensor to store all predictions
                p = torch.empty_like(x_t, device="cpu")

                # make predictions in batches
                for i in range(0, len(x_t), batch_size):
                    torch.cuda.empty_cache()
                    if i/len(x_t) % 20 ==0:
                        print(f"getting batches... {i/len(x_t):.2f}", )
                    
                    x_batch = x_t[i:i+batch_size]

                    with torch.no_grad():
                        # all one pmask (no position is masked)
                        pmask = torch.ones((x_batch.shape[0], 1, 1, x_batch.shape[1]), device=device)
                        x_batch = x_batch.to(device)
                        outputs = model(x_batch, pmask, fmask)

                    # Store the predictions in the large tensor
                    p[i:i+batch_size, :, :] = outputs.cpu()
                # Evaluate the model's performance on the entire tensor
                for j in range(y_e.shape[2]):  # for each feature i.e. assay
                    pred = p[:, :, j].numpy()
                    metrics_list = []

                    if j in missing_f_ind_t and j not in missing_f_ind_e:  # if the feature is missing in the input
                        target = y_e[:, :, j].numpy()
                        comparison = 'imputed'
                    
                    elif j not in missing_f_ind_t:
                        target = x_t[:, :, j].numpy()
                        comparison = 'denoised'

                    else:
                        continue

                    for i in range(target.shape[0]): # for each sample
                        metrics = evaluate(pred[i], target[i])
                        metrics_list.append(metrics)

                    metrics_mean = {k: np.mean([dic[k] for dic in metrics_list]) for k in metrics_list[0]}
                    metrics_stderr = {k: scipy.stats.sem([dic[k] for dic in metrics_list]) for k in metrics_list[0]}

                    # Add the results to the DataFrame
                    results.append({
                        'celltype': b_e,
                        'feature': all_assays[j],
                        'comparison': comparison,
                        'PCC_mean': metrics_mean['PCC'],
                        'PCC_stderr': metrics_stderr['PCC'],
                        'spearman_rho_mean': metrics_mean['spearman_rho'],
                        'spearman_rho_stderr': metrics_stderr['spearman_rho'],
                        'MSE_mean': metrics_mean['MSE'],
                        'MSE_stderr': metrics_stderr['MSE'],
                        'MSE1obs_mean': metrics_mean['MSE1obs'],
                        'MSE1obs_stderr': metrics_stderr['MSE1obs'],
                        'MSE1imp_mean': metrics_mean['MSE1imp'],
                        'MSE1imp_stderr': metrics_stderr['MSE1imp']
                    })
    
    results = pd.DataFrame(results, columns=['celltype', 'feature', 'comparison', 'PCC_mean', 'PCC_stderr', 
        'spearman_rho_mean', 'spearman_rho_stderr', 'MSE_mean', 'MSE_stderr', 
        'MSE1obs_mean', 'MSE1obs_stderr', 'MSE1imp_mean', 'MSE1imp_stderr'])

    # Save the DataFrame to a CSV file
    results.to_csv(outdir, index=False)

    return results

if __name__=="__main__":
    # e = Evaluation(
    #     model_path= "models/model_checkpoint_bios_5.pth", 
    #     hyper_parameters_path= "models/hyper_parameters.pkl", 
    #     traindata_path="/project/compbio-lab/EIC/training_data/", 
    #     evaldata_path="/project/compbio-lab/EIC/validation_data/"
    # )
    # e.evaluate_model("eval_Ep5.csv")

    e = Evaluation(
        model_path= "models/model_checkpoint_bios_11.pth", 
        hyper_parameters_path= "models/hyper_parameters.pkl", 
        traindata_path="/project/compbio-lab/EIC/training_data/", 
        evaldata_path="/project/compbio-lab/EIC/validation_data/"
    )
    e.evaluate_model("eval_Ep11.csv")

    # e = Evaluation(
    #     model_path= "models/model_checkpoint_bios_1.pth", 
    #     hyper_parameters_path= "models/hyper_parameters.pkl", 
    #     traindata_path="/project/compbio-lab/EIC/training_data/", 
    #     evaldata_path="/project/compbio-lab/EIC/validation_data/"
    # )
    # e.evaluate_model("eval_Ep1.csv")