from model import *
import pandas as pd

class Evaluation:
    def __init__(self):
        pass

    def mse(self, y_true, y_pred):
        """
        Calculate the genome-wide Mean Squared Error (MSE). This is a measure of the average squared difference 
        between the true and predicted values across the entire genome at a resolution of 25bp.
        """
        pass

    def pearson_correlation(self, y_true, y_pred):
        """
        Calculate the genome-wide Pearson Correlation. This measures the linear relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        pass

    def spearman_correlation(self, y_true, y_pred):
        """
        Calculate the genome-wide Spearman Correlation. This measures the monotonic relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        pass

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
        pass

    def mse1imp(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error at the top 1% of genomic positions ranked by predicted signal (mse1imp). 
        This is a measure of how well predictions match observations at positions with high predicted signal. 
        It's similar to precision.
        """
        pass

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
    results = pd.DataFrame(
        columns=
        ['celltype', 'feature', 'comparison', 'PCC_mean', 'PCC_stderr', 
        'spearman_rho_mean', 'spearman_rho_stderr', 'MSE_mean', 'MSE_stderr', 
        'MSE1obs_mean', 'MSE1obs_stderr', 'MSE1imp_mean', 'MSE1imp_stderr'])

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

                # Initialize a tensor to store all predictions
                p = torch.empty_like(x_t)

                # make predictions in batches
                for i in range(0, len(x_t), batch_size):
                    torch.cuda.empty_cache()
                    print(f"getting batches... {i/len(x_t):.2f}", )
                    
                    x_batch = x_t[i:i+batch_size]

                    if context_length < 8000:
                        rand_start = random.randint(0, 8000 - (context_length+1))
                        rand_end = rand_start + context_length

                        x_batch = x_batch[:, rand_start:rand_end, :]

                    x_batch = x_batch.to(device)
                    # all one pmask (no position is masked)
                    pmask = torch.ones((x_batch.shape[0], 1, 1, x_batch.shape[1]), device=device)
                    outputs = model(x_batch, pmask, fmask)

                    # Store the predictions in the large tensor
                    p[i:i+batch_size, :, :] = outputs.cpu()

                print("p shape", p.shape)
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

                    print("pred shape", pred.shape)
                    print("target shape", target.shape)

                    for i in range(target.shape[0]): # for each sample
                        metrics = evaluate(pred[i], target[i])
                        metrics_list.append(metrics)

                    metrics_mean = {k: np.mean([dic[k] for dic in metrics_list]) for k in metrics_list[0]}
                    metrics_stderr = {k: scipy.stats.sem([dic[k] for dic in metrics_list]) for k in metrics_list[0]}

                    # Add the results to the DataFrame
                    results = results.append({
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
                    }, ignore_index=True)

    # Save the DataFrame to a CSV file
    results.to_csv(outdir, index=False)

    return results

if __name__=="__main__":

    evaluate_epidenoise(
        model_path= "models/model_checkpoint_epoch_18.pth", 
        hyper_parameters_path= "models/hyper_parameters.pkl", 
        traindata_path="/project/compbio-lab/EIC/training_data/", 
        evaldata_path="/project/compbio-lab/EIC/validation_data/", 
        outdir="eval_results_E18.csv", 
        batch_size=20, context_length=1600)