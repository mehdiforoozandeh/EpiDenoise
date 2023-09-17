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
