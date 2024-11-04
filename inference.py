import torch
import pickle
import os
from CANDI import *


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

    def pred_cropped(self, X, mX, mY, avail, imp_target=[], seq=None):
        """
        given 
        """
        pass

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

   

if __name__ == "__main__":
    model_path = "models/CANDIeic_DNA_random_mask_oct17-expan2_model_checkpoint_epoch5.pth"
    hyper_parameters_path = "models/hyper_parameters_eic_DNA_random_mask_oct17-expan2_CANDIeic_DNA_random_mask_oct17-expan2_20241017130209_params14059878.pkl"
    dataset_path = "/project/compbio-lab/encode_data/"
    output_dir = "output"
    number_of_states = 10
    DNA = True

    CANDIP = CANDIPredictor(
        model_path, hyper_parameters_path, number_of_states, data_path=dataset_path, DNA=DNA, split="test", chr="chr21", resolution=25)
    
    os.makedirs(output_dir, exist_ok=True)

    if DNA:
        X, Y, P, seq, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
    else:
        X, Y, P, mX, mY, avX, avY = CANDIP.load_bios(bios_name, x_dsf=dsf)
        seq = None
        
    print(X.shape, Y.shape, P.shape)
    print(mX.shape, mY.shape, avX.shape, avY.shape)
