import os, pyBigWig

def get_bin_value(bw, chr, start, end, resolution):
    return bw.stats(chr, start, end, type="mean", nBins=(end - start) // resolution)

class BIOSAMPLE(object):
    def __init__(self, path, biosample_name, chr_sizes_file="data/hg38.chrom.sizes", resolution=25):
        """
        given the biosample name, look for all available tracks and have a list of available track names
        """
        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]

        self.resolution = resolution
        self.chr_sizes = {}
        with open(chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)
        
        self.bins = {chr: [0] * (size // self.resolution + 1) for chr, size in self.chr_sizes.items()}

        self.tracks = {}
        for f in os.listdir(path):
            if biosample_name in f:
                trackname = f.replace(biosample_name, "").replace(".bigwig", "")
                self.tracks[trackname] = pyBigWig.open(path + "/" + f)

class ENCODE_IMPUTATION_DATASET(object):
    def __init__(self, path, segment_adjacency=False):
        """
        divide the genome into ``context-length'' sizes (just indices) -> let's call # of all possible segments N

        let's define M as the number of samples to randomly draw from N (M <= N)

        if segment_adjacency:
            randomly select M/2 adjacent indices from all N possible indices
        else:
            randomly select M indices from all N possible indices

        for each biosample, for all tracks get the signal values of M positions
        construct the dataset
        """
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass



class augment(object):
    def __init__(self):
        pass 

    def random_gaussian(self):
        pass
