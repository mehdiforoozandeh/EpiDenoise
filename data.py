import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import requests, os, itertools, ast, io, pysam, pickle, datetime, pyBigWig
from tqdm import tqdm

def single_download(dl_dict):
    url, save_dir_name, exp, bios = dl_dict["url"], dl_dict["save_dir_name"], dl_dict["exp"], dl_dict["bios"]
    if os.path.exists(save_dir_name) ==  False:
        try:
            print(f"downloading assay: {exp} | biosample: {bios}")
            download_response = requests.get(url, allow_redirects=True)
            open(save_dir_name, 'wb').write(download_response.content)
        except Exception as e:
            with open("data/" + bios  + f"/failed_{exp}", "w") as f:
                f.write(f"failed to download {bios}_{exp}\n {e}")
            print(e)

        if "bam" in save_dir_name:
            os.system(f"samtools index {save_dir_name}")

            print(f"processing BAM to Signal | assay: {exp} | biosample: {bios}")
            preprocessor = BAM_TO_SIGNAL()
            preprocessor.get_coverage(
                bam_file=save_dir_name, 
                chr_sizes_file="data/hg38.chrom.sizes")
            
            os.system(f"rm {save_dir_name}")

    else:
        print(f"assay: {exp} | biosample: {bios} already exists!")
    
class GET_DATA(object):
    def __init__(self):
        self.encode_imputation_challenge_assays = ["DNase-seq", "H3K4me3", "H3K36me3", "H3K27ac", "H3K9me3",
                "H3K27me3", "H3K4me1", "H3K9ac", "H3K4me2", "H2AFZ", "H3K79me2", "ATAC-seq",
                "H3K18ac", "H4K20me1", "H3K4ac", "H4K8ac", "H3K79me1", "H3K14ac", "H2BK120ac", 
                "H2BK12ac", "H2BK5ac",  "H4K91ac", "H2BK15ac", "H3K23ac",  "H4K5ac",
                "H3K5bac", "H3K23me2", "H2BK20ac", "H3K9me1", "H3F3A", "H4K12ac",  "H3T11ph", "HAk9ac", "H3K9me2"]

        self.select_assays = ["DNase-seq", "H3K4me3", "H3K36me3", "H3K27ac", "H3K9me3", "H3K27me3", "H3K4me1", "ATAC-seq", "CTCF"]

        self.expression_data = ["RNA-seq", "CAGE"]
        
        self.target_file_format = {
            "ChIP-seq": "bam", 
            "CAGE": "bam",
            "ChIA-PET": "bam",
            "ATAC-seq": "bam", 
            "RNA-seq": "bam",
            "DNase-seq": "bam"}

        self.headers = {'accept': 'application/json'}
        # report_url= https://www.encodeproject.org/report/?type=Experiment&control_type!=*&perturbed=false&assay_title=TF+ChIP-seq&assay_title=Histone+ChIP-seq&assay_title=DNase-seq&assay_title=ATAC-seq&assay_title=ChIA-PET&assay_title=CAGE&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=total+RNA-seq&status=released
        self.exp_tsv_url = """https://www.encodeproject.org/report.tsv?type=Experiment&control_type!=*&perturbed=false&assay_title=TF+ChIP-seq&assay_title=Histone+ChIP-seq&assay_title=DNase-seq&assay_title=ATAC-seq&assay_title=ChIA-PET&assay_title=CAGE&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assay_title=total+RNA-seq&status=released"""
        self.biosample_tsv_url = """https://www.encodeproject.org/report.tsv?type=Biosample&organism.scientific_name=Homo+sapiens"""

        self.experiment_url = """https://www.encodeproject.org/experiments/"""
        self.biosample_url = """https://www.encodeproject.org/biosamples/"""

    def search_ENCODE(self):
        """
        DF1:
            rows: biosamples
            columns: experiments (from self.experiments)
        
        DF2:
            rows: biosamples
            columns: metadata fields (including replicate biosample if any)
        """

        exp_search_report = pd.read_csv(self.exp_tsv_url, sep='\t', skiprows=1)
        """
        exp search report columns:
        ['ID', 'Accession', 'Assay name', 'Assay title', 'Biosample classification', 'Target', 'Target of assay',
       'Target gene symbol', 'Biosample summary', 'Biosample term name', 'Dbxrefs', 'Description', 'Lab', 'Project', 'Status', 'Files',
       'Related series', 'Biosample accession', 'Biological replicate', 'Technical replicate', 'Linked antibody', 'Organism', 'Life stage',
       'Biosample age', 'Biosample treatment', 'Biosample treatment ontology ID', 'Biosample treatment amount',
       'Biosample treatment amount units', 'Biosample treatment duration', 'Biosample treatment duration units', 'Synchronization',
       'Post-synchronization time', 'Post-synchronization time units', 'Biosample modification site target organism',
       'Biosample modification site introduced gene organism', 'Replicates', 'Mixed biosamples', 'Cellular component',
       'Library construction platform', 'Library construction method']
        """

        bios_search_report = pd.read_csv(self.biosample_tsv_url, sep='\t', skiprows=1)
        """
        biosample search report columns:
        ['ID', 'Accession', 'Description', 'Age', 'Age units', 'Biosample age', 'Biosample term name', 'Biosample classification', 'Synchronization',
       'Post-synchronization time', 'Post-synchronization time units', 'Organism', 'Child biosamples', 'Modification site',
       'Modification site target organism', 'Modification site introduced gene organism', 'Modification purpose',
       'Modification method', 'Modification category', 'Source', 'Lab', 'Life stage', 'Status', 'Biosample treatment',
       'Biosample treatment type details', 'Biosample treatment purpose', 'Culture harvest date', 'Date obtained', 'Summary']
        """
        bios_search_report.set_index('Accession', inplace=True)
        bios_search_report = bios_search_report.to_dict('index')

        self.DF1 = {}
        self.DF2 = {}

        # for each experiment, look up the biosample and connect the experiment and biosample data
        for i in range(len(exp_search_report)):
            if i%1000 == 0:
                print(f"{i}/{len(exp_search_report)}")

            exp = exp_search_report["Accession"][i]

            if exp_search_report["Assay name"][i] == "DNase-seq":
                assay = "DNase-seq"
            elif exp_search_report["Assay name"][i] == "RNA-seq":
                assay = "RNA-seq"
            elif exp_search_report["Assay name"][i] == "ATAC-seq":
                assay = "ATAC-seq"
            elif exp_search_report["Assay name"][i] == "CAGE":
                assay = "CAGE"
            elif exp_search_report["Assay name"][i] == "ChIA-PET":
                assay = "ChIA-PET"
            elif exp_search_report["Assay name"][i] == "ChIP-seq":
                assay = exp_search_report["Target of assay"][i]

            # limit the search to select_assays only
            if assay in list(set( self.select_assays + self.expression_data )):
                biosample_accessions = exp_search_report["Biosample accession"][i].split(",")

                for biosample_accession in biosample_accessions:
                    if biosample_accession in self.DF1.keys():
                        self.DF1[biosample_accession][assay] = exp

                    else:
                        self.DF1[biosample_accession] = {}
                        self.DF1[biosample_accession][assay] = exp

                    if biosample_accession not in self.DF2.keys():
                        try:
                            self.DF2[biosample_accession] = bios_search_report[biosample_accession]

                            if len(biosample_accessions) == 1:
                                self.DF2[biosample_accession]["isogenic_replicates"] = None
                            else:
                                self.DF2[biosample_accession]["isogenic_replicates"] = ",".join([x for x in biosample_accessions if x != biosample_accession])

                        except:
                            pass
        
        self.DF1 = pd.DataFrame.from_dict(self.DF1, orient='index').sort_index(axis=1)
        self.DF2 = pd.DataFrame.from_dict(self.DF2, orient='index').sort_index(axis=1)
        print(self.DF1)
        print(self.DF2)

    def save_metadata(self, metadata_file_path="data/"):
        """
        save DF1 and DF2 from search_ENCODE
        """
        self.DF1.to_csv(metadata_file_path + "DF1.csv")
        self.DF2.to_csv(metadata_file_path + "DF2.csv")

    def download_from_metadata(self, metadata_file_path="data/", parallel=True, n_p=15):
        """
        read DF1 and DF2 metadata files and run download_search_results on them
        """
        self.DF1 = pd.read_csv(metadata_file_path + "DF1.csv")
        self.DF2 = pd.read_csv(metadata_file_path + "DF2.csv")

        self.DF1.rename(columns={'Unnamed: 0': 'Accession'}, inplace=True)
        self.DF2.rename(columns={'Unnamed: 0': 'Accession'}, inplace=True)
        
        self.DF1['num_available'] = self.DF1.count(axis=1) - 1
        self.DF1['num_nonexp_available'] = self.DF1.drop(columns=['Accession', "CTCF", 'RNA-seq', 'CAGE', "num_available"]).count(axis=1)

        self.DF1 = self.DF1.sort_values(by='num_nonexp_available', ascending=False)
        self.DF1 = self.DF1.reset_index(drop=True)

        # filter DF1 based on availability of data (remove biosamples that dont have anything but expression data)
        self.DF1 = self.DF1[self.DF1['num_nonexp_available'] != 0]

        self.DF1 = self.DF1[self.DF1['num_nonexp_available'] >= 5]
        # print(self.DF1)

        # self.DF1 = self.DF1.iloc[:2,:]
        # print(self.DF1)

        to_download = []
        for i in range(len(self.DF1)):
            bios = self.DF1["Accession"][i]
            if os.path.exists(metadata_file_path + "/" + bios) == False:
                os.mkdir(metadata_file_path + "/" + bios)

            for exp in self.DF1.columns:
                if exp not in ["Accession", "num_nonexp_available", "num_available"]:
                    try:
                        if os.path.exists(metadata_file_path + "/" + bios + "/" + exp) == False:
                            os.mkdir(metadata_file_path + "/" + bios + "/" + exp)
                        
                        if pd.notnull(self.DF1[exp][i]):
                            experiment_accession = self.DF1[exp][i]
                        else:
                            continue
                        
                        exp_url = self.experiment_url + experiment_accession
                        
                        exp_respond = requests.get(exp_url, headers=self.headers)
                        exp_results = exp_respond.json()
                        
                        e_fileslist = list(exp_results['original_files'])
                        e_files_navigation = []

                        for ef in e_fileslist:
                            efile_respond = requests.get("https://www.encodeproject.org{}".format(ef), headers=self.headers)
                            efile_results = efile_respond.json()

                            if efile_results['file_format'] == "bam" or "tsv":
                                try: #ignore files without sufficient info or metadata

                                    if ',' not in str(efile_results['origin_batches']):
                                        if efile_results['status'] == "released": 
                                            #ignore old and depricated versions

                                            e_file_biosample = str(efile_results['origin_batches'])
                                            e_file_biosample = e_file_biosample.replace('/', '')
                                            e_file_biosample = e_file_biosample.replace('biosamples','')[2:-2]
                                            
                                            # ignore files that contain both replicates 
                                            if e_file_biosample == bios:
                                                e_files_navigation.append(
                                                    [exp, efile_results['accession'], e_file_biosample,
                                                    efile_results['file_format'], efile_results['output_type'], 
                                                    efile_results['dataset'], efile_results['biological_replicates'], 
                                                    efile_results['file_size'], efile_results['assembly'], 
                                                    "https://www.encodeproject.org{}".format(efile_results['href']), 
                                                    efile_results['date_created'], efile_results['status']])
                                except:
                                    pass

                        e_files_navigation = pd.DataFrame(e_files_navigation, columns=[
                            'assay', 'accession', 'biosample', 'file_format', 
                            'output_type', 'experiment', 'bio_replicate_number', 
                            'file_size', 'assembly', 'download_url', 'date_created', 'status'])

                        # select one file from e_files_navigation to download
                        e_files_navigation.to_csv(metadata_file_path + "/" + bios + "/" + exp + "/all_files.csv")
                        
                        # Convert 'date_created' to datetime
                        e_files_navigation['date_created'] = pd.to_datetime(e_files_navigation['date_created'])
                        
                        if exp == "RNA-seq":
                            # Filter rows where 'output_type' is 'gene quantification'
                            filtered_df = e_files_navigation[e_files_navigation['output_type'] == 'gene quantifications']
                        else:
                            # Filter rows where 'output_type' is 'alignments'
                            filtered_df = e_files_navigation[e_files_navigation['output_type'] == 'alignments']

                        # Find the row with the newest 'date_created'
                        newest_row = filtered_df[filtered_df['date_created'] == filtered_df['date_created'].max()]

                        # Print the newest row
                        with open(metadata_file_path + "/" + bios + "/" + exp + "/file_metadata.txt", "w") as f:
                            for c in newest_row.columns:
                                f.write(f"{c}\t{newest_row[c].values[0]}\n")
                        
                        if exp == "RNA-seq":
                            save_dir_name = metadata_file_path + "/" + bios + "/" + exp + "/" + newest_row["accession"].values[0] + ".tsv"
                        else:
                            save_dir_name = metadata_file_path + "/" + bios + "/" + exp + "/" + newest_row["accession"].values[0] + ".bam"
                        
                        url = newest_row["download_url"].values[0]

                        to_download.append({"url":url, "save_dir_name":save_dir_name, "exp":exp, "bios":bios})

                    except Exception as e:
                        with open(metadata_file_path + "/" + bios  + f"/failed_{exp}", "w") as f:
                            f.write(f"failed to download {bios}_{exp}\n {e}")
                        print(e)
        
        if parallel:
            with mp.Pool(n_p) as pool:
                pool.map(single_download, to_download)
        else:
            for d in to_download:
                single_download(d)

class BAM_TO_SIGNAL(object):
    def __init__(self):
        """
        Initialize the object
        """
        pass

    def read_chr_sizes(self):
        """
        Read a file with chromosome sizes and return a dictionary where keys are 
        chromosome names and values are chromosome sizes.
        
        Parameters:
        file_path (str): The path to the file with chromosome sizes.

        Returns:
        dict: A dictionary where keys are chromosome names and values are chromosome sizes.
        """

        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        
        self.chr_sizes = {}
        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

    def load_bam(self):
        """
        Load the BAM file using pysam.
        """
        self.bam = pysam.AlignmentFile(self.bam_file, 'rb')

    def initialize_empty_bins(self):
        """
        Initialize empty bins for each chromosome based on the resolution.
        """
        self.bins = {chr: [0] * (size // self.resolution + 1) for chr, size in self.chr_sizes.items()}

    def calculate_coverage(self):
        """
        Calculate the coverage for each bin.
        """
        self.bins = {chr: [0] * (self.chr_sizes[chr] // self.resolution + 1) for chr in self.chr_sizes}
        self.coverage = {
            'chr': [],
            'start': [],
            'end': [],
            'read_count': []
        }

        for chr in self.chr_sizes:
            # print(f"getting {chr} coverage...")
            for read in self.bam.fetch(chr):
                start_bin = read.reference_start // self.resolution
                end_bin = read.reference_end // self.resolution
                for i in range(start_bin, end_bin+1):
                    self.bins[chr][i] += 1

            for i, count in enumerate(self.bins[chr]):
                start = i * self.resolution
                end = start + self.resolution
                self.coverage["chr"].append(str(chr))
                self.coverage["start"].append(int(start))
                self.coverage["end"].append(int(end))
                self.coverage["read_count"].append(float(count))

    def save_coverage_pkl(self):
        """
        Save the coverage data to a pickle file.

        Parameters:
        file_path (str): The path to the pickle file.
        """
        file_path = self.bam_file.replace(".bam", f"_cvrg{self.resolution}bp.pkl")

        with open(file_path, 'wb') as f:
            pickle.dump(self.coverage, f)
        
        os.system(f"gzip {file_path}")
    
    def save_coverage_bigwig(self):
        """
        Save the coverage data to a BigWig file.

        Parameters:
        file_path (str): The path to the BigWig file.
        """
        file_path = self.bam_file.replace(".bam", f"_cvrg{self.resolution}bp.bw")
        bw = pyBigWig.open(file_path, 'w')
        bw.addHeader([(k, v) for k, v in self.chr_sizes.items()])

        bw.addEntries(
            self.coverage["chr"], 
            self.coverage["start"], 
            ends=self.coverage["end"], 
            values=self.coverage["read_count"])

        bw.close()
    
    def load_coverage_bigwig(self, file_path, chr_sizes_file):
        """
        Load the coverage data from a BigWig file.

        Parameters:
        file_path (str): The path to the BigWig file.
        """

        self.chr_sizes_file = chr_sizes_file
        self.read_chr_sizes()

        self.bw = pyBigWig.open(file_path)
        self.coverage = {
            'chr': [],
            'start': [],
            'end': [],
            'read_count': []
        }

        for chr in self.chr_sizes:
            intervals = self.bw.intervals(chr)
            for interval in intervals:
                start, end, count = interval
                self.coverage["chr"].append(str(chr))
                self.coverage["start"].append(int(start))
                self.coverage["end"].append(int(end))
                self.coverage["read_count"].append(float(count))
        
        self.bw.close()
        return self.coverage

    def get_coverage(self, bam_file, chr_sizes_file, resolution=25):
        t0 = datetime.datetime.now()
        self.bam_file = bam_file
        self.chr_sizes_file = chr_sizes_file
        self.resolution = resolution

        self.read_chr_sizes()
        self.load_bam()
        self.initialize_empty_bins()
        self.calculate_coverage()
        self.save_coverage_pkl()
        # self.save_coverage_bigwig()

        t1 = datetime.datetime.now()
        print(f"took {t1-t0} to get coverage for {bam_file} at resolution: {resolution}bp")

    def load_coverage_pkl(self, file_path):
        """
        Load the coverage data from a pickle file.

        Parameters:
        file_path (str): The path to the pickle file.
        """
        with gzip.open(file_path, 'rb') as f:
            self.coverage = pickle.load(f)
        
        return self.coverage

if __name__ == "__main__":

    d = GET_DATA()
    # d.search_ENCODE()
    # d.save_metadata()
    d.download_from_metadata()

    # df1 =pd.read_csv("data/DF1.csv")
    # df2 =pd.read_csv("data/DF2.csv")

    # print(df1)
    # print(df2)

    # print(len(df2["Biosample term name"].unique()))

    # preprocessor = BAM_TO_SIGNAL()
    # preprocessor.get_coverage(
    #     bam_file="data/ENCBS343AKO/H3K4me3/ENCFF984WUD.bam", 
    #     chr_sizes_file="data/hg38.chrom.sizes", 
    #     resolution=1000)

    # preprocessor = BAM_TO_SIGNAL()
    # new = preprocessor.load_coverage_bigwig("data/ENCBS343AKO/H3K4me3/ENCFF984WUD_cvrg1000bp.bw", chr_sizes_file="data/hg38.chrom.sizes")

    # new = preprocessor.load_coverage_pkl("data/ENCBS343AKO/H3K4me3/ENCFF984WUD_cvrg25bp.pkl.gz")

    # preprocessor.get_coverage(bam_file="data/ENCBS343AKO/H3K4me3/ENCFF984WUD.bam", chr_sizes_file="data/hg38.chrom.sizes", resolution=1000)
    # new = preprocessor.load_coverage_pkl("data/ENCBS343AKO/H3K4me3/ENCFF984WUD_cvrg1000bp.pkl.gz")