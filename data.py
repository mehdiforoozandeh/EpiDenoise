import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import multiprocessing as mp
import requests, os, itertools, ast, io, pysam, datetime, pyBigWig, time, gzip, pickle, json, subprocess, random, glob
from torch.utils.data import Dataset
import torch, sys, math
import pybedtools
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing
from multiprocessing import Pool

def extract_donor_information(json_data):
    # Check if 'donor' key exists in the JSON data
    # Initialize an empty dictionary to store donor information
    donor_info = json_data.get('donor', {})
    extracted_info = {}
    
    # Extract relevant donor information
    extracted_info['Status'] = donor_info.get('status')
    extracted_info['Accession'] = donor_info.get('accession')
    extracted_info['Aliases'] = donor_info.get('aliases')
    extracted_info['Species'] = donor_info.get('organism', {}).get('scientific_name')
    extracted_info['Life Stage'] = donor_info.get('life_stage')
    extracted_info['Age'] = donor_info.get('age')
    extracted_info['Sex'] = donor_info.get('sex')
    extracted_info['Ethnicity'] = donor_info.get('ethnicity')
    
    return extracted_info

def visualize_encode_data(df):
    # Remove all rows for which num_nonexp_available < 3
    df_filtered = df[df['num_nonexp_available'] >= 3]

    # Sort biosamples based on num_nonexp_available
    df_sorted = df_filtered.sort_values('num_nonexp_available', ascending=False)

    # Prepare the DataFrame for the heatmap (experiments as rows, biosamples as columns)
    heatmap_df = df_sorted.set_index('Accession').drop(['num_available', 'num_nonexp_available'], axis=1).T

    # Convert experiments to numerical values: 1 for available data and NaN for missing
    heatmap_numeric = heatmap_df.notna().astype(int)

    # Calculate the sum of non-NaN entries for each row (experiment) and sort the DataFrame
    heatmap_numeric['non_nan_count'] = heatmap_numeric.sum(axis=1)
    heatmap_numeric_sorted = heatmap_numeric.sort_values('non_nan_count', ascending=False).drop('non_nan_count', axis=1)

    # Create a custom colormap
    cmap = ListedColormap(['white', 'blue'])

    # Plot the heatmap with a larger figure size
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(20, 15))  # Increase figure size

    # Create the heatmap
    sns.heatmap(heatmap_numeric_sorted, cmap=cmap, cbar=False, linewidths=0.0)

    # Remove x-axis labels
    ax.set_xticklabels([])

    # Decrease font-size for y-axis labels
    plt.setp(ax.get_yticklabels(), fontsize=9)
    plt.savefig(f"data/dataset.png", dpi=200)

def visualize_availability(
    sorted_data_hist_uniq_exp, sorted_data_tf_uniq_exp, 
    sorted_data_rest_uniq_exp, encode_imputation_challenge_assays): # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Histone Modifications
        axs[0].bar([item[0] for item in sorted_data_hist_uniq_exp], [item[1] for item in sorted_data_hist_uniq_exp], color='green', alpha=0.8)
        axs[0].set_title('Histone Modifications')
        axs[0].tick_params(axis='x', rotation=90, labelsize=7)
        assays = [item[0] for item in sorted_data_hist_uniq_exp]
        values = [item[1] for item in sorted_data_hist_uniq_exp]
        # for assay, value in sorted_data_hist_uniq_exp:
        #     if assay in encode_imputation_challenge_assays:
        #         axs[0].text(assays.index(assay), value, '*', color='red', fontsize=14, ha='center')

        # TF Binding
        axs[1].bar([item[0] for item in sorted_data_tf_uniq_exp if item[1]>15], [item[1] for item in sorted_data_tf_uniq_exp if item[1]>15], color='orange', alpha=0.8)
        axs[1].set_title('TF Binding')
        axs[1].tick_params(axis='x', rotation=90, labelsize=7)
        assays = [item[0] for item in sorted_data_tf_uniq_exp]
        values = [item[1] for item in sorted_data_tf_uniq_exp]
        # for assay, value in sorted_data_tf_uniq_exp:
        #     if assay in encode_imputation_challenge_assays:
        #         axs[1].text(assays.index(assay), value, '*', color='red', fontsize=14, ha='center')

        # Other Assays
        axs[2].bar([item[0] for item in sorted_data_rest_uniq_exp], [item[1] for item in sorted_data_rest_uniq_exp], color='blue', alpha=0.8)
        axs[2].set_title('Other Assays')
        axs[2].tick_params(axis='x', rotation=90, labelsize=10)
        assays = [item[0] for item in sorted_data_rest_uniq_exp]
        values = [item[1] for item in sorted_data_rest_uniq_exp]
        # for assay, value in sorted_data_rest_uniq_exp:
        #     if assay in encode_imputation_challenge_assays:
        #         axs[2].text(assays.index(assay), value, '*', color='red', fontsize=14, ha='center')

        plt.tight_layout()
        plt.savefig(f"data/availability.png", dpi=200)

def single_download(dl_dict):
    num_attempts = 10

    def download_save(url, save_dir_name):
        try:
            # Stream the download; this loads the file piece by piece
            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Check for request errors
                with open(save_dir_name, 'wb') as file:
                    # Iterate over the response in chunks (e.g., 8KB each)
                    for chunk in response.iter_content(chunk_size=int(1e3*1024)):
                        # Write each chunk to the file immediately
                        file.write(chunk)
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    url, save_dir_name, exp, bios = dl_dict["url"], dl_dict["save_dir_name"], dl_dict["exp"], dl_dict["bios"]

    if os.path.exists(save_dir_name) ==  False:
        print(f"downloading assay: {exp} | biosample: {bios}")
        attempt = 0
        is_done = False
        while is_done == False and attempt < num_attempts:
            if attempt > 0:
                time.sleep(10)

            print(f"attemp number {attempt}")
            is_done = download_save(url, save_dir_name)
            attempt += 1
                
        if "bam" in save_dir_name:
            os.system(f"samtools index {save_dir_name}")
            print(f"processing BAM to Signal | assay: {exp} | biosample: {bios}")

            bam_to_signal = BAM_TO_SIGNAL(
                bam_file=save_dir_name, 
                chr_sizes_file="data/hg38.chrom.sizes")

            bam_to_signal.full_preprocess()
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

    def search_ENCODE(self, metadata_file_path="data/"):
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

        hist_uniq_exp = {}
        tf_uniq_exp = {}
        rest_uniq_exp = {}

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

            
            ########################################################################

            if exp_search_report["Assay title"][i] == "Histone ChIP-seq":
                if assay not in hist_uniq_exp.keys():
                    hist_uniq_exp[assay] = 1
                else:
                    hist_uniq_exp[assay] += 1

            elif exp_search_report["Assay title"][i] == "TF ChIP-seq":
                if assay not in tf_uniq_exp.keys():
                    tf_uniq_exp[assay] = 1
                else:
                    tf_uniq_exp[assay] += 1
            else:
                if assay not in rest_uniq_exp.keys():
                    rest_uniq_exp[assay] = 1
                else:
                    rest_uniq_exp[assay] += 1

        ########################################################################

        sorted_data_hist_uniq_exp = sorted(hist_uniq_exp.items(), key=lambda x: x[1], reverse=True)
        sorted_data_tf_uniq_exp = sorted(tf_uniq_exp.items(), key=lambda x: x[1], reverse=True)
        sorted_data_rest_uniq_exp = sorted(rest_uniq_exp.items(), key=lambda x: x[1], reverse=True)

        ########################################################################

        visualize_availability(
            sorted_data_hist_uniq_exp, sorted_data_tf_uniq_exp, 
            sorted_data_rest_uniq_exp, self.encode_imputation_challenge_assays)
        
        ########################################################################
        
        self.sorted_data_hist_uniq_exp = dict(sorted_data_hist_uniq_exp)
        self.sorted_data_rest_uniq_exp = dict(sorted_data_rest_uniq_exp)
        self.sorted_data_tf_uniq_exp = dict(sorted_data_tf_uniq_exp)

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

            ########################################################################
            """
            what to assays to include in search:
                - all histone mods
                - TF avail > 15
                - DNase, ATAC, CAGE, RNA-seq, ChIA-PET
            """

            statement1 = bool(exp_search_report["Assay title"][i] == "Histone ChIP-seq")
            statement2 = bool(exp_search_report["Assay title"][i] == "TF ChIP-seq") and bool(self.sorted_data_tf_uniq_exp[assay] > 15)
            statement3 = bool(assay in ["DNase-seq", "RNA-seq", "ATAC-seq", "CAGE", "ChIA-PET"])

            if statement1 or statement2 or statement3:
                biosample_accessions = exp_search_report["Biosample accession"][i].split(",")

                for biosample_accession in biosample_accessions:
                    if biosample_accession not in bios_search_report.keys():
                        continue
                    
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
                                self.DF2[biosample_accession]["isogenic_replicates"] = ",".join(
                                    [x for x in biosample_accessions if x != biosample_accession])

                        except:
                            pass

        self.DF1 = pd.DataFrame.from_dict(self.DF1, orient='index').sort_index(axis=1)
        self.DF2 = pd.DataFrame.from_dict(self.DF2, orient='index').sort_index(axis=1)
        """
        save DF1 and DF2 from search_ENCODE
        """
        self.DF1.to_csv(metadata_file_path + "DF1.csv")
        self.DF2.to_csv(metadata_file_path + "DF2.csv")

    def filter_biosamples(self, metadata_file_path="data/"):

        """
        read DF1 and DF2 metadata files and run download_search_results on them
        """
        self.DF1 = pd.read_csv(metadata_file_path + "DF1.csv")
        self.DF2 = pd.read_csv(metadata_file_path + "DF2.csv")

        self.DF1.rename(columns={'Unnamed: 0': 'Accession'}, inplace=True)
        self.DF2.rename(columns={'Unnamed: 0': 'Accession'}, inplace=True)
        
        self.DF1['num_available'] = self.DF1.count(axis=1) - 1
        self.DF1['num_nonexp_available'] = self.DF1.drop(
            columns=['Accession', 'RNA-seq', 'CAGE', "num_available"]).count(axis=1)
            
        self.DF1 = self.DF1.sort_values(by='Accession').reset_index(drop=True)
        self.DF2 = self.DF2.sort_values(by='Accession').reset_index(drop=True)

        ########################################################################

        """
        what biosamples to download
            - any histone mod available
            - DNase or ATAC available
            - > 3 TF available
        """

        statement1 = self.DF1[list(self.sorted_data_hist_uniq_exp.keys())].count(axis=1) > 0
        statement2 = self.DF1[["DNase-seq", "ATAC-seq"]].count(axis=1) > 0
        statement3 = self.DF1[[
            tf for tf in self.sorted_data_tf_uniq_exp.keys() if self.sorted_data_tf_uniq_exp[tf] > 15]].count(axis=1) > 3

        combined_statement = (statement1 | statement2 | statement3)

        self.DF1 = self.DF1[combined_statement].reset_index(drop=True)
        self.DF2 = self.DF2[combined_statement].reset_index(drop=True)
    
        ########################################################################

        visualize_encode_data(self.DF1)

        ########################################################################
                
        self.DF1 = self.DF1.drop(["num_available", "num_nonexp_available"], axis=1)
    
        """
        save DF1 and DF2 from search_ENCODE
        """
        self.DF1.to_csv(metadata_file_path + "DF1.csv")
        self.DF2.to_csv(metadata_file_path + "DF2.csv")

    def load_metadata(self, metadata_file_path="data/"):
        self.DF1 = pd.read_csv(metadata_file_path + "DF1.csv").drop(["Unnamed: 0"], axis=1)
        self.DF2 = pd.read_csv(metadata_file_path + "DF2.csv").drop(["Unnamed: 0"], axis=1)

    def get_experiment(self, dl_dict, process_bam=True):
        num_attempts = 10

        def download_save(url, save_dir_name):
            try:
                download_response = requests.get(url, allow_redirects=True)
                open(save_dir_name, 'wb').write(download_response.content)
                return True

            except:
                return False

        url, save_dir_name, exp, bios = dl_dict["url"], dl_dict["save_dir_name"], dl_dict["exp"], dl_dict["bios"]

        if os.path.exists(save_dir_name) ==  False and os.path.exists(
            f"{'/'.join(save_dir_name.split('/')[:-1])}/signal_DSF1_res25/") == False:

            print(f"downloading assay: {exp} | biosample: {bios}")
            attempt = 0
            is_done = False
            while is_done == False and attempt < num_attempts:
                if attempt > 0:
                    time.sleep(20)

                print(f"    attemp number {attempt}")
                is_done = download_save(url, save_dir_name)
                attempt += 1
            
            if is_done == False:
                open(save_dir_name.replace(".bam",".failed"), 'w').write("failed to download")
                print("failed to download", save_dir_name)
                return

            else:  
                if "bam" in save_dir_name:
                    try:
                        os.system(f"samtools index {save_dir_name}")

                        if process_bam:
                            print(f"processing BAM to Signal | assay: {exp} | biosample: {bios}")

                            bam_to_signal = BAM_TO_SIGNAL(
                                bam_file=save_dir_name, 
                                chr_sizes_file="data/hg38.chrom.sizes")

                            bam_to_signal.full_preprocess()
                            
                            os.system(f"rm {save_dir_name}")

                    except:
                        print("failed to process", save_dir_name)

        else:
            print(f"assay: {exp} | biosample: {bios} already exists!")

    def get_biosample(self, bios, df1_ind, metadata_file_path, assembly):
        i = df1_ind
        to_download_bios = []

        bios_data =  requests.get(f"""https://www.encodeproject.org/biosamples/{bios}""", headers=self.headers)
        bios_data = bios_data.json()
        donor_info = extract_donor_information(bios_data)

        if os.path.exists(metadata_file_path + "/" + bios + "/") == False:
            os.mkdir(metadata_file_path + "/" + bios + "/")
        
        with open(metadata_file_path + "/" + bios + '/donor.json', 'w') as file:
            json.dump(donor_info, file, indent=4)

        for exp in self.DF1.columns:
            if exp not in ["Accession", "num_nonexp_available", "num_available"]:
                try:
                    if pd.notnull(self.DF1[exp][i]):
                        # print(bios, exp, self.DF1[exp][i])
                        experiment_accession = self.DF1[exp][i]
                        if os.path.exists(metadata_file_path + "/" + bios + "/" + exp) == False:
                            os.mkdir(metadata_file_path + "/" + bios + "/" + exp)

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

                        if efile_results['file_format'] == "bam" or efile_results['file_format'] == "tsv":
                            # try: #ignore files without sufficient info or metadata

                            if efile_results['status'] == "released": 
                                #ignore old and depricated versions

                                if "origin_batches" in efile_results.keys():
                                    if ',' not in str(efile_results['origin_batches']):
                                        e_file_biosample = str(efile_results['origin_batches'])
                                        e_file_biosample = e_file_biosample.replace('/', '')
                                        e_file_biosample = e_file_biosample.replace('biosamples','')[2:-2]
                                    else:
                                        repnumber = int(efile_results['biological_replicates'][0]) - 1
                                        e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]
                                else:
                                    repnumber = int(efile_results['biological_replicates'][0]) - 1
                                    e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]

                                # ignore files that contain both replicates 
                                if e_file_biosample == bios:
                                    parsed = [exp, efile_results['accession'], e_file_biosample,
                                        efile_results['file_format'], efile_results['output_type'], 
                                        efile_results['dataset'], efile_results['biological_replicates'], 
                                        efile_results['file_size'], efile_results['assembly'], 
                                        "https://www.encodeproject.org{}".format(efile_results['href']), 
                                        efile_results['date_created'], efile_results['status']]

                                    if "read_length" in efile_results:
                                        read_length = efile_results["read_length"]
                                        run_type = efile_results["run_type"]
                                        parsed.append(read_length)
                                        parsed.append(run_type)

                                    elif "mapped_read_length" in efile_results:
                                        read_length = efile_results["mapped_read_length"]
                                        run_type = efile_results["mapped_run_type"]
                                        parsed.append(read_length)
                                        parsed.append(run_type)

                                    else:
                                        parsed.append(None)
                                        parsed.append(None)

                                    e_files_navigation.append(parsed)
                        # except:
                        #     pass

                    if len(e_files_navigation) == 0:
                        for ef in e_fileslist:
                            efile_respond = requests.get("https://www.encodeproject.org{}".format(ef), headers=self.headers)
                            efile_results = efile_respond.json()
                            if efile_results['file_format'] == "bam" or efile_results['file_format'] == "tsv":
                                if efile_results['status'] == "released": 
                                    if "origin_batches" in efile_results.keys():
                                        if ',' not in str(efile_results['origin_batches']):
                                            e_file_biosample = str(efile_results['origin_batches'])
                                            e_file_biosample = e_file_biosample.replace('/', '')
                                            e_file_biosample = e_file_biosample.replace('biosamples','')[2:-2]
                                        else:
                                            repnumber = int(efile_results['biological_replicates'][0]) - 1
                                            e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]
                                    else:
                                        repnumber = int(efile_results['biological_replicates'][0]) - 1
                                        e_file_biosample = exp_results["replicates"][repnumber]["library"]["biosample"]["accession"]
                                    
                                    parsed = [exp, efile_results['accession'], e_file_biosample,
                                        efile_results['file_format'], efile_results['output_type'], 
                                        efile_results['dataset'], efile_results['biological_replicates'], 
                                        efile_results['file_size'], efile_results['assembly'], 
                                        "https://www.encodeproject.org{}".format(efile_results['href']), 
                                        efile_results['date_created'], efile_results['status']]

                                    if "read_length" in efile_results:
                                        read_length = efile_results["read_length"]
                                        run_type = efile_results["run_type"]
                                        parsed.append(read_length)
                                        parsed.append(run_type)

                                    elif "mapped_read_length" in efile_results:
                                        read_length = efile_results["mapped_read_length"]
                                        run_type = efile_results["mapped_run_type"]
                                        parsed.append(read_length)
                                        parsed.append(run_type)

                                    else:
                                        parsed.append(None)
                                        parsed.append(None)

                                    e_files_navigation.append(parsed)

                    e_files_navigation = pd.DataFrame(e_files_navigation, columns=[
                        'assay', 'accession', 'biosample', 'file_format', 
                        'output_type', 'experiment', 'bio_replicate_number', 
                        'file_size', 'assembly', 'download_url', 'date_created', 
                        'status', 'read_length', "run_type"])

                    # select one file from e_files_navigation to download
                    e_files_navigation.to_csv(metadata_file_path + "/" + bios + "/" + exp + "/all_files.csv")
                    
                    # just keep target assembly
                    e_files_navigation = e_files_navigation[e_files_navigation['assembly'] == assembly]

                    # Convert 'date_created' to datetime
                    e_files_navigation['date_created'] = pd.to_datetime(e_files_navigation['date_created'])
                    
                    if exp == "RNA-seq":
                        # Filter rows where 'output_type' is 'gene quantification'
                        filtered_df = e_files_navigation[e_files_navigation['output_type'] == 'gene quantifications']
                    else:
                        # Filter rows where 'output_type' is 'alignments'
                        if "alignments" in e_files_navigation['output_type'].unique():
                            filtered_df = e_files_navigation[e_files_navigation['output_type'] == 'alignments']

                        elif "redacted alignments" in e_files_navigation['output_type'].unique():
                            filtered_df = e_files_navigation[e_files_navigation['output_type'] == 'redacted alignments']

                    # Find the row with the newest 'date_created'
                    newest_row = filtered_df[filtered_df['date_created'] == filtered_df['date_created'].max()]

                    # Print the newest row
                    # print(newest_row)
                    # print(newest_row.to_json(indent=4))

                    with open(metadata_file_path + "/" + bios + "/" + exp + "/file_metadata.json", "w") as f:
                        f.write(newest_row.to_json(indent=4))
                    
                    if exp == "RNA-seq":
                        save_dir_name = metadata_file_path + "/" + bios + "/" + exp + "/" + newest_row["accession"].values[0] + ".tsv"
                    else:
                        save_dir_name = metadata_file_path + "/" + bios + "/" + exp + "/" + newest_row["accession"].values[0] + ".bam"
                    
                    url = newest_row["download_url"].values[0]
                    file_size = newest_row["file_size"].values[0]

                    to_download_bios.append({"url":url, "save_dir_name":save_dir_name, "exp":exp, "bios":bios})

                except Exception as e:
                    with open(metadata_file_path + "/" + bios  + f"/failed_{exp}", "w") as f:
                        f.write(f"failed to download {bios}_{exp}\n {e}")

        # NUM_BIOS_DOWNLOADED += 1
        # if NUM_BIOS_DOWNLOADED % 30 == 0:
        #     print(NUM_BIOS_DOWNLOADED)
        return to_download_bios

    def get_biosample_wrapper(self, *args):
        # Wrapper method that can be called in a multiprocessing context
        return self.get_biosample(*args)

    def get_all(self, metadata_file_path="data/", mode="parallel", n_p=25, assembly="GRCh38"):
        to_download = []
        if os.path.exists(metadata_file_path + "DF3.csv"):
            # parse to_download from DF3
            df = pd.read_csv(metadata_file_path + "DF3.csv").drop("Unnamed: 0", axis=1)
            for i in range(len(df)):
                to_download.append(
                    {
                        "url":df["url"][i], "save_dir_name":df["save_dir_name"][i], 
                        "exp":df["exp"][i], "bios":df["bios"][i]
                        }
                )
                if os.path.exists(metadata_file_path + "/" + df["bios"][i]) == False:
                    os.mkdir(metadata_file_path + "/" + df["bios"][i])
                
                if os.path.exists(metadata_file_path + "/" + df["bios"][i] + "/" + df["exp"][i]) == False:
                    os.mkdir(metadata_file_path + "/" + df["bios"][i] + "/" + df["exp"][i])

        else:

            if mode == "parallel":
                def pool_get_biosample(args):
                    return self.get_biosample(*args)

                args_list = [(self.DF1["Accession"][i], i, metadata_file_path, assembly) for i in range(len(self.DF1))]
                with mp.Pool(n_p) as pool:
                    # Map the get_biosample function to the arguments
                    results = pool.starmap(self.get_biosample_wrapper, args_list)

                for bios_dl in results:
                    to_download.extend(bios_dl)

            else:
                for i in range(len(self.DF1)):
                    bios = self.DF1["Accession"][i]
                    bios_dl = self.get_biosample(bios, i, metadata_file_path, assembly)
                    for dl in bios_dl:
                        to_download.append(dl)
                
            df3 = pd.DataFrame(to_download, columns=["url", "save_dir_name", "exp", "bios"])
            df3.to_csv(metadata_file_path + "/DF3.csv")

        if mode == "parallel":
            with mp.Pool(n_p) as pool:
                pool.map(self.get_experiment, to_download)

        else:
            for d in to_download:
                self.get_experiment(d)

class BAM_TO_SIGNAL(object):
    def __init__(self, bam_file, chr_sizes_file, resolution=25):
        self.bam_file = bam_file
        self.chr_sizes_file = chr_sizes_file
        self.resolution = resolution
        self.read_chr_sizes()
        self.load_bam()

    def read_chr_sizes(self):
        main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
        self.chr_sizes = {}
        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

    def load_bam(self):
        self.bam = pysam.AlignmentFile(self.bam_file, 'rb')

    def initialize_empty_bins(self):
        return {chr: [0] * (size // self.resolution + 1) for chr, size in self.chr_sizes.items()}

    def calculate_coverage_pysam(self, downsampling_factor=1.0):
        bins = self.initialize_empty_bins()

        total_mapped_reads = 0 
        bins_with_reads = 0  

        for chr in self.chr_sizes:
            for read in self.bam.fetch(chr):
                if random.random() < 1.0 / downsampling_factor:
                    if read.is_unmapped:
                        continue
                    total_mapped_reads += 1  

                    start_bin = read.reference_start // self.resolution
                    end_bin = read.reference_end // self.resolution
                    for i in range(start_bin, end_bin + 1):
                        if bins[chr][i] == 0:  
                            bins_with_reads += 1  
                        bins[chr][i] += 1
        
        # Calculate coverage as the percentage of bins with at least one read
        total_bins = sum(len(b) for b in bins.values())  
        coverage = (bins_with_reads / total_bins) if total_bins > 0 else 0

        return bins, total_mapped_reads, coverage

    def save_signal_metadata(self, depth, coverage, downsampling_factor):
        if os.path.exists(f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/") == False:
            os.mkdir(f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/")
        
        filename = f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/metadata.json"
        mdict = {
            "coverage":coverage,
            "depth":depth,
            "dsf":downsampling_factor}

        with open(filename, 'w') as file:
            json.dump(mdict, file, indent=4)
    
    def save_signal(self, bins, downsampling_factor=1):
        if os.path.exists(f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/") == False:
            os.mkdir(f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/")

        for chr, data in bins.items():
            np.savez_compressed(
                f"{'/'.join(self.bam_file.split('/')[:-1])}/signal_DSF{downsampling_factor}_res{self.resolution}/{chr}.npz", 
                np.array(data))
            # data_tensor = torch.tensor(data)
            # torch.save(data_tensor, f"{'/'.join(self.bam_file.split('/')[:-1])}/tensors_DSF{downsampling_factor}_res{self.resolution}/{chr}.pt")
    
    def full_preprocess(self, dsf_list=[1,2,4,8]):
        t0 = datetime.datetime.now()

        for dsf in dsf_list:
            data, depth, coverage = self.calculate_coverage_pysam(downsampling_factor=dsf)
            self.save_signal(data, downsampling_factor=dsf)
            self.save_signal_metadata(depth, coverage, downsampling_factor=dsf)

        t1 = datetime.datetime.now()
        print(f"took {t1-t0} to get signals for {self.bam_file} at resolution: {self.resolution}bp")

class ENCODE_IMPUTATION_DATASET(object):
    def __init__(self, path):
        """
        each pkl.gz file is for one biosample and is a dictionary:
        d = {
            "assay1":[list of several pairs of ([chr, start, end], [signal_array]) ],
            "assay2":[list of several pairs of ([chr, start, end], [signal_array]) ],
            "assay3":[list of several pairs of ([chr, start, end], [signal_array]) ],
        }

        let's say we have A assays, and M sample ( len(d["assay1"])=M ).
        signal_arrays are of the same length and for all assays, signal_array[i] corresponds to the same genomic position. 
        if we have M pairs of ([chr, start, end], [signal_array]) for each assay, we will have M samples of size: (len(signal_array), number_of_assays)
        """

        self.path = path
        self.all_assays = ['M{:02d}'.format(i) for i in range(1, 36)]
        self.all_ct = ['C{:02d}'.format(i) for i in range(1, 52)]

        availability = {}
        for f in os.listdir(self.path):
            if ".bigwig" in f: 
                if f[:3] not in availability.keys():
                    availability[f[:3]] = 0
                availability[f[:3]] += 1
                

        self.biosamples = {}
        for f in os.listdir(self.path):
            if ".pkl.gz" in f: 
                self.biosamples[f[:3]] = f"{self.path}/{f}"
        
        # Sort the keys in availability in descending order
        sorted_keys = sorted(availability, key=availability.get, reverse=True)

        # Create a new dictionary with sorted keys
        self.biosamples = {key: self.biosamples[key] for key in sorted_keys if key in self.biosamples}

        self.preprocessed_datasets = []
        for f in os.listdir(self.path):
            if ".pt" in f and "mixed_dataset" in f: 
                self.preprocessed_datasets.append(f"{self.path}/{f}")
        
    def get_biosample_pkl(self, pkl_path):
        with gzip.open(pkl_path, 'rb') as f:
            loaded_file = pickle.load(f)

        bios_assays = loaded_file.keys()
        assay_availability = {ass: (True if ass in bios_assays else False) for ass in self.all_assays}

        M = len(loaded_file[list(loaded_file.keys())[0]])
        L = len(loaded_file[list(loaded_file.keys())[0]][0][1])
        D = len(self.all_assays)

        missing_f_i = []
        # Initialize an empty list to hold all samples
        all_samples = []
        
        # Iterate over all assays
        for i, assay in enumerate(self.all_assays):
            if assay_availability[assay]:
                # If assay is available, append its signal arrays to all_samples
                assay_samples = []
                for j in range(len(loaded_file[assay])):
                    assay_samples.append(loaded_file[assay][j][1])

            else:
                missing_f_i.append(i)
                # If assay is not available, append -1 of appropriate shape
                assay_samples = []
                for j in range(M):
                    assay_samples.append([-1 for _ in range(L)])
            
            all_samples.append(assay_samples)

        # Convert all_samples to a numpy array and transpose to get shape (M, L, D)
        all_samples_tensor = np.array(all_samples, dtype=np.float32).transpose(1, 2, 0)

        # Convert numpy array to PyTorch tensor
        all_samples_tensor = torch.from_numpy(all_samples_tensor)
        all_samples_tensor = all_samples_tensor.float() 

        # Create a mask tensor
        mask = (all_samples_tensor == -1)

        return all_samples_tensor, mask, missing_f_i
    
    def get_dataset_pt(self, pt_path):
        ds = torch.load(pt_path)
        mask = (ds == -1)
        mask_2 = (ds.sum(dim=1) < 0) # missing assay pattern per sample

        indices = [torch.nonzero(mask_2[i, :], as_tuple=True)[0].tolist() for i in range(mask_2.shape[0])]

        unique_indices = [list(x) for x in set(tuple(x) for x in indices)]
        pattern_dict = {tuple(pattern): [] for pattern in unique_indices}

        for i, pattern in enumerate(indices):
            pattern_dict[tuple(pattern)].append(i)

        return ds, mask, pattern_dict
       
class ExtendedEncodeDataHandler:
    """
        set alias for bios(E) and experiments(M) -> save in json
        navigate all bios-exps -> save in json
        [OPTIONAL: merge bios by donor ID?]
            - write in updated navigation json + alias json
        [train-val-test-split] -- EIC + whole_bios -> save in json
        [OPTIONAL: npz-to-npy?]
            - convert all npz files to npy format
        generate_genome_loci(context_length, ccre=False, frac_genome=0.1)
            if not ccre:
                generate_random_loci(frac_genome, context_length)
            else:
                generate_ccre_loci(context_length)
        load_exp(bios_name, exp_name, locus)
        load_bios(bios_name, locus) 
            - for all available exp for each bios, load all
        
        make tensor(loci)
    """
    def __init__(self, base_path, resolution=25):
        self.base_path = base_path
        self.chr_sizes_file = os.path.join(self.base_path, "hg38.chrom.sizes")
        self.alias_path = os.path.join(self.base_path, "aliases.json")
        self.navigation_path = os.path.join(self.base_path, "navigation.json")
        self.split_path = os.path.join(self.base_path, "train_va_test_split.json")
        
        self.resolution = resolution
        self.df1_path = os.path.join(self.base_path, "DF1.csv")
        self.df1 = pd.read_csv(self.df1_path)

        self.df2_path = os.path.join(self.base_path, "DF2.csv")
        self.df2 = pd.read_csv(self.df2_path).drop("Unnamed: 0", axis=1)

        self.df3_path = os.path.join(self.base_path, "DF3.csv")
        self.df3 = pd.read_csv(self.df3_path).drop("Unnamed: 0", axis=1)

        
        self.eicdf_path = os.path.join(self.base_path, "EIC_experiments.csv")
        self.eic_df = pd.read_csv(self.eicdf_path)
        print(self.eic_df)
        
    def report(self):
        """
        Generates a formatted text report of the dataset.
        """
        print("Extended Encode Data Report")
        print("----------------------------")
        print(f"Total number of complete biosamples: {sum(self.DS_checkup())}")
        
        # Count biosamples with more than n assays available
        assay_count = self.df1.notna().sum(axis=1)
        max_assays = assay_count.max()
        print("\nNumber of biosamples with more than N assays available:")
        for n in range(max_assays, 0, -1):
            count = (assay_count >= n).sum()
            print(f"  - More than {n} assays: {count}")
        
        # Count of biosamples where each assay is available
        print("\nAssays availability in biosamples:")
        for assay in self.df1.columns[1:]:  # Skip 'Accession'
            available_count = self.df1[assay].notna().sum()
            print(f"  - {assay}: {available_count} biosamples")
        
        # Training and testing biosamples
        print("\nTraining and testing biosamples:")
        print(f"  - Number of training biosamples: {len([b for b, s in self.split_dict.items() if s == 'train'])}")
        print(f"  - Number of testing biosamples: {len([b for b, s in self.split_dict.items() if s == 'test'])}")
        
        # Count of isogenic replicates
        isogenic_train_test = self.df2[self.df2['isogenic_replicates'].notnull()]
        isogenic_count = 0
        for index, row in isogenic_train_test.iterrows():
            if row['Accession'] in self.split_dict and self.split_dict[row['Accession']] == 'train' \
            and row['isogenic_replicates'] in self.split_dict and self.split_dict[row['isogenic_replicates']] == 'test':
                isogenic_count += 1
        print(f"  - Number of isogenic replicates, one in train and one in test: {isogenic_count}\n")

    def coords(self, mode="train"):
        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        if mode == "train":
            main_chrs.remove("chr21") # reserved for validation
        self.chr_sizes = {}

        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)    
        
        self.genomesize = sum(list(self.chr_sizes.values()))

    def is_bios_complete(self, bios_name):
        """Check if a biosample has all required files."""
        required_dsfs = ['DSF1', 'DSF2', 'DSF4', 'DSF8']
        missing_files = []

        try:
            available_exps = self.df1.loc[self.df1['Accession'] == bios_name].dropna(axis=1).columns.tolist()[1:]
            available_exps.remove("Accession")
        except Exception as e:
            return f"Error reading DF1.csv: {e}"

        missing_exp = []
        bios_path = os.path.join(self.base_path, bios_name)
        for exp in available_exps:
            exp_path = os.path.join(bios_path, exp)
            exp_listdir = os.listdir(exp_path)
            exp_full = True
            for dsf in required_dsfs:
                if exp_full == True:
                    if  "file_metadata.json" in exp_listdir:
                        if f'signal_{dsf}_res25' in exp_listdir:
                            md1_path = os.path.join(exp_path, f'signal_{dsf}_res25', "metadata.json")
                            exp_full = os.path.exists(md1_path)
                        else:
                            exp_full = False
                    else:
                        exp_full = False
                        
            if exp_full == False:
                missing_exp.append(exp)

        return missing_exp

    def fix_bios(self, bios_name):
        missing_exp = self.is_bios_complete(bios_name)
        missingrows = []
        if len(missing_exp) > 0:
            print(f"fixing {bios_name}!")
            for exp in missing_exp:
                rows = self.df3.loc[(self.df3["bios"] == bios_name)&(self.df3["exp"] == exp), :]
                missingrows.append(rows)

            missingrows = pd.concat(missingrows, axis=0).reset_index(drop=True)
            # print(missingrows)
            for i in range(len(missingrows)):
                dl_dict = {}
                dl_dict["url"] = missingrows.loc[i, "url"]
                dl_dict["save_dir_name"] = missingrows.loc[i, "save_dir_name"]
                dl_dict["exp"] = missingrows.loc[i, "exp"]
                dl_dict["bios"] = missingrows.loc[i, "bios"]
                single_download(dl_dict)
    
    def mp_fix_DS(self, n_p=5):
        bios_list = self.df1.Accession.to_list()
        random.shuffle(bios_list)
        with mp.Pool(n_p) as p:
            p.map(self.fix_bios, bios_list)
        
    def DS_checkup(self):
        bios_list = self.df1.Accession.to_list()
        is_comp = []
        for bs in bios_list:
            missing = self.is_bios_complete(bs)
            if len(missing) > 0:
                is_comp.append(0)
            else:
                is_comp.append(1)
        
        return sum(is_comp) / len(is_comp)

    def set_alias(self, excludes=["ChIA-PET", "CAGE", "RNA-seq"]):
        if os.path.exists(self.alias_path):
            with open(self.alias_path, 'r') as file:
                self.aliases = json.load(file)
            return

        """Set aliases for biosamples, experiments, and donors based on data availability."""
        self.df1.set_index('Accession', inplace=True)
        self.df1 = self.df1.drop("Unnamed: 0", axis=1)

        # Alias for biosamples
        biosample_counts = self.df1.count(axis=1).sort_values(ascending=False)
        num_biosamples = len(biosample_counts)
        biosample_alias = {biosample: f"E{str(index+1).zfill(len(str(num_biosamples)))}" for index, biosample in enumerate(biosample_counts.index)}

        # Alias for experiments
        experiment_counts = self.df1.count().sort_values(ascending=False)
        experiment_counts = experiment_counts.drop(excludes)
        num_experiments = len(experiment_counts)
        experiment_alias = {
            experiment: f"M{str(index+1).zfill(len(str(num_experiments)))}" for index, experiment in enumerate(
                experiment_counts.index)}

        self.aliases = {
            "biosample_aliases": biosample_alias,
            "experiment_aliases": experiment_alias}

        with open(self.alias_path, 'w') as file:
            json.dump(self.aliases, file, indent=4)

    def navigate_bios_exps(self):
        """Navigate all biosample-experiment pairs and save in JSON."""
        navigation = {}
        for bios in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, bios)):
                navigation[bios] = {}
                for exp in os.listdir(os.path.join(self.base_path, bios)):
                    exp_path = os.path.join(self.base_path, bios, exp)
                    if os.path.isdir(exp_path):
                        navigation[bios][exp] = os.listdir(exp_path)
        
        with open(self.navigation_path, 'w') as file:
            json.dump(navigation, file, indent=4)

    def merge_celltypes(self):
        celltypes = {ct:[] for ct in self.df2["Biosample term name"].unique()}
        for i in range(len(self.df2)):
            celltypes[self.df2["Biosample term name"][i]].append(self.df2["Accession"][i])

        new_nav = {}
        for ct in celltypes.keys():
            for sub_bios in celltypes[ct]:
                if sub_bios in self.navigation.keys():
                    if ct not in new_nav.keys():
                        new_nav[ct] = {}

                    for exp in self.navigation[sub_bios].keys():
                        if exp not in new_nav[ct]:
                            new_nav[ct][exp] = self.navigation[sub_bios][exp]
        
        self.navigation = new_nav

    def init_eic_subset(self):
        celltypes = {ct:[] for ct in self.df2["Biosample term name"].unique()}
        for i in range(len(self.df2)):
            celltypes[self.df2["Biosample term name"][i]].append(self.df2["Accession"][i])

        split = {} # keys are bios accessions | values are "train"/"test"/"val"
        nav = {} # keys are bios accessions | values are "train"/"test"/"val"

        aliases = {
            "biosample_aliases": {}, # keys are bios accessions
            "experiment_aliases": {} # keys are exp names
        }

        so_far = {}
        # replace self.navigation
        for i in range(self.eic_df.shape[0]):
            exp_accession = self.eic_df["experiment"][i] 
            exp_type = self.eic_df["mark/assay"][i]
            data_type = self.eic_df["data_type"][i]
            ct = self.eic_df["cell_type"][i].replace("_", " ")
            if ct == "H1-hESC":
                ct = "H1"
            
            # find corresponding bios in df1
            if exp_accession in self.df1[exp_type].values:
                bios_accession = self.df1.loc[self.df1[exp_type] == exp_accession, "Accession"]
                if ct not in so_far.keys():
                    so_far[ct] = []
                so_far[ct].append(bios_accession)
                
            else:

                print("bios missing", exp_type, exp_accession, data_type, ct, so_far[ct])
                # print("found these substitute biosamples", celltypes[ct])


        exit()


        # replace self.split_dict
        # replace self.aliases



    def filter_navigation(self, include=[], exclude=[]):
        """
        filter based on a list of assays to include
        """
        if len(include) == 0 and len(exclude) == 0:
            return

        elif len(exclude) == 0 and len(include) != 0:
            for bios in list(self.navigation.keys()):
                for exp in list(self.navigation[bios].keys()):
                    if exp not in include:
                        del self.navigation[bios][exp]

            for exp in list(self.aliases["experiment_aliases"].keys()):
                if exp not in include:
                    del self.aliases["experiment_aliases"][exp]

        elif len(include) == 0 and len(exclude) != 0:
            for bios in list(self.navigation.keys()):
                for exp in list(self.navigation[bios].keys()):
                    if exp in exclude:
                        del self.navigation[bios][exp]
                        
            for exp in exclude:
                if exp in self.aliases["experiment_aliases"].keys():
                    del self.aliases["experiment_aliases"][exp]

        else:
            return
                        
    def train_val_test_split(self, splits=(0.7, 0.15, 0.15), random_seed=42):
        if os.path.exists(self.split_path):
            with open(self.split_path, 'r') as file:
                self.split_dict = json.load(file)
            return

        if sum(splits) != 1:
            raise ValueError("Sum of splits tuple must be 1.")

        train_size, val_size, test_size = splits
        train_data, temp_data = train_test_split(self.df1, test_size=(1 - train_size), random_state=random_seed)
        relative_val_size = val_size / (val_size + test_size)  # Relative size of validation in the temp data
        val_data, test_data = train_test_split(temp_data, test_size=(1 - relative_val_size), random_state=random_seed)
        
        self.split_dict = {}
        
        for idx in train_data.Accession:
            if self.has_rnaseq(idx):
                self.split_dict[idx] = 'test'
            else:
                self.split_dict[idx] = 'train'
        
        for idx in val_data.Accession:
            if self.has_rnaseq(idx):
                self.split_dict[idx] = 'test'
            else:
                self.split_dict[idx] = 'val'
        
        for idx in test_data.Accession:
            self.split_dict[idx] = 'test'
        
        with open(self.split_path, 'w') as file:
            json.dump(self.split_dict, file, indent=4)

    def convert_npz_to_npy(self):
        """Convert all NPZ files to NPY format."""
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.npz'):
                    data = np.load(os.path.join(root, file))
                    for arr_name in data.files:
                        np.save(os.path.join(root, f"{file.replace('.npz', '')}.npy"), data[arr_name])

    def generate_random_loci(self, m, context_length, exclude_chr=['chr21']):
        """Generate random genomic loci, excluding specified chromosomes."""
        self.m_regions = []
        used_regions = {chr: [] for chr in self.chr_sizes.keys() if chr not in exclude_chr}
        for chr in used_regions.keys():
            size = self.chr_sizes[chr]
            m_c = int(m * (size / self.genomesize)) + 1  # Calculate the proportional count of regions to generate
            mii = 0
            while mii < m_c:
                # Generate a random start position that is divisible by self.resolution
                rand_start = random.randint(0, (size - context_length) // self.resolution) * self.resolution
                rand_end = rand_start + context_length

                # Check if the region overlaps with any existing region in the same chromosome
                if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                    self.m_regions.append([chr, rand_start, rand_end])
                    used_regions[chr].append((rand_start, rand_end))
                    mii += 1
        
        while sum([len(v) for v in used_regions.values()]) < m:
            # Generate a random start position that is divisible by self.resolution
            rand_start = random.randint(0, (size - context_length) // self.resolution) * self.resolution
            rand_end = rand_start + context_length

            # Check if the region overlaps with any existing region in the same chromosome
            if not any(start <= rand_end and end >= rand_start for start, end in used_regions[chr]):
                self.m_regions.append([chr, rand_start, rand_end])
                used_regions[chr].append((rand_start, rand_end))
                mii += 1

    def generate_ccre_loci(self, m, context_length, ccre_filename="GRCh38-cCREs.bed", exclude_chr=['chr21']):
        """Generate loci based on CCRE data."""

        # Implement based on available CCRE data.
        self.ccres = pd.read_csv(os.path.join(self.base_path, ccre_filename), sep="\t", header=None)
        self.ccres.columns = ["chrom", "start", "end", "id1", "id2", "desc"]

        self.ccres = self.ccres[self.ccres["chrom"].isin(self.chr_sizes.keys())]
        self.ccres = self.ccres[~self.ccres["chrom"].isin(exclude_chr)]

        self.m_regions = []
        used_regions = {chr: [] for chr in self.ccres['chrom'].unique()}

        # Sort the DataFrame by chromosome and start position
        self.ccres = self.ccres.sort_values(['chrom', 'start'])

        # Select m/2 regions from the DataFrame
        while len(self.m_regions) < (m):
            while True:
                # Select a random row from the DataFrame
                row = self.ccres.sample(1).iloc[0]

                # Generate a start position that is divisible by self.resolution and within the region
                rand_start = random.randint(row['start'] // self.resolution, (row['end']) // self.resolution) * self.resolution
                rand_end = rand_start + context_length

                # Check if the region overlaps with any existing region in the same chromosome
                if rand_start >= 0 and rand_end <= self.chr_sizes[row['chrom']]:
                    if not any(start <= rand_end and end >= rand_start for start, end in used_regions[row['chrom']]):
                        self.m_regions.append([row['chrom'], rand_start, rand_end])
                        used_regions[row['chrom']].append((rand_start, rand_end))
                        break
                        
    def generate_full_chr_loci(self, context_length, chrs=["chr19"]):
        self.m_regions = []
        for chr in chrs:
            size = (self.chr_sizes[chr] // context_length) * context_length
            for i in range(0, size, context_length):
                self.m_regions.append([chr, i, i+context_length])
        
    def load_npz(self, file_name):
        with np.load(file_name, allow_pickle=True) as data:
            return {file_name.split("/")[-3]: data[data.files[0]]}
        
    def load_bios(self, bios_name, locus, DSF, f_format="npz"):
        """Load all available experiments for a given biosample and locus."""
        
        exps = list(self.navigation[bios_name].keys())
        if "RNA-seq" in exps:
            exps.remove("RNA-seq")

        loaded_data = {}
        loaded_metadata = {}

        npz_files = []
        for e in exps:
            l = os.path.join(self.base_path, bios_name, e, f"signal_DSF{DSF}_res{self.resolution}", f"{locus[0]}.{f_format}")
            npz_files.append(l)

            jsn1 = os.path.join(self.base_path, bios_name, e, f"signal_DSF{DSF}_res{self.resolution}", "metadata.json")
            with open(jsn1, 'r') as jsnfile:
                md1 = json.load(jsnfile)

            jsn2 = os.path.join(self.base_path, bios_name, e, "file_metadata.json")
            with open(jsn2, 'r') as jsnfile:
                md2 = json.load(jsnfile)

            md = {
                "depth":md1["depth"], "coverage":md1["coverage"], 
                "read_length":md2["read_length"], "run_type":md2["run_type"] 
            }
            loaded_metadata[e] = md
            
        # Load files in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            loaded = list(executor.map(self.load_npz, npz_files))
        
        if len(locus) == 1:
            for l in loaded:
                for exp, data in l.items():
                    loaded_data[exp] = data

            return loaded_data, loaded_metadata

        else:
            start_bin = int(locus[1]) // self.resolution
            end_bin = int(locus[2]) // self.resolution
            for l in loaded:
                for exp, data in l.items():
                    loaded_data[exp] = data[start_bin:end_bin]
            
            return loaded_data, loaded_metadata
    
    def select_region_from_loaded_data(self, loaded_data, locus):
        region = {}
        start_bin = int(locus[1]) // self.resolution
        end_bin = int(locus[2]) // self.resolution
        for exp, data in loaded_data.items():
            region[exp] = data[start_bin:end_bin]
        
        return region

    def make_bios_tensor(self, loaded_data, loaded_metadata, missing_value=-1):
        dtensor = []
        mdtensor = []
        availability = []

        L = len(loaded_data[list(loaded_data.keys())[0]])
        i = 0
        for assay, alias in self.aliases["experiment_aliases"].items():
            
            assert i+1 == int(alias.replace("M",""))
            if assay in loaded_data.keys():
                dtensor.append(loaded_data[assay])
                availability.append(1)

                if "single" in loaded_metadata[assay]['run_type'][list(loaded_metadata[assay]['run_type'].keys())[0]]:
                    runt = 0
                elif "pair" in loaded_metadata[assay]['run_type'][list(loaded_metadata[assay]['run_type'].keys())[0]]:
                    runt = 1

                readl = loaded_metadata[assay]['read_length'][list(loaded_metadata[assay]['read_length'].keys())[0]]

                mdtensor.append([
                    np.log2(loaded_metadata[assay]['depth']), loaded_metadata[assay]['coverage'],
                    readl, runt])

            else:
                dtensor.append([missing_value for _ in range(L)])
                availability.append(0)
                mdtensor.append([missing_value, missing_value, missing_value, missing_value])

            i += 1
        
        dtensor = torch.tensor(np.array(dtensor)).permute(1, 0)
        mdtensor = torch.tensor(np.array(mdtensor)).permute(1, 0)
        availability = torch.tensor(np.array(availability))
        return dtensor, mdtensor, availability
    
    def __make_region_tensor(self, list_bios, locus, DSF, max_workers=-1):
        """Load and process data for multiple biosamples in parallel."""
        def load_and_process(bios):
            try:
                loaded_data, loaded_metadata = self.load_bios(bios, locus, DSF)
                return self.make_bios_tensor(loaded_data, loaded_metadata)
            except Exception as e:
                print(f"Failed to process {bios}: {e}")
                return None

        if max_workers == -1:
            max_workers = self.bios_batchsize//2

        # Use ThreadPoolExecutor to handle biosamples in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(load_and_process, list_bios))

        # Aggregate results
        data, metadata, availability = [], [], []
        for result in results:
            if result is not None:
                d, md, avl = result
                data.append(d)
                metadata.append(md)
                availability.append(avl)

        data, metadata, availability = torch.stack(data), torch.stack(metadata), torch.stack(availability)
        return data, metadata, availability

    def make_region_tensor(self, loaded_data, loaded_metadata):
        data, metadata, availability = [], [], []
        for i in range(len(loaded_data)):
            d, md, avl = self.make_bios_tensor(loaded_data[i], loaded_metadata[i])
            data.append(d)
            metadata.append(md)
            availability.append(avl)
        
        data, metadata, availability = torch.stack(data), torch.stack(metadata), torch.stack(availability)
        return data, metadata, availability

    def initialize_EED(self,
        m, context_length, bios_batchsize, loci_batchsize, loci_gen="chr19", 
        bios_min_exp_avail_threshold=4, check_completeness=True, shuffle_bios=True, 
        excludes=["CAGE", "RNA-seq", "ChIA-PET", "H3T11ph", "H2AK9ac"], 
        includes=[], merge_ct=False, eic=False, DSF_list=[1]):

        self.set_alias()
        self.train_val_test_split()
        self.coords(mode="train")

        if loci_gen == "ccre":
            print("generating cCRE loci")
            self.generate_ccre_loci(m, context_length)
        elif loci_gen == "random":
            print("generating random loci")
            self.generate_random_loci(m, context_length)
        else:
            self.generate_full_chr_loci(context_length, chrs=loci_gen)

        print(f"num loci: {len(self.m_regions)}")
        
        if os.path.exists(self.navigation_path) == False:
            self.navigate_bios_exps()
            
        with open(self.navigation_path, 'r') as navfile:
            self.navigation  = json.load(navfile)

        if eic:
            # replace self.navigation
            # replace self.split_dict
            # replcate self.aliases
            self.init_eic_subset()

        else:
            self.filter_navigation(exclude=excludes, include=includes)
        
        

        if merge_ct:
            self.merge_celltypes()

        # filter biosamples
        for bios in list(self.navigation.keys()):
            if len(self.navigation[bios]) < bios_min_exp_avail_threshold:
                del self.navigation[bios]

            elif self.split_dict[bios] != "train":
                del self.navigation[bios]

            elif check_completeness:
                if len(self.is_bios_complete(bios))>0:
                    del self.navigation[bios]

        if shuffle_bios:
            keys = list(self.navigation.keys())
            random.shuffle(keys)
            self.navigation = {key: self.navigation[key] for key in keys}

        self.num_regions = len(self.m_regions)
        self.num_bios = len(self.navigation)

        self.bios_batchsize = bios_batchsize
        self.loci_batchsize = loci_batchsize

        self.loci = {}
        for i in range(len(self.m_regions)):
            if self.m_regions[i][0] not in self.loci.keys():
                self.loci[self.m_regions[i][0]] = []

            self.loci[self.m_regions[i][0]].append(self.m_regions[i])
        self.dsf_list = DSF_list

    def __new_epoch(self):
        self.current_bios_batch_pointer = 0
        self.current_loci_batch_pointer = 0
    
    def __update_batch_pointers(self, cycle_biosamples_first=True):
        if cycle_biosamples_first:
            # Cycle through all biosamples for each loci before moving to the next loci
            if self.current_bios_batch_pointer + self.bios_batchsize >= self.num_bios:
                self.current_bios_batch_pointer = 0
                if self.current_loci_batch_pointer + self.loci_batchsize < self.num_regions:
                    self.current_loci_batch_pointer += self.loci_batchsize
                else:
                    self.current_loci_batch_pointer = 0  # Reset loci pointer after the last one
                    return True
            else:
                self.current_bios_batch_pointer += self.bios_batchsize
        else:
            # Cycle through all loci for each batch of biosamples before moving to the next batch of biosamples
            if self.current_loci_batch_pointer + self.loci_batchsize >= self.num_regions:
                self.current_loci_batch_pointer = 0
                if self.current_bios_batch_pointer + self.bios_batchsize < self.num_bios:
                    self.current_bios_batch_pointer += self.bios_batchsize
                else:
                    self.current_bios_batch_pointer = 0  # Reset biosample pointer after the last one
                    return True
            else:
                self.current_loci_batch_pointer += self.loci_batchsize

        return False

    def __get_batch(self, dsf):
        batch_loci_list = self.m_regions[self.current_loci_batch_pointer : self.current_loci_batch_pointer+self.loci_batchsize]
        batch_bios_list = list(self.navigation.keys())[self.current_bios_batch_pointer : self.current_bios_batch_pointer+self.bios_batchsize]
        
        batch_data = []
        batch_metadata = []
        batch_availability = []

        for locus in batch_loci_list:
            self.make_region_tensor
            d, md, avl = self.__make_region_tensor(batch_bios_list, locus, DSF=dsf)
            batch_data.append(d)
            batch_metadata.append(md)
            batch_availability.append(avl)
        
        batch_data, batch_metadata, batch_availability = torch.concat(batch_data), torch.concat(batch_metadata), torch.concat(batch_availability)
        return batch_data, batch_metadata, batch_availability

    def new_epoch(self):
        self.chr_pointer = 0 
        self.bios_pointer = 0
        self.dsf_pointer = 0
        self.chr_loci_pointer = 0

        batch_bios_list = list(self.navigation.keys())[self.bios_pointer : self.bios_pointer+self.bios_batchsize]
        self.loaded_data = []
        self.loaded_metadata = []

        for bios in batch_bios_list:
            d, md = self.load_bios(bios, [list(self.loci.keys())[self.chr_pointer]], self.dsf_list[self.dsf_pointer])
            self.loaded_data.append(d)
            self.loaded_metadata.append(md)

        self.Y_loaded_data = self.loaded_data
        self.Y_loaded_metadata = self.loaded_metadata

    def update_batch_pointers(self):
        if self.chr_loci_pointer + self.loci_batchsize >= len(self.loci[list(self.loci.keys())[self.chr_pointer]]):
            self.chr_loci_pointer = 0

            if self.dsf_pointer + 1 >= len(self.dsf_list):
                self.dsf_pointer = 0

                if self.bios_pointer + self.bios_batchsize >= self.num_bios:
                    self.bios_pointer = 0 

                    if self.chr_pointer + 1 >= len(self.loci.keys()):
                        self.chr_pointer = 0
                        return True

                    else:
                        self.chr_pointer += 1
                        
                else:
                    self.bios_pointer += self.bios_batchsize
                    
            else: 
                self.dsf_pointer += 1
            
            batch_bios_list = list(self.navigation.keys())[self.bios_pointer : self.bios_pointer+self.bios_batchsize]
            self.loaded_data = []
            self.loaded_metadata = []

            for bios in batch_bios_list:
                d, md = self.load_bios(bios, [list(self.loci.keys())[self.chr_pointer]], self.dsf_list[self.dsf_pointer])
                self.loaded_data.append(d)
                self.loaded_metadata.append(md)
            
            if self.dsf_pointer == 0:
                self.Y_loaded_data = self.loaded_data
                self.Y_loaded_metadata = self.loaded_metadata

        else:
            self.chr_loci_pointer += self.loci_batchsize
        
        return False

    def get_batch(self, side="x"):
        """
        select subset of loci in working chr
        chr_loci = [locus for locus in self.loci if locus[0] == working_chr]
        
        for chr in loci.chrs:
            for batch in biosamples:
                for dsf in dsf_list:
                    load all bios_chr_dsf

                    for locus in chr_loci:
                        return bios_chr_dsf[locus]
        """
        current_chr = list(self.loci.keys())[self.chr_pointer]
        batch_loci_list = self.loci[current_chr][self.chr_loci_pointer : self.chr_loci_pointer+self.loci_batchsize]

        batch_data = []
        batch_metadata = []
        batch_availability = []

        for locus in batch_loci_list:
            loc_d = []

            if side == "x":
                for d in self.loaded_data:
                    loc_d.append(self.select_region_from_loaded_data(d, locus))
                d, md, avl = self.make_region_tensor(loc_d, self.loaded_metadata)

            elif side == "y":
                for d in self.Y_loaded_data:
                    loc_d.append(self.select_region_from_loaded_data(d, locus))
                d, md, avl = self.make_region_tensor(loc_d, self.Y_loaded_metadata)

            batch_data.append(d)
            batch_metadata.append(md)
            batch_availability.append(avl)
        
        
        batch_data, batch_metadata, batch_availability = torch.concat(batch_data), torch.concat(batch_metadata), torch.concat(batch_availability)
        return batch_data, batch_metadata, batch_availability
        
    def init_eval(
        self, context_length, bios_min_exp_avail_threshold=3, 
        check_completeness=False, split="test",
        excludes=["CAGE", "RNA-seq", "ChIA-PET", "H3T11ph", "H2AK9ac"], 
        includes=[]): #split in ["test", "val"]
        self.set_alias()
        self.train_val_test_split()
        self.coords(mode="eval")
        
        if os.path.exists(self.navigation_path) == False:
            self.navigate_bios_exps()
            
        with open(self.navigation_path, 'r') as navfile:
            self.navigation  = json.load(navfile)
        
        self.filter_navigation(exclude=excludes, include=includes)

        # filter biosamples
        for bios in list(self.navigation.keys()):
            if split == "test" and self.has_rnaseq(bios):
                continue

            elif len(self.navigation[bios]) < bios_min_exp_avail_threshold:
                del self.navigation[bios]

            elif self.split_dict[bios] != split:
                del self.navigation[bios]

            elif check_completeness:
                if len(self.is_bios_complete(bios))>0:
                    del self.navigation[bios]
        
        self.num_bios = len(self.navigation)
        self.test_bios = []
        for b, s in self.split_dict.items():
            if s == split:
                if b in list(self.navigation.keys()):
                    self.test_bios.append(b)        

    def has_rnaseq(self, bios_name):
        if os.path.exists(os.path.join(self.base_path, bios_name, "RNA-seq")):
            return True
        else:
            return False

    def load_rna_seq_data(self, bios_name, gene_coord):
        directory = os.path.join(self.base_path, bios_name, "RNA-seq/")
        tsv_files = glob.glob(os.path.join(directory, '*.tsv'))

        file = os.path.join(directory, tsv_files[0])
        trn_data = pd.read_csv(file, sep="\t")

        for j in range(len(trn_data)):
            trn_data.at[j, "gene_id"] = trn_data["gene_id"][j].split(".")[0]
        
        for i in range(len(gene_coord)):
            gene_coord.at[i, "gene_id"] = gene_coord["gene_id"][i].split(".")[0]

        mapped_trn_data = []
        for i in range(len(gene_coord)):
            geneID = gene_coord["gene_id"][i]
            subset = trn_data.loc[trn_data["gene_id"] == geneID, :].reset_index(drop=True)

            if len(subset) > 0:
                mapped_trn_data.append([
                    gene_coord["chr"][i], gene_coord["start"][i], gene_coord["end"][i], gene_coord["strand"][i], geneID, subset["length"][0], subset["TPM"][0], subset["FPKM"][0]
                ])

        mapped_trn_data = pd.DataFrame(mapped_trn_data, columns=["chr", "start", "end", "strand", "geneID", "length", "TPM", "FPKM"])
        return mapped_trn_data

class SyntheticData:
    def __init__(self, n, p, num_features, sequence_length):
        self.n = n
        self.p = p
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.transformations = [
            (self.transform_scale, {'scale': 2}),
            (self.transform_exponential, {'base': 1.03, 'scale': 3}),
            (self.transform_log_scale, {'scale': 40}),
            (self.transform_sqrt_scale, {'scale': 15}),
            (self.transform_piecewise_linear, {'scale_factors': [1.5, 50, 0.5, 100, 0.3]}),
            (self.transform_scaled_sin, {'scale': 30}),
            (self.transform_scaled_cos, {'scale': 30}),
            (self.transform_hyperbolic_sinh, {'scale': 10}),
            (self.transform_polynomial, {'scale': 0.02, 'power': 2}),
            (self.transform_exponential, {'base': 1.05, 'scale': 2}),
            (self.transform_log_scale, {'scale': 60}),
            (self.transform_sqrt_scale, {'scale': 25}),
            (self.transform_piecewise_linear, {'scale_factors': [2, 60, 0.4, 110, 0.2]}),
            (self.transform_scaled_sin, {'scale': 60}),
            (self.transform_scaled_cos, {'scale': 60}),
            (self.transform_hyperbolic_sinh, {'scale': 15}),
            (self.transform_polynomial, {'scale': 0.005, 'power': 3}),
            (self.transform_exponential, {'base': 1.05, 'scale': 1}),
            (self.transform_log_scale, {'scale': 20}),
            (self.transform_sqrt_scale, {'scale': 15}),
            (self.transform_piecewise_linear, {'scale_factors': [2.5, 40, 0.3, 120, 0.25]}),
            (self.transform_scaled_sin, {'scale': 20}),
            (self.transform_scaled_cos, {'scale': 20}),
            (self.transform_hyperbolic_sinh, {'scale': 20}),
            (self.transform_polynomial, {'scale': 0.01, 'power': 3})
        ]

    def generate_base_sequence(self):
        self.base_sequence = np.random.negative_binomial(self.n, self.p, self.sequence_length)
        return self.base_sequence

    def transform_scale(self, sequence, scale):
        return np.clip(sequence * scale, 0, 200)

    def transform_exponential(self, sequence, base, scale):
        return np.clip(np.round(scale * (base ** (sequence))), 0, 1000)

    def transform_log_scale(self, sequence, scale):
        return np.clip(np.round(scale * np.log1p(sequence)), 0, 1000)

    def transform_sqrt_scale(self, sequence, scale):
        return np.clip(np.round(scale * np.sqrt(sequence)), 0, 1000)

    def transform_piecewise_linear(self, sequence, scale_factors):
        transformed = np.zeros_like(sequence)
        for i, value in enumerate(sequence):
            if value < 50:
                transformed[i] = value * scale_factors[0]
            elif value < 150:
                transformed[i] = scale_factors[1] + scale_factors[2] * value
            else:
                transformed[i] = scale_factors[3] + scale_factors[4] * value
        return np.clip(np.round(transformed), 0, 1000)

    def transform_scaled_sin(self, sequence, scale):
        return np.clip(np.round(scale * np.abs(np.sin(sequence))), 0, 1000)

    def transform_scaled_cos(self, sequence, scale):
        return np.clip(np.round(scale * np.abs(np.cos(sequence))), 0, 1000)

    def transform_hyperbolic_sinh(self, sequence, scale):
        return np.clip(np.round(scale * np.sinh(sequence / 50)), 0, 1000)

    def transform_polynomial(self, sequence, scale, power):
        return np.clip(np.round(scale * (sequence ** power)), 0, 1000)

    def apply_transformations(self):
        transformed_sequences = []
        for i in range(self.num_features):
            transform, params = self.transformations[i % len(self.transformations)]
            transformed_seq = transform(self.base_sequence, **params)
            transformed_sequences.append((transformed_seq, transform.__name__))

        return transformed_sequences

    def smooth_sequence(self, sequence, sigma=0.001):
        return gaussian_filter1d(sequence, sigma=sigma)

    def apply_smoothing(self, sequences):
        return [self.smooth_sequence(seq).astype(int) for seq, name in sequences]
    
    def synth_metadata(self, sequences):
        def depth(seq):
            return np.log2((3e9*np.sum(seq))/len(seq))
        def coverage(seq):
            return 100 * np.abs(np.sin(np.sum(seq)))
        def read_length(seq):
            return np.log10(np.sum(seq)+1)
        def run_type(seq):
            if np.mean(seq) <= np.median(seq):
                return 1
            else:
                return 0

        return [np.array([depth(seq), coverage(seq), read_length(seq), run_type(seq)]) for seq, name in sequences]

    def miss(self, sequences, metadata, missing_percentage):
        to_miss = random.choices(range(self.num_features), k=int(self.num_features*missing_percentage))
        avail = [1 for i in range(self.num_features)]

        for miss in to_miss:
            sequences[miss, :] = -1
            metadata[miss, :] = -1
            avail[miss] = 0
        
        return sequences, metadata, avail

    def mask(self, sequences, metadata, avail, mask_percentage):
        to_mask = random.choices([x for x in range(self.num_features) if avail[x]==1], k=int(self.num_features*mask_percentage))

        for mask in to_mask:
            sequences[mask, :] = -2
            metadata[mask, :] = -2
            avail[mask] = -2

        return sequences, metadata, avail

    def get_batch(self, batch_size, miss_perc_range=(0.3, 0.9), mask_perc_range=(0.1, 0.2)):
        batch_X, batch_Y = [], []
        md_batch_X, md_batch_Y = [], []
        av_batch_X, av_batch_Y = [], []
        
        for b in range(batch_size):
            self.generate_base_sequence()
            transformed_sequences = self.apply_transformations()

            smoothed_sequences = self.apply_smoothing(transformed_sequences)
            smoothed_sequences = np.array(smoothed_sequences)

            syn_metadata = self.synth_metadata(transformed_sequences)
            syn_metadata = np.array(syn_metadata)

            miss_p_b = random.uniform(miss_perc_range[0], miss_perc_range[1])
            mask_p_b = random.uniform(mask_perc_range[0], mask_perc_range[1])
            
            y_b, ymd_b, yav_b = self.miss(smoothed_sequences, syn_metadata, miss_p_b)
            x_b, xmd_b, xav_b = self.mask(y_b.copy(), ymd_b.copy(), yav_b.copy(), mask_p_b)

            batch_X.append(x_b)
            batch_Y.append(y_b)

            md_batch_X.append(xmd_b)
            md_batch_Y.append(ymd_b)

            av_batch_X.append(xav_b)
            av_batch_Y.append(yav_b)
        
        batch_X, batch_Y = torch.Tensor(np.array(batch_X)).permute(0, 2, 1), torch.Tensor(np.array(batch_Y)).permute(0, 2, 1)
        md_batch_X, md_batch_Y = torch.Tensor(np.array(md_batch_X)).permute(0, 2, 1), torch.Tensor(np.array(md_batch_Y)).permute(0, 2, 1)
        av_batch_X, av_batch_Y = torch.Tensor(np.array(av_batch_X)), torch.Tensor(np.array(av_batch_Y))

        return batch_X, batch_Y, md_batch_X, md_batch_Y, av_batch_X, av_batch_Y
    
if __name__ == "__main__": 

    solar_data_path = "/project/compbio-lab/encode_data/"
    if sys.argv[1] == "check":
        eed = ExtendedEncodeDataHandler(solar_data_path)
        print(eed.is_bios_complete(sys.argv[2]))

    elif sys.argv[1] == "fix":
        eed = ExtendedEncodeDataHandler(solar_data_path)
        # eed.fix_bios(sys.argv[2])
        eed.mp_fix_DS()
    
    elif sys.argv[1] == "checkup":
        eed = ExtendedEncodeDataHandler(solar_data_path)
        print(eed.DS_checkup())
    
    elif sys.argv[1] == "test":
        eed = ExtendedEncodeDataHandler(solar_data_path)
        eed.set_alias()
        eed.coords(mode="train")

        if os.path.exists(eed.navigation_path) == False:
            eed.navigate_bios_exps()
            
        with open(eed.navigation_path, 'r') as navfile:
            eed.navigation  = json.load(navfile)

        eed.filter_navigation(exclude=["CAGE", "RNA-seq", "ChIA-PET"])
        eed.merge_celltypes()
        # print({ct:len(v) for ct,v in eed.navigation.items()})

        eed.report()

        exit()

        t0 = datetime.datetime.now()

        eed.generate_random_loci(m=10, context_length=20000)

        batch_data, batch_metadata, batch_availability = eed.make_region_tensor(
            ["ENCBS075PNA" for _ in range(5)], ["chr21", 0, eed.chr_sizes["chr21"]], DSF=8)
        batch_data, batch_metadata, batch_availability = torch.concat([batch_data]), torch.concat([batch_metadata]), torch.concat([batch_availability])

        print(batch_data.shape, batch_metadata.shape, batch_availability.shape)

        t1 = datetime.datetime.now()
        print(f"took {t1-t0} ")

    elif sys.argv[1] == "test_solar":
        dataset = ExtendedEncodeDataHandler(solar_data_path)
        dataset.initialize_EED(
            m=10, context_length=200*25, 
            bios_batchsize=50, loci_batchsize=1, ccre=False, 
            bios_min_exp_avail_threshold=4, check_completeness=True)
        
        avail = {}
        for k, v in dataset.navigation.items():
            avail[k] = len(v)
        
        print(avail)
        print(len(avail))
    
    elif sys.argv[1] == "synthetic":
        # Initialize the SyntheticData class with updated parameters
        synthetic_data = SyntheticData(n=0.1485, p=0.0203, num_features=47, sequence_length=1600)

        # Generate and visualize the base sequence
        base_sequence = synthetic_data.generate_base_sequence()

        plt.figure(figsize=(12, 4))
        plt.plot(base_sequence, label='Base Sequence')
        plt.title('Base Sequence')
        plt.xlabel('Position')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

        # Apply transformations to derive 47 distinct sequences
        transformed_sequences = synthetic_data.apply_transformations()

        # Apply smoothing to introduce sequence dependence
        smoothed_sequences = synthetic_data.apply_smoothing(transformed_sequences)
        smoothed_sequences = np.array(smoothed_sequences)

        syn_metadata = synthetic_data.synth_metadata(transformed_sequences)
        syn_metadata = np.array(syn_metadata)

        num_labels = synthetic_data.num_features
        n_cols = math.floor(math.sqrt(num_labels))
        n_rows = math.ceil(num_labels / n_cols)

        # Visualize the smoothed sequences
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 8))
        fig.suptitle('Smoothed Sequences', fontsize=16)
        for i in range(num_labels):
            row, col = divmod(i, n_cols)
            ax = axs[row, col]
            seq = smoothed_sequences[i]
            md = syn_metadata[i]
            ax.plot(seq, label=f'F{i+1}: {md[0]:.1f}-{md[1]:.1f}-{md[2]:.1f}-{md[3]:.1f}')
            # ax.legend(fontsize=5)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    elif sys.argv[1] == "eic":
        dataset = ExtendedEncodeDataHandler(solar_data_path)
        context_length=3200
        resolution = 25

        dataset.initialize_EED(
            m=100, context_length=context_length*resolution, 
            bios_batchsize=50, loci_batchsize=1, loci_gen="random",
            bios_min_exp_avail_threshold=3, check_completeness=True, eic=True)

    else:
        d = GET_DATA()
        d.search_ENCODE(metadata_file_path=solar_data_path)
        d.filter_biosamples(metadata_file_path=solar_data_path)
        d.load_metadata(metadata_file_path=solar_data_path)
        d.get_all(metadata_file_path=solar_data_path)
