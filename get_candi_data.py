#!/usr/bin/env python3
"""
CANDI Data Download and Processing Pipeline

This module provides functionality to download and process ENCODE data
for the CANDI dataset, ensuring reproducibility and publication standards.

Usage:
    python get_candi_data.py --dataset eic --download-directory /path/to/data
    python get_candi_data.py --dataset merged --download-directory /path/to/data
"""

import json
import os
import sys
import argparse
import pandas as pd
import numpy as np
import requests
import datetime
import time
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple fallback progress indicator
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.current = 0
            if desc:
                print(f"{desc}: Starting...")
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if self.desc:
                print(f"{self.desc}: Complete")
        
        def update(self, n=1):
            self.current += n
            if self.total and self.current % max(1, self.total // 10) == 0:
                progress = (self.current / self.total) * 100
                print(f"{self.desc}: {self.current}/{self.total} ({progress:.1f}%)")
        
        def set_postfix(self, *args, **kwargs):
            pass

# Import from existing data.py (with fallbacks for missing dependencies)
try:
    from data import (
        BAM_TO_SIGNAL, download_save, get_binned_values, get_binned_bigBed_peaks,
        extract_donor_information
    )
    HAS_DATA_MODULE = True
except ImportError as e:
    print(f"Warning: Could not import from data.py ({e}). Some functionality will be limited.")
    HAS_DATA_MODULE = False
    
    # Provide fallback functions
    def download_save(url, save_path, max_retries=10):
        """Fallback download function with retry logic."""
        import requests
        import time
        import os
        
        for attempt in range(max_retries):
            try:
                # Remove partial file if it exists
                if os.path.exists(save_path):
                    os.remove(save_path)
                
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                # Memory-efficient download with progress
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0 and downloaded % (50*1024*1024) == 0:  # Log every 50MB
                                progress = (downloaded / total_size) * 100
                                print(f"  Download progress: {progress:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)")
                
                return True
                
            except Exception as e:
                print(f"Download attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Log final failure
                    with open("failed_downloads.log", "a") as log_file:
                        log_file.write(f"{url}\t{save_path}\t{str(e)}\n")
                    return False
    
    def BAM_TO_SIGNAL(*args, **kwargs):
        """Fallback BAM processing - requires actual implementation."""
        raise NotImplementedError("BAM processing requires pysam and other dependencies")
    
    def get_binned_values(*args, **kwargs):
        """Fallback BigWig processing - requires actual implementation."""
        raise NotImplementedError("BigWig processing requires pyBigWig")
    
    def get_binned_bigBed_peaks(*args, **kwargs):
        """Fallback BigBed processing - requires actual implementation."""
        raise NotImplementedError("BigBed processing requires pyBigWig")
    
    def extract_donor_information(data):
        """Fallback donor info extraction."""
        return {"status": "unknown"}


class TaskStatus(Enum):
    """Enumeration for task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Represents a single celltype-assay processing task."""
    celltype: str
    assay: str
    bios_accession: str
    exp_accession: str
    file_accession: str
    bam_accession: Optional[str]
    tsv_accession: Optional[str]
    signal_bigwig_accession: Optional[str]
    peaks_bigbed_accession: Optional[str]
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None
    
    @property
    def task_id(self) -> str:
        """Unique identifier for this task."""
        return f"{self.celltype}-{self.assay}"
    
    @property
    def has_files_to_download(self) -> bool:
        """Check if task has any files to download."""
        return any([
            self.bam_accession,
            self.tsv_accession, 
            self.signal_bigwig_accession,
            self.peaks_bigbed_accession
        ])


class DownloadPlanLoader:
    """Load and parse download plan JSON files."""
    
    def __init__(self, dataset_name: str):
        """
        Initialize loader for download plan.
        
        Args:
            dataset_name (str): Either 'eic' or 'merged'
        """
        self.dataset_name = dataset_name.lower()
        self.download_plan_file = self._get_download_plan_path()
        self.download_plan = self._load_download_plan()
        
    def _get_download_plan_path(self) -> str:
        """Get path to download plan JSON file."""
        if self.dataset_name == 'eic':
            return "data/download_plan_eic.json"
        elif self.dataset_name == 'merged':
            return "data/download_plan_merged.json"
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}. Must be 'eic' or 'merged'")
    
    def _load_download_plan(self) -> Dict:
        """Load and parse download plan JSON file."""
        try:
            if not os.path.exists(self.download_plan_file):
                raise FileNotFoundError(f"Download plan file not found: {self.download_plan_file}")
                
            with open(self.download_plan_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load download plan file {self.download_plan_file}: {e}")
    
    def validate_download_plan(self) -> bool:
        """Validate download plan structure."""
        required_fields = [
            'bios_accession', 'exp_accession', 'file_accession', 
            'bam_accession', 'tsv_accession', 'signal_bigwig_accession', 
            'peaks_bigbed_accession'
        ]
        
        for celltype, assays in self.download_plan.items():
            if not isinstance(assays, dict):
                print(f"Warning: Invalid structure for celltype {celltype}")
                return False
                
            for assay, files in assays.items():
                if not isinstance(files, dict):
                    print(f"Warning: Invalid structure for {celltype}-{assay}")
                    return False
                    
                for field in required_fields:
                    if field not in files:
                        print(f"Warning: Missing field {field} in {celltype}-{assay}")
                        return False
        
        return True
    
    def create_task_list(self) -> List[Task]:
        """Convert download plan to list of Task objects."""
        tasks = []
        
        for celltype, assays in self.download_plan.items():
            for assay, files in assays.items():
                task = Task(
                    celltype=celltype,
                    assay=assay,
                    bios_accession=files['bios_accession'],
                    exp_accession=files['exp_accession'],
                    file_accession=files['file_accession'],
                    bam_accession=files['bam_accession'],
                    tsv_accession=files['tsv_accession'],
                    signal_bigwig_accession=files['signal_bigwig_accession'],
                    peaks_bigbed_accession=files['peaks_bigbed_accession']
                )
                tasks.append(task)
        
        return tasks
    
    def get_experiments_dict(self) -> Dict[str, List[str]]:
        """Get celltype-experiment mapping for compatibility."""
        experiments_dict = {}
        for celltype, assays in self.download_plan.items():
            experiments_dict[celltype] = list(assays.keys())
        return experiments_dict
    
    def get_missing_tasks(self, base_path: str, resolution: int = 25) -> List[Task]:
        """
        Check which tasks are missing required processed files.
        
        Args:
            base_path (str): Base path where data should be stored
            resolution (int): Resolution for file checking
            
        Returns:
            List[Task]: Tasks that need processing
        """
        missing_tasks = []
        all_tasks = self.create_task_list()
        
        for task in all_tasks:
            exp_path = os.path.join(base_path, task.celltype, task.assay)
            completion_status = self._check_experiment_completion(exp_path, resolution)
            
            if completion_status != 'complete':
                if completion_status == 'missing':
                    task.status = TaskStatus.PENDING
                else:  # incomplete
                    task.status = TaskStatus.PENDING
                missing_tasks.append(task)
            else:
                task.status = TaskStatus.COMPLETED
                
        return missing_tasks
    
    def _check_experiment_completion(self, exp_path: str, resolution: int = 25) -> str:
        """
        Check if an experiment has all required processed files.
        
        Returns:
            str: 'complete', 'missing', or 'incomplete'
        """
        if not os.path.exists(exp_path):
            return 'missing'
            
        required_dsf = [1, 2, 4, 8]
        main_chrs = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
        
        # Check file_metadata.json
        if not os.path.exists(os.path.join(exp_path, "file_metadata.json")):
            return 'incomplete'
        
        # Check DSF signal directories
        for dsf in required_dsf:
            dsf_path = os.path.join(exp_path, f"signal_DSF{dsf}_res{resolution}")
            if not os.path.exists(dsf_path):
                return 'incomplete'
                
            # Check metadata.json in DSF directory
            if not os.path.exists(os.path.join(dsf_path, "metadata.json")):
                return 'incomplete'
                
            # Check chromosome files
            for chr_name in main_chrs:
                chr_file = os.path.join(dsf_path, f"{chr_name}.npz")
                if not os.path.exists(chr_file):
                    return 'incomplete'
        
        # Check signal_BW_res25 directory (if BigWig was processed)
        bw_path = os.path.join(exp_path, f"signal_BW_res{resolution}")
        if os.path.exists(bw_path):
            for chr_name in main_chrs:
                chr_file = os.path.join(bw_path, f"{chr_name}.npz")
                if not os.path.exists(chr_file):
                    return 'incomplete'
        
        # Check peaks_res25 directory (if BigBed was processed)
        peaks_path = os.path.join(exp_path, f"peaks_res{resolution}")
        if os.path.exists(peaks_path):
            for chr_name in main_chrs:
                chr_file = os.path.join(peaks_path, f"{chr_name}.npz")
                if not os.path.exists(chr_file):
                    return 'incomplete'
                    
        return 'complete'


class TaskManager:
    """Manage tasks and their execution."""
    
    def __init__(self, base_path: str, resolution: int = 25):
        self.base_path = base_path
        self.resolution = resolution
        self.tasks: List[Task] = []
        self.logger = logging.getLogger(__name__)
        
    def add_tasks(self, tasks: List[Task]) -> None:
        """Add tasks to the manager."""
        self.tasks.extend(tasks)
        
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.PENDING]
        
    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.COMPLETED]
        
    def get_failed_tasks(self) -> List[Task]:
        """Get all failed tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.FAILED]
        
    def update_task_status(self, task_id: str, status: TaskStatus, error_message: str = None) -> None:
        """Update status of a specific task."""
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = status
                if error_message:
                    task.error_message = error_message
                break
        
    def get_task_stats(self) -> Dict[str, int]:
        """Get statistics about task statuses."""
        stats = {}
        for status in TaskStatus:
            stats[status.value] = len([task for task in self.tasks if task.status == status])
        return stats
        
    def filter_tasks_by_celltype(self, celltype: str) -> List[Task]:
        """Get tasks for a specific celltype."""
        return [task for task in self.tasks if task.celltype == celltype]
        
    def print_summary(self) -> None:
        """Print a summary of task statistics."""
        stats = self.get_task_stats()
        total = len(self.tasks)
        
        print(f"\n=== Task Summary ===")
        print(f"Total tasks: {total}")
        for status, count in stats.items():
            if count > 0:
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"{status.capitalize()}: {count} ({percentage:.1f}%)")



class CANDIDownloadManager:
    """Manage downloading and processing of ENCODE files."""
    
    def __init__(self, base_path: str, resolution: int = 25, dsf_list: List[int] = [1,2,4,8]):
        self.base_path = base_path
        self.resolution = resolution
        self.dsf_list = dsf_list
        self.logger = logging.getLogger(__name__)
        
    def process_task(self, task: Task) -> Task:
        """
        Process a single task (download and process all files for one celltype-assay).
        
        Args:
            task (Task): Task to process
        
        Returns:
            Task: Updated task with new status
        """
        try:
            self.logger.info(f"Processing task: {task.task_id}")
            
            # Create experiment directory
            exp_path = os.path.join(self.base_path, task.celltype, task.assay)
            os.makedirs(exp_path, exist_ok=True)
            
            # Check if already completed (resume capability)
            if self._is_task_completed(task, exp_path):
                self.logger.info(f"Task {task.task_id} already completed, skipping")
                task.status = TaskStatus.COMPLETED
                return task
                
            task.status = TaskStatus.IN_PROGRESS
            
            # Create file metadata
            file_metadata = self._create_file_metadata(task)
            self._current_file_metadata = file_metadata  # Store for file size validation
            self._save_file_metadata(file_metadata, exp_path)
            
            # Process BAM file if available
            if task.bam_accession:
                success = self._download_and_process_bam(task, exp_path)
                if not success:
                    raise Exception("Failed to process BAM file")
            
            # Process TSV file if available (RNA-seq)
            if task.tsv_accession:
                success = self._download_and_process_tsv(task, exp_path)
                if not success:
                    raise Exception("Failed to process TSV file")
            
            # Process BigWig file if available  
            if task.signal_bigwig_accession:
                success = self._download_and_process_bigwig(task, exp_path)
                if not success:
                    raise Exception("Failed to process BigWig file")
                
            # Process BigBed file if available
            if task.peaks_bigbed_accession:
                success = self._download_and_process_bigbed(task, exp_path)
                if not success:
                    raise Exception("Failed to process BigBed file")
            
            # Final validation
            if self._is_task_completed(task, exp_path):
                task.status = TaskStatus.COMPLETED
                self.logger.info(f"Task {task.task_id} completed successfully")
                
                # Clean up large files after successful validation
                self._cleanup_large_files(task, exp_path)
            else:
                raise Exception("Task validation failed after processing")
                
        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {str(e)}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
        
        return task
    
    def _is_task_completed(self, task: Task, exp_path: str) -> bool:
        """Check if a task has already been completed."""
        validator = CANDIValidator(self.resolution)
        result = validator.validate_experiment_completion(exp_path)
        return result.get("overall", False)
    
    def _create_file_metadata(self, task: Task) -> Dict:
        """Create file metadata in the expected format by querying ENCODE API."""
        import requests
        
        # Get the primary file accession (BAM for most experiments, TSV for RNA-seq)
        primary_accession = task.bam_accession if task.bam_accession else task.tsv_accession
        
        if not primary_accession:
            raise ValueError(f"No primary file accession found for task {task.task_id}")
        
        # Query ENCODE API for complete file metadata
        file_metadata = self._fetch_encode_file_metadata(primary_accession)
        
        # Format in the expected indexed structure (like the reference format you showed)
        # Using index "2" to match the expected format
        formatted_metadata = {
            "assay": {"2": task.assay},
            "accession": {"2": primary_accession},
            "biosample": {"2": task.bios_accession},
            "file_format": {"2": file_metadata.get("file_format", "bam")},
            "output_type": {"2": file_metadata.get("output_type", "alignments")},
            "experiment": {"2": f"/experiments/{task.exp_accession}/"},
            "bio_replicate_number": {"2": file_metadata.get("biological_replicates", [1])},
            "file_size": {"2": file_metadata.get("file_size", 0)},
            "assembly": {"2": file_metadata.get("assembly", "GRCh38")},
            "download_url": {"2": f"https://www.encodeproject.org/files/{primary_accession}/@@download/{primary_accession}.{file_metadata.get('file_format', 'bam')}"},
            "date_created": {"2": file_metadata.get("date_created", int(time.time() * 1000))},
            "status": {"2": file_metadata.get("status", "released")}
        }
        
        # Add read_length and run_type following the exact logic from data.py get_biosample()
        # Priority: read_length > mapped_read_length > None
        if "read_length" in file_metadata:
            formatted_metadata["read_length"] = {"2": file_metadata["read_length"]}
            formatted_metadata["run_type"] = {"2": file_metadata.get("run_type", "single-ended")}
        elif "mapped_read_length" in file_metadata:
            formatted_metadata["read_length"] = {"2": file_metadata["mapped_read_length"]}
            formatted_metadata["run_type"] = {"2": file_metadata.get("mapped_run_type", "single-ended")}
        else:
            # Always include read_length and run_type, even if None
            formatted_metadata["read_length"] = {"2": None}
            formatted_metadata["run_type"] = {"2": "single-ended"}  # Default value
        
        return formatted_metadata
    
    def _fetch_encode_file_metadata(self, file_accession: str) -> Dict:
        """Fetch complete file metadata from ENCODE API."""
        import requests
        import time
        
        url = f"https://www.encodeproject.org/files/{file_accession}/?format=json"
        headers = {'accept': 'application/json'}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                metadata = response.json()
                self.logger.debug(f"Fetched metadata for {file_accession}")
                return metadata
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} to fetch metadata for {file_accession} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Failed to fetch metadata for {file_accession} after {max_retries} attempts")
                    # Return minimal metadata if API fails
                    return {
                        "file_format": "bam",
                        "output_type": "alignments", 
                        "assembly": "GRCh38",
                        "status": "released",
                        "file_size": 0
                    }
    
    def _save_file_metadata(self, metadata: Dict, exp_path: str) -> None:
        """Save file metadata to file_metadata.json."""
        # Convert to pandas Series to use the to_json method like in data.py
        series_data = pd.Series(metadata)
        
        metadata_path = os.path.join(exp_path, "file_metadata.json")
        with open(metadata_path, "w") as f:
            f.write(series_data.to_json(indent=4))
    
    def _download_and_process_bam(self, task: Task, exp_path: str) -> bool:
        """
        Download and process BAM file.
        
        Args:
            task (Task): Task containing BAM accession
            exp_path (str): Experiment directory path
            
        Returns:
            bool: True if successful
        """
        try:
            bam_file = os.path.join(exp_path, f"{task.bam_accession}.bam")
            download_url = f"https://www.encodeproject.org/files/{task.bam_accession}/@@download/{task.bam_accession}.bam"
            
            # Check if already processed
            if self._are_dsf_signals_complete(exp_path):
                self.logger.info(f"DSF signals already exist for {task.task_id}")
                return True
            
            # Download BAM file
            self.logger.info(f"Downloading BAM file for {task.task_id}")
            if not download_save(download_url, bam_file):
                raise Exception("Failed to download BAM file")
            
            # Validate file size if metadata is available
            if hasattr(self, '_current_file_metadata') and self._current_file_metadata:
                expected_size = self._current_file_metadata.get("file_size", {}).get("2", 0)
                if expected_size > 0:
                    actual_size = os.path.getsize(bam_file)
                    size_diff_percent = abs(actual_size - expected_size) / expected_size * 100
                    if size_diff_percent > 5:  # Allow 5% variance
                        self.logger.warning(f"File size mismatch for {task.task_id}: expected {expected_size}, got {actual_size} ({size_diff_percent:.1f}% difference)")
                    else:
                        self.logger.info(f"File size validation passed for {task.task_id}: {actual_size} bytes")
            
            # Index BAM file
            os.system(f"samtools index {bam_file}")
            
            # Process BAM to signals using existing BAM_TO_SIGNAL
            self.logger.info(f"Processing BAM to signals for {task.task_id}")
            bam_processor = BAM_TO_SIGNAL(
                bam_file=bam_file,
                chr_sizes_file="data/hg38.chrom.sizes"
            )
            bam_processor.full_preprocess()
            
            # Remove BAM file after processing to save space
            os.remove(bam_file)
            if os.path.exists(f"{bam_file}.bai"):
                os.remove(f"{bam_file}.bai")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process BAM for {task.task_id}: {e}")
            # Clean up on failure
            if os.path.exists(bam_file):
                os.remove(bam_file)
            if os.path.exists(f"{bam_file}.bai"):
                os.remove(f"{bam_file}.bai")
            return False
    
    def _download_and_process_tsv(self, task: Task, exp_path: str) -> bool:
        """
        Download and process TSV file (for RNA-seq).
        
        Args:
            task (Task): Task containing TSV accession
            exp_path (str): Experiment directory path
            
        Returns:
            bool: True if successful
        """
        try:
            tsv_file = os.path.join(exp_path, f"{task.tsv_accession}.tsv")
            download_url = f"https://www.encodeproject.org/files/{task.tsv_accession}/@@download/{task.tsv_accession}.tsv"
            
            # Download TSV file
            self.logger.info(f"Downloading TSV file for {task.task_id}")
            if not download_save(download_url, tsv_file):
                raise Exception("Failed to download TSV file")
            
            # For RNA-seq, we just keep the TSV file as is
            # No further processing needed for now
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process TSV for {task.task_id}: {e}")
            if os.path.exists(tsv_file):
                os.remove(tsv_file)
            return False
    
    def _download_and_process_bigwig(self, task: Task, exp_path: str) -> bool:
        """
        Download and process BigWig file.
        
        Args:
            task (Task): Task containing BigWig accession
            exp_path (str): Experiment directory path
            
        Returns:
            bool: True if successful
        """
        try:
            bigwig_file = os.path.join(exp_path, f"{task.signal_bigwig_accession}.bigWig")
            download_url = f"https://www.encodeproject.org/files/{task.signal_bigwig_accession}/@@download/{task.signal_bigwig_accession}.bigWig"
            
            # Check if signal_BW already exists and is complete
            signal_bw_path = os.path.join(exp_path, f"signal_BW_res{self.resolution}")
            if self._is_signal_bw_complete(signal_bw_path):
                self.logger.info(f"BigWig signals already exist for {task.task_id}")
                return True
            
            # Download BigWig file
            self.logger.info(f"Downloading BigWig file for {task.task_id}")
            if not download_save(download_url, bigwig_file):
                raise Exception("Failed to download BigWig file")
                
            # Process using existing function from data.py
            self.logger.info(f"Processing BigWig for {task.task_id}")
            binned_bw = get_binned_values(bigwig_file, bin_size=self.resolution)
            
            # Remove old signal_BW directory if it exists
            if os.path.exists(signal_bw_path):
                import shutil
                shutil.rmtree(signal_bw_path)
            
            # Create signal_BW directory structure
            os.makedirs(signal_bw_path, exist_ok=True)
            
            for chr_name, data in binned_bw.items():
                np.savez_compressed(
                    os.path.join(signal_bw_path, f"{chr_name}.npz"),
                    np.array(data)
                )
            
            # Clean up downloaded file
            os.remove(bigwig_file)
                
            return True
                
        except Exception as e:
            self.logger.error(f"Failed to process BigWig for {task.task_id}: {e}")
            # Clean up on failure
            if os.path.exists(bigwig_file):
                os.remove(bigwig_file)
            return False

    def _download_and_process_bigbed(self, task: Task, exp_path: str) -> bool:
        """
        Download and process BigBed file.
        
        Args:
            task (Task): Task containing BigBed accession
            exp_path (str): Experiment directory path
            
        Returns:
            bool: True if successful
        """
        try:
            bigbed_file = os.path.join(exp_path, f"{task.peaks_bigbed_accession}.bigBed")
            download_url = f"https://www.encodeproject.org/files/{task.peaks_bigbed_accession}/@@download/{task.peaks_bigbed_accession}.bigBed"
            
            # Check if peaks already exist and are complete
            peaks_path = os.path.join(exp_path, f"peaks_res{self.resolution}")
            if self._is_peaks_complete(peaks_path):
                self.logger.info(f"Peaks already exist for {task.task_id}")
                return True
            
            # Download BigBed file
            self.logger.info(f"Downloading BigBed file for {task.task_id}")
            if not download_save(download_url, bigbed_file):
                raise Exception("Failed to download BigBed file")
                
            # Process using existing function from data.py
            self.logger.info(f"Processing BigBed for {task.task_id}")
            binned_peaks = get_binned_bigBed_peaks(bigbed_file, resolution=self.resolution)
            
            # Remove old peaks directory if it exists
            if os.path.exists(peaks_path):
                import shutil
                shutil.rmtree(peaks_path)
            
            # Create peaks directory structure  
            os.makedirs(peaks_path, exist_ok=True)
            
            for chr_name, data in binned_peaks.items():
                np.savez_compressed(
                    os.path.join(peaks_path, f"{chr_name}.npz"),
                    np.array(data)
                )
            
            # Clean up downloaded file
            os.remove(bigbed_file)
                
            return True
                
        except Exception as e:
            self.logger.error(f"Failed to process BigBed for {task.task_id}: {e}")
            # Clean up on failure
            if os.path.exists(bigbed_file):
                os.remove(bigbed_file)
            return False
            
    def _cleanup_large_files(self, task: Task, exp_path: str):
        """Clean up large files after successful processing and validation."""
        try:
            # Remove BAM file (largest file) after successful processing
            if task.bam_accession:
                bam_file = os.path.join(exp_path, f"{task.bam_accession}.bam")
                if os.path.exists(bam_file):
                    file_size = os.path.getsize(bam_file) / (1024 * 1024 * 1024)  # GB
                    os.remove(bam_file)
                    self.logger.info(f"Cleaned up BAM file {task.bam_accession} ({file_size:.1f} GB)")
            
            # Optionally remove other raw files, keeping only processed signals
            # BigWig and BigBed are typically much smaller than BAM, so we can keep them
            # for now unless disk space becomes critical
            
        except Exception as e:
            self.logger.warning(f"Failed to cleanup files for {task.task_id}: {e}")
    
    def _is_task_completed(self, task: Task, exp_path: str) -> bool:
        """Check if task is fully completed."""
        # Check file_metadata.json exists
        metadata_file = os.path.join(exp_path, "file_metadata.json")
        if not os.path.exists(metadata_file):
            return False
        
        # Check required signal directories based on task type
        if task.assay == "RNA-seq":
            # For RNA-seq, only check TSV file exists
            tsv_file = os.path.join(exp_path, f"{task.tsv_accession}.tsv")
            return os.path.exists(tsv_file)
        else:
            # For other assays, check DSF signals, BW signals, and peaks
            if task.bam_accession and not self._are_dsf_signals_complete(exp_path):
                return False
            if task.signal_bigwig_accession:
                signal_bw_path = os.path.join(exp_path, f"signal_BW_res{self.resolution}")
                if not self._is_signal_bw_complete(signal_bw_path):
                    return False
            if task.peaks_bigbed_accession:
                peaks_path = os.path.join(exp_path, f"peaks_res{self.resolution}")
                if not self._is_peaks_complete(peaks_path):
                    return False
        
        return True
            
    def _are_dsf_signals_complete(self, exp_path: str) -> bool:
        """Check if all DSF signal directories are complete."""
        for dsf in self.dsf_list:
            dsf_path = os.path.join(exp_path, f"signal_DSF{dsf}_res{self.resolution}")
            if not self._is_signal_complete(dsf_path):
                return False
        return True
    
    def _is_signal_bw_complete(self, signal_bw_path: str) -> bool:
        """Check if signal_BW directory is complete."""
        return self._is_signal_complete(signal_bw_path)
    
    def _is_peaks_complete(self, peaks_path: str) -> bool:
        """Check if peaks directory is complete."""
        return self._is_signal_complete(peaks_path)
    
    def _is_signal_complete(self, signal_path: str) -> bool:
        """Check if a signal directory has all required chromosome files."""
        if not os.path.exists(signal_path):
            return False
            
        main_chrs = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
        for chr_name in main_chrs:
            chr_file = os.path.join(signal_path, f"{chr_name}.npz")
            if not os.path.exists(chr_file):
                return False
        return True


class ParallelTaskExecutor:
    """Execute tasks in parallel using multiprocessing."""
    
    def __init__(self, download_manager: CANDIDownloadManager, max_workers: int = None):
        self.download_manager = download_manager
        # Limit to number of CPU cores available, never exceed it
        available_cpus = mp.cpu_count()
        if max_workers is None:
            self.max_workers = max(1, available_cpus - 1)  # Leave 1 CPU for system
        else:
            self.max_workers = min(max_workers, available_cpus)  # Never exceed available CPUs
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ParallelTaskExecutor initialized with {self.max_workers} workers (Available CPUs: {available_cpus})")
        
    def execute_tasks(self, tasks: List[Task], show_progress: bool = True) -> List[Task]:
        """
        Execute tasks in parallel.
        
        Args:
            tasks (List[Task]): Tasks to execute
            show_progress (bool): Whether to show progress bar
            
        Returns:
            List[Task]: Updated tasks with results
        """
        if not tasks:
            return []
            
        self.logger.info(f"Executing {len(tasks)} tasks with {self.max_workers} workers")
        
        # For multiprocessing, we need to use a function that can be pickled
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                future = executor.submit(process_single_task, task, self.download_manager.base_path, 
                                       self.download_manager.resolution, self.download_manager.dsf_list)
                future_to_task[future] = task
            
            # Collect results with progress bar
            completed_tasks = []
            if show_progress:
                with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
                    for future in as_completed(future_to_task):
                        try:
                            result_task = future.result()
                            completed_tasks.append(result_task)
                            
                            # Update progress bar description
                            if result_task.status == TaskStatus.COMPLETED:
                                pbar.set_postfix({"Status": "✓ Completed"})
                            elif result_task.status == TaskStatus.FAILED:
                                pbar.set_postfix({"Status": "✗ Failed"})
                            else:
                                pbar.set_postfix({"Status": "? Unknown"})
                                
                            pbar.update(1)
                            
                        except Exception as e:
                            original_task = future_to_task[future]
                            original_task.status = TaskStatus.FAILED
                            original_task.error_message = f"Execution error: {str(e)}"
                            completed_tasks.append(original_task)
                            pbar.set_postfix({"Status": "✗ Error"})
                            pbar.update(1)
            else:
                for future in as_completed(future_to_task):
                    try:
                        result_task = future.result()
                        completed_tasks.append(result_task)
                    except Exception as e:
                        original_task = future_to_task[future]
                        original_task.status = TaskStatus.FAILED
                        original_task.error_message = f"Execution error: {str(e)}"
                        completed_tasks.append(original_task)
        
        return completed_tasks


def process_single_task(task: Task, base_path: str, resolution: int, dsf_list: List[int]) -> Task:
    """
    Process a single task (used for multiprocessing).
    
    This function needs to be at module level for pickling in multiprocessing.
    
    Args:
        task (Task): Task to process
        base_path (str): Base download path
        resolution (int): Resolution for processing
        dsf_list (List[int]): Downsampling factors
        
    Returns:
        Task: Updated task with results
    """
    # Create a new download manager instance for this process
    download_manager = CANDIDownloadManager(base_path, resolution, dsf_list)
    
    # Process the task
    return download_manager.process_task(task)


class CANDIValidator:
    """Validate completion and integrity of processed data."""
    
    def __init__(self, resolution=25):
        self.resolution = resolution
        self.required_dsf = [1, 2, 4, 8]
        self.main_chrs = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
        
    def validate_experiment_completion(self, exp_path):
        """
        Validate that experiment processing is complete.
        
        Args:
            exp_path (str): Path to experiment directory
            
        Returns:
            dict: Validation results for different components
        """
        validation_results = {
            "dsf_signals": self.validate_dsf_signals(exp_path),
            "bw_signals": self.validate_bw_signals(exp_path), 
            "peaks": self.validate_peaks(exp_path),
            "metadata": self.validate_metadata(exp_path),
            "overall": False
        }
        
        # Overall validation requires DSF signals and metadata at minimum
        validation_results["overall"] = (
            validation_results["dsf_signals"] and 
            validation_results["metadata"]
        )
        
        return validation_results
        
    def validate_dsf_signals(self, exp_path):
        """
        Check DSF signal directories and files.
        
        Args:
            exp_path (str): Path to experiment directory
            
        Returns:
            bool: True if all DSF signals are complete
        """
        for dsf in self.required_dsf:
            dsf_path = os.path.join(exp_path, f"signal_DSF{dsf}_res{self.resolution}")
            
            # Check if directory exists
            if not os.path.exists(dsf_path):
                print(f"Missing DSF directory: {dsf_path}")
                return False
                
            # Check metadata.json exists and is valid
            metadata_file = os.path.join(dsf_path, "metadata.json")
            if not os.path.exists(metadata_file):
                print(f"Missing metadata file: {metadata_file}")
                return False
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                # Check required fields
                required_fields = ['coverage', 'depth', 'dsf']
                for field in required_fields:
                    if field not in metadata:
                        print(f"Missing field {field} in {metadata_file}")
                        return False
            except Exception as e:
                print(f"Invalid metadata file {metadata_file}: {e}")
                return False
                
            # Check all chromosome files exist and are valid
            for chr_name in self.main_chrs:
                chr_file = os.path.join(dsf_path, f"{chr_name}.npz")
                if not os.path.exists(chr_file):
                    print(f"Missing chromosome file: {chr_file}")
                    return False
                    
                # Check if file can be loaded
                try:
                    data = np.load(chr_file)
                    if len(data.files) == 0:
                        print(f"Empty NPZ file: {chr_file}")
                        return False
                except Exception as e:
                    print(f"Corrupted NPZ file {chr_file}: {e}")
                    return False
                    
        return True
    
    def validate_bw_signals(self, exp_path):
        """
        Check BigWig signal directory and files.
        
        Args:
            exp_path (str): Path to experiment directory
            
        Returns:
            bool: True if BW signals are complete (or directory doesn't exist)
        """
        bw_path = os.path.join(exp_path, f"signal_BW_res{self.resolution}")
        
        # If directory doesn't exist, consider it valid (not all experiments have BigWig)
        if not os.path.exists(bw_path):
            return True
            
        # If directory exists, check all chromosome files
        for chr_name in self.main_chrs:
            chr_file = os.path.join(bw_path, f"{chr_name}.npz")
            if not os.path.exists(chr_file):
                print(f"Missing BW chromosome file: {chr_file}")
                return False
                
            # Check if file can be loaded
            try:
                data = np.load(chr_file)
                if len(data.files) == 0:
                    print(f"Empty BW NPZ file: {chr_file}")
                    return False
            except Exception as e:
                print(f"Corrupted BW NPZ file {chr_file}: {e}")
                return False
                
        return True
    
    def validate_peaks(self, exp_path):
        """
        Check peaks directory and files.
        
        Args:
            exp_path (str): Path to experiment directory
            
        Returns:
            bool: True if peaks are complete (or directory doesn't exist)
        """
        peaks_path = os.path.join(exp_path, f"peaks_res{self.resolution}")
        
        # If directory doesn't exist, consider it valid (not all experiments have peaks)
        if not os.path.exists(peaks_path):
            return True
            
        # If directory exists, check all chromosome files
        for chr_name in self.main_chrs:
            chr_file = os.path.join(peaks_path, f"{chr_name}.npz")
            if not os.path.exists(chr_file):
                print(f"Missing peaks chromosome file: {chr_file}")
                return False
                
            # Check if file can be loaded
            try:
                data = np.load(chr_file)
                if len(data.files) == 0:
                    print(f"Empty peaks NPZ file: {chr_file}")
                    return False
            except Exception as e:
                print(f"Corrupted peaks NPZ file {chr_file}: {e}")
                return False
                
        return True
    
    def validate_metadata(self, exp_path):
        """
        Check experiment metadata files.
        
        Args:
            exp_path (str): Path to experiment directory
            
        Returns:
            bool: True if metadata is complete and valid
        """
        # Check file_metadata.json exists and is valid
        file_metadata_path = os.path.join(exp_path, "file_metadata.json")
        if not os.path.exists(file_metadata_path):
            print(f"Missing file_metadata.json: {file_metadata_path}")
            return False
            
        try:
            with open(file_metadata_path, 'r') as f:
                file_metadata = json.load(f)
            
            # Check required fields in file metadata
            required_fields = [
                'assay', 'accession', 'biosample', 'file_format', 
                'output_type', 'experiment', 'bio_replicate_number',
                'file_size', 'assembly', 'download_url', 'date_created', 'status'
            ]
            
            for field in required_fields:
                if field not in file_metadata:
                    print(f"Missing field {field} in file_metadata.json")
                    return False
                    
        except Exception as e:
            print(f"Invalid file_metadata.json {file_metadata_path}: {e}")
            return False
        
        # Check all_files.csv exists
        all_files_path = os.path.join(exp_path, "all_files.csv")
        if not os.path.exists(all_files_path):
            print(f"Missing all_files.csv: {all_files_path}")
            return False
            
        try:
            df = pd.read_csv(all_files_path)
            if len(df) == 0:
                print(f"Empty all_files.csv: {all_files_path}")
                return False
        except Exception as e:
            print(f"Invalid all_files.csv {all_files_path}: {e}")
            return False
            
        return True


class CANDIDataPipeline:
    """Main pipeline for CANDI data download and processing."""
    
    def __init__(self, base_path: str, resolution: int = 25, max_workers: int = None):
        self.base_path = base_path
        self.resolution = resolution
        # Limit max_workers to available CPU cores
        available_cpus = mp.cpu_count()
        if max_workers is None:
            self.max_workers = max(1, available_cpus - 1)  # Leave 1 CPU for system
        else:
            self.max_workers = min(max_workers, available_cpus)  # Never exceed available CPUs
        self.validator = CANDIValidator(resolution)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline initialized with {self.max_workers} max workers (Available CPUs: {available_cpus})")
        
    def run_pipeline(self, dataset_name: str, download: bool = True, validate_only: bool = False) -> Dict:
        """
        Main pipeline execution.
        
        Args:
            dataset_name (str): 'eic' or 'merged'
            download (bool): Whether to download missing data
            validate_only (bool): Only validate existing data
            
        Returns:
            dict: Pipeline results and validation report
        """
        print("=== CANDI Data Pipeline Started ===")
        print(f"Dataset: {dataset_name}")
        print(f"Base path: {self.base_path}")
        print(f"Resolution: {self.resolution}bp")
        print(f"Max workers: {self.max_workers or 'auto'}")
        
        # Step 1: Load download plan
        print("\n1. Loading download plan...")
        try:
            loader = DownloadPlanLoader(dataset_name)
            if not loader.validate_download_plan():
                raise ValueError("Download plan validation failed")
            
            all_tasks = loader.create_task_list()
            print(f"Found {len(all_tasks)} total tasks")
            
        except Exception as e:
            print(f"Error loading download plan: {e}")
            return {"status": "error", "message": str(e)}
        
        # Step 2: Initialize task manager
        task_manager = TaskManager(self.base_path, self.resolution)
        task_manager.add_tasks(all_tasks)
        
        # Step 3: Check for missing/incomplete tasks
        print("\n2. Checking for missing/incomplete tasks...")
        missing_tasks = loader.get_missing_tasks(self.base_path, self.resolution)
        
        print(f"Found {len(missing_tasks)} tasks requiring processing")
        task_manager.print_summary()
        
        if validate_only:
            print("\n=== Validation Only Mode ===")
            return self._validate_all_tasks(all_tasks)
        
        if not download or len(missing_tasks) == 0:
            print("\n=== No downloads required ===")
            return {"status": "complete", "missing_count": len(missing_tasks)}
        
        # Step 4: Execute tasks in parallel
        print(f"\n3. Executing {len(missing_tasks)} tasks...")
        download_manager = CANDIDownloadManager(self.base_path, self.resolution)
        executor = ParallelTaskExecutor(download_manager, self.max_workers)
        
        try:
            completed_tasks = executor.execute_tasks(missing_tasks, show_progress=True)
            
            # Update task manager with results
            for completed_task in completed_tasks:
                task_manager.update_task_status(
                    completed_task.task_id, 
                    completed_task.status, 
                    completed_task.error_message
                )
            
        except Exception as e:
            print(f"Error during task execution: {e}")
            return {"status": "error", "message": str(e)}
        
        # Step 5: Final validation and reporting
        print("\n4. Final validation...")
        validation_report = self._validate_all_tasks(all_tasks)
        
        # Print final summary
        task_manager.print_summary()
        
        failed_tasks = task_manager.get_failed_tasks()
        if failed_tasks:
            print(f"\n⚠️  {len(failed_tasks)} tasks failed:")
            for task in failed_tasks[:5]:  # Show first 5 failures
                print(f"  - {task.task_id}: {task.error_message}")
            if len(failed_tasks) > 5:
                print(f"  ... and {len(failed_tasks) - 5} more")
        
        print("\n=== Pipeline Complete ===")
        return validation_report
    
    def _validate_all_tasks(self, tasks: List[Task]) -> Dict:
        """Validate all tasks in the dataset."""
        validation_report = {
            "total_tasks": 0,
            "valid_tasks": 0,
            "invalid_tasks": [],
            "validation_details": {}
        }
        
        for task in tasks:
            validation_report["total_tasks"] += 1
            
            exp_path = os.path.join(self.base_path, task.celltype, task.assay)
            validation_result = self.validator.validate_experiment_completion(exp_path)
            
            task_key = task.task_id
            validation_report["validation_details"][task_key] = validation_result
            
            if validation_result["overall"]:
                validation_report["valid_tasks"] += 1
            else:
                validation_report["invalid_tasks"].append(task_key)
        
        if validation_report["total_tasks"] > 0:
            success_rate = validation_report["valid_tasks"] / validation_report["total_tasks"]
            print(f"Validation Results: {validation_report['valid_tasks']}/{validation_report['total_tasks']} tasks valid ({success_rate:.1%})")
        
        if validation_report["invalid_tasks"]:
            print("Invalid tasks:")
            for task_id in validation_report["invalid_tasks"][:10]:  # Show first 10
                print(f"  - {task_id}")
            if len(validation_report["invalid_tasks"]) > 10:
                print(f"  ... and {len(validation_report['invalid_tasks']) - 10} more")
        
        return validation_report


def main():
    """Command line interface for CANDI data pipeline."""
    parser = argparse.ArgumentParser(
        description="CANDI Data Download and Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and process EIC dataset
  python get_candi_data.py eic /path/to/data
  
  # Download and process merged dataset  
  python get_candi_data.py merged /path/to/data
  
  # Only validate existing data
  python get_candi_data.py eic /path/to/data --validate-only
  
  # Custom resolution and parallelism
  python get_candi_data.py merged /path/to/data --resolution 50 --max-workers 8
        """
    )
    
    parser.add_argument('dataset', choices=['eic', 'merged'],
                       help='Dataset type: eic or merged')
    parser.add_argument('download_directory', 
                       help='Directory where data will be downloaded and processed')
    parser.add_argument('--resolution', default=25, type=int,
                       help='Resolution for binning in base pairs (default: 25)')
    parser.add_argument('--max-workers', type=int,
                       help='Maximum number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing data without downloading')
    parser.add_argument('--no-download', action='store_true',
                       help='Skip downloading, only check what is missing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    download_dir = os.path.abspath(args.download_directory)
    
    # Create download directory if it doesn't exist
    try:
        os.makedirs(download_dir, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create download directory {download_dir}: {e}")
        sys.exit(1)
    
    # Check if download plans exist
    loader = DownloadPlanLoader(args.dataset)
    if not os.path.exists(loader.download_plan_file):
        print(f"Error: Download plan file not found: {loader.download_plan_file}")
        print("Make sure you're running from the correct directory with data/ subdirectory")
        sys.exit(1)
    
    # Initialize and run pipeline
    pipeline = CANDIDataPipeline(
        base_path=download_dir,
        resolution=args.resolution,
        max_workers=args.max_workers
    )
    
    try:
        result = pipeline.run_pipeline(
            dataset_name=args.dataset,
            download=not args.no_download,
            validate_only=args.validate_only
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        
        # Print summary results
        if "total_tasks" in result:
            total = result["total_tasks"]
            valid = result["valid_tasks"]
            print(f"📊 Final Results: {valid}/{total} tasks completed successfully")
            
            if valid < total:
                print(f"⚠️  {total - valid} tasks failed or incomplete")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
