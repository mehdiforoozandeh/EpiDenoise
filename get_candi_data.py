#!/usr/bin/env python3
"""
CANDI Data Download and Processing Pipeline - Unified Interface

This module provides comprehensive functionality to download, process, validate,
visualize, and retry ENCODE data for the CANDI dataset.

Features:
- Download and process EIC/MERGED datasets
- Parallel processing with CPU core limiting
- Comprehensive validation and visualization
- Retry failed experiments with enhanced error handling
- Complete CLI interface for all operations

Usage:
    # Download and process datasets
    python get_candi_data.py process eic /path/to/data
    python get_candi_data.py process merged /path/to/data
    
    # Validation and analysis
    python get_candi_data.py validate eic /path/to/data
    python get_candi_data.py analyze-missing merged /path/to/data
    
    # Visualization
    python get_candi_data.py create-plots /path/to/data
    
    # Retry failed experiments
    python get_candi_data.py retry eic /path/to/data
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
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import logging
from collections import defaultdict
import shutil


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
        elif self.dataset_name == 'eic_test':
            return "data/download_plan_eic_test.json"
        elif self.dataset_name == 'merged_test':
            return "data/download_plan_merged_test.json"
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}. Must be 'eic', 'merged', 'eic_test', or 'merged_test'")
    
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
        
        # Add new fields: sequencing platform and lab
        # Fetch experiment metadata for platform and lab info
        exp_metadata = self._fetch_encode_experiment_metadata(task.exp_accession)
        
        if exp_metadata:
            sequencing_platform = self._extract_sequencing_platform(exp_metadata)
            lab = self._extract_lab_information(exp_metadata)
            
            if sequencing_platform:
                formatted_metadata["sequencing_platform"] = {"2": sequencing_platform}
            
            if lab:
                formatted_metadata["lab"] = {"2": lab}
        
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
    
    def _fetch_encode_experiment_metadata(self, exp_accession: str) -> Dict:
        """Fetch experiment metadata from ENCODE API."""
        import requests
        import time
        
        url = f"https://www.encodeproject.org/experiments/{exp_accession}/?format=json"
        headers = {'accept': 'application/json'}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                metadata = response.json()
                self.logger.debug(f"Fetched experiment metadata for {exp_accession}")
                return metadata
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} to fetch experiment metadata for {exp_accession} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        else:
                    self.logger.error(f"Failed to fetch experiment metadata for {exp_accession} after {max_retries} attempts")
                    return {}
    
    def _extract_sequencing_platform(self, exp_metadata: Dict) -> Optional[str]:
        """Extract sequencing platform information from experiment metadata."""
        # Check original files for platform info (for processed files)
        if 'files' in exp_metadata:
            for file_info in exp_metadata['files']:
                if isinstance(file_info, dict) and 'platform' in file_info:
                    platform = file_info['platform']
                    if isinstance(platform, dict) and 'term_name' in platform:
                        return platform['term_name']
                    elif isinstance(platform, str):
                        return platform
        
        # Check experiment-level platform info
        if 'platform' in exp_metadata:
            platform = exp_metadata['platform']
            if isinstance(platform, dict) and 'term_name' in platform:
                return platform['term_name']
            elif isinstance(platform, str):
                return platform
        
        # Check library-level platform info in replicates
        if 'replicates' in exp_metadata:
            for replicate in exp_metadata['replicates']:
                if 'library' in replicate and 'platform' in replicate['library']:
                    platform = replicate['library']['platform']
                    if isinstance(platform, dict) and 'term_name' in platform:
                        return platform['term_name']
                    elif isinstance(platform, str):
                        return platform
        
        return None
    
    def _extract_lab_information(self, exp_metadata: Dict) -> Optional[str]:
        """Extract lab information from experiment metadata."""
        # Check experiment-level lab info (original lab)
        if 'lab' in exp_metadata:
            lab = exp_metadata['lab']
            if isinstance(lab, dict) and 'title' in lab:
                return lab['title']
            elif isinstance(lab, str):
                return lab
        
        # Check replicate-level lab info (library lab)
        if 'replicates' in exp_metadata:
            for replicate in exp_metadata['replicates']:
                if 'library' in replicate and 'lab' in replicate['library']:
                    lab = replicate['library']['lab']
                    if isinstance(lab, dict) and 'title' in lab:
                        return lab['title']
                    elif isinstance(lab, str):
                        return lab
        
        return None
    
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
                if HAS_TQDM:
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
                    # Fallback without tqdm
                    completed = 0
                    for future in as_completed(future_to_task):
                        try:
                            result_task = future.result()
                            completed_tasks.append(result_task)
                            completed += 1
                            
                            # Progress update every 10 or at end
                            if completed % 10 == 0 or completed == len(tasks):
                                progress = (completed / len(tasks)) * 100
                                print(f"Processing tasks: {completed}/{len(tasks)} ({progress:.1f}%)")
                            
                        except Exception as e:
                            original_task = future_to_task[future]
                            original_task.status = TaskStatus.FAILED
                            original_task.error_message = f"Execution error: {str(e)}"
                            completed_tasks.append(original_task)
                            completed += 1
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


class MetadataUpdater:
    """Update existing file_metadata.json files with new fields."""
    
    def __init__(self, base_path: str, dataset_name: str, max_workers: int = 6, 
                 create_backups: bool = True, force_update: bool = False, test_mode: bool = False, test_count: int = 5):
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.max_workers = max_workers
        self.create_backups = create_backups
        self.force_update = force_update
        self.test_mode = test_mode
        self.test_count = test_count
        self.logger = logging.getLogger(__name__)
        
        # Load download plan to get experiment accessions
        self.loader = DownloadPlanLoader(dataset_name)
        self.download_plan = self.loader.download_plan
        
        # Setup logging directory
        self.log_dir = os.path.join(os.path.dirname(self.base_path), "log")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup file logging
        log_file = os.path.join(self.log_dir, f"metadata_update_{dataset_name}_{int(time.time())}.log")
        self._setup_logging(log_file)
        
    def _setup_logging(self, log_file: str):
        """Setup file logging for metadata updates."""
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info(f"Metadata updater initialized. Logging to: {log_file}")
        
    def update_all_metadata(self) -> Dict:
        """Update metadata for all experiments in the dataset."""
        # Find all experiment directories
        experiment_dirs = self._find_experiment_directories()
        
        self.logger.info(f"Found {len(experiment_dirs)} experiment directories to process")
        print(f"Found {len(experiment_dirs)} experiment directories to process")
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for exp_dir in experiment_dirs:
                future = executor.submit(self._update_single_metadata, exp_dir)
                futures.append(future)
            
            # Collect results
            results = {
                'total': len(experiment_dirs),
                'successful': 0,
                'failed': 0,
                'up_to_date': 0,
                'failures': []
            }
            
            # Process futures with progress tracking
            if HAS_TQDM:
                for future in tqdm(as_completed(futures), total=len(futures), desc="Updating metadata"):
                    try:
                        result = future.result()
                        if result['status'] == 'success':
                            results['successful'] += 1
                            self.logger.info(f"Successfully updated: {result['message']}")
                        elif result['status'] == 'up_to_date':
                            results['up_to_date'] += 1
                            self.logger.info(f"Already up-to-date: {result['message']}")
                        else:
                            results['failed'] += 1
                            results['failures'].append(result['message'])
                            self.logger.error(f"Failed to update: {result['message']}")
                    except Exception as e:
                        results['failed'] += 1
                        error_msg = f"Exception: {str(e)}"
                        results['failures'].append(error_msg)
                        self.logger.error(f"Exception during update: {error_msg}")
            else:
                # Fallback without tqdm
                completed = 0
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        completed += 1
                        if completed % 10 == 0 or completed == len(futures):  # Progress every 10 or at end
                            progress = (completed / len(futures)) * 100
                            print(f"Updating metadata: {completed}/{len(futures)} ({progress:.1f}%)")
                        
                        if result['status'] == 'success':
                            results['successful'] += 1
                            self.logger.info(f"Successfully updated: {result['message']}")
                        elif result['status'] == 'up_to_date':
                            results['up_to_date'] += 1
                            self.logger.info(f"Already up-to-date: {result['message']}")
                        else:
                            results['failed'] += 1
                            results['failures'].append(result['message'])
                            self.logger.error(f"Failed to update: {result['message']}")
                    except Exception as e:
                        results['failed'] += 1
                        error_msg = f"Exception: {str(e)}"
                        results['failures'].append(error_msg)
                        self.logger.error(f"Exception during update: {error_msg}")
        
        # Calculate success score
        success_score = results['successful'] / results['total'] if results['total'] > 0 else 0
        
        self.logger.info(f"Metadata update completed. Success score: {success_score:.2%}")
        self.logger.info(f"Results: {results['successful']} successful, {results['failed']} failed, {results['up_to_date']} up-to-date")
        
        return results
    
    def test_random_experiments(self) -> List[str]:
        """Test the metadata updater on a random subset of experiments."""
        self.test_mode = True
        experiment_dirs = self._find_experiment_directories()
        
        print(f"🧪 Test mode: Processing {len(experiment_dirs)} random experiments:")
        for i, exp_dir in enumerate(experiment_dirs, 1):
            print(f"  {i}. {exp_dir}")
        
        return experiment_dirs
    
    def _find_experiment_directories(self) -> List[str]:
        """Find all experiment directories that contain file_metadata.json."""
        experiment_dirs = []
        
        for celltype in os.listdir(self.base_path):
            celltype_path = os.path.join(self.base_path, celltype)
            if os.path.isdir(celltype_path):
                for assay in os.listdir(celltype_path):
                    assay_path = os.path.join(celltype_path, assay)
                    if os.path.isdir(assay_path):
                        metadata_file = os.path.join(assay_path, "file_metadata.json")
                        if os.path.exists(metadata_file):
                            experiment_dirs.append(assay_path)
        
        # If in test mode, return only a random subset
        if self.test_mode:
            import random
            if len(experiment_dirs) > self.test_count:
                experiment_dirs = random.sample(experiment_dirs, self.test_count)
                self.logger.info(f"Test mode: Selected {len(experiment_dirs)} random experiments for testing")
                print(f"🧪 Test mode: Selected {len(experiment_dirs)} random experiments for testing")
        
        return experiment_dirs
    
    def _update_single_metadata(self, exp_dir: str) -> Dict:
        """Update metadata for a single experiment directory."""
        try:
            # Load existing metadata
            metadata_file = os.path.join(exp_dir, "file_metadata.json")
            with open(metadata_file, 'r') as f:
                existing_metadata = json.load(f)
            
            # Check if already has new fields
            if not self.force_update and 'sequencing_platform' in existing_metadata and 'lab' in existing_metadata:
                return {'status': 'up_to_date', 'message': f"{exp_dir}"}
            
            # Get experiment accession from existing metadata
            experiment_field = existing_metadata.get('experiment', {}).get('2', '')
            if experiment_field.startswith('/experiments/'):
                exp_accession = experiment_field.split('/')[-2]
            else:
                exp_accession = experiment_field
            
            if not exp_accession:
                return {'status': 'failed', 'message': f"No experiment accession found in {exp_dir}"}
            
            # Fetch experiment metadata from ENCODE API
            exp_metadata = self._fetch_encode_experiment_metadata(exp_accession)
            if not exp_metadata:
                return {'status': 'failed', 'message': f"Failed to fetch experiment metadata for {exp_accession}"}
            
            # Extract new fields
            sequencing_platform = self._extract_sequencing_platform(exp_metadata)
            lab = self._extract_lab_information(exp_metadata)
            
            # Create backup if requested
            if self.create_backups:
                backup_file = os.path.join(exp_dir, "file_metadata.json.backup")
                shutil.copy2(metadata_file, backup_file)
                self.logger.info(f"Created backup: {backup_file}")
            
            # Update metadata
            updated_metadata = existing_metadata.copy()
            if sequencing_platform:
                updated_metadata["sequencing_platform"] = {"2": sequencing_platform}
            if lab:
                updated_metadata["lab"] = {"2": lab}
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(updated_metadata, f, indent=4)
            
            return {'status': 'success', 'message': f"{exp_dir} (platform: {sequencing_platform or 'None'}, lab: {lab or 'None'})"}
            
        except Exception as e:
            return {'status': 'failed', 'message': f"Error updating {exp_dir}: {str(e)}"}
    
    def _fetch_encode_experiment_metadata(self, exp_accession: str) -> Dict:
        """Fetch experiment metadata from ENCODE API."""
        import requests
        import time
        
        url = f"https://www.encodeproject.org/experiments/{exp_accession}/?format=json"
        headers = {'accept': 'application/json'}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                metadata = response.json()
                return metadata
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return {}
    
    def _extract_sequencing_platform(self, exp_metadata: Dict) -> Optional[str]:
        """Extract sequencing platform information from experiment metadata."""
        # Check original files for platform info (for processed files)
        if 'files' in exp_metadata:
            for file_info in exp_metadata['files']:
                if isinstance(file_info, dict) and 'platform' in file_info:
                    platform = file_info['platform']
                    if isinstance(platform, dict) and 'term_name' in platform:
                        return platform['term_name']
                    elif isinstance(platform, str):
                        return platform
        
        # Check experiment-level platform info
        if 'platform' in exp_metadata:
            platform = exp_metadata['platform']
            if isinstance(platform, dict) and 'term_name' in platform:
                return platform['term_name']
            elif isinstance(platform, str):
                return platform
        
        # Check library-level platform info in replicates
        if 'replicates' in exp_metadata:
            for replicate in exp_metadata['replicates']:
                if 'library' in replicate and 'platform' in replicate['library']:
                    platform = replicate['library']['platform']
                    if isinstance(platform, dict) and 'term_name' in platform:
                        return platform['term_name']
                    elif isinstance(platform, str):
                        return platform
        
        return None
    
    def _extract_lab_information(self, exp_metadata: Dict) -> Optional[str]:
        """Extract lab information from experiment metadata."""
        # Check experiment-level lab info (original lab)
        if 'lab' in exp_metadata:
            lab = exp_metadata['lab']
            if isinstance(lab, dict) and 'title' in lab:
                return lab['title']
            elif isinstance(lab, str):
                return lab
        
        # Check replicate-level lab info (library lab)
        if 'replicates' in exp_metadata:
            for replicate in exp_metadata['replicates']:
                if 'library' in replicate and 'lab' in replicate['library']:
                    lab = replicate['library']['lab']
                    if isinstance(lab, dict) and 'title' in lab:
                        return lab['title']
                    elif isinstance(lab, str):
                        return lab
        
        return None


class MetadataCSVExporter:
    """Export metadata from file_metadata.json files to CSV format."""
    
    def __init__(self, base_path: str, dataset_name: str):
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.logger = logging.getLogger(__name__)
        
        # Setup output directory
        self.output_dir = "data"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export_to_csv(self) -> str:
        """Export all metadata to CSV file."""
        print(f"📊 Exporting {self.dataset_name.upper()} metadata to CSV...")
        
        # Find all experiment directories
        experiment_data = self._collect_experiment_data()
        
        if not experiment_data:
            print(f"❌ No experiments found in {self.base_path}")
            return ""
        
        # Create DataFrame
        df = pd.DataFrame(experiment_data)
        
        # Define column order
        columns = [
            'biosample_name', 'assay_name', 'bios_accession', 'exp_accession', 
            'file_accession', 'assembly', 'read_length', 'run_type', 
            'sequencing_platform', 'lab', 'depth'
        ]
        
        # Reorder columns and fill missing values
        for col in columns:
            if col not in df.columns:
                df[col] = None
        
        df = df[columns]
        
        # Save to CSV
        output_file = os.path.join(self.output_dir, f"{self.dataset_name}_metadata.csv")
        df.to_csv(output_file, index=False)
        
        print(f"✅ Exported {len(df)} experiments to {output_file}")
        print(f"📊 Columns: {', '.join(columns)}")
        
        # Show sample data
        print(f"\n🔬 Sample data (first 3 rows):")
        print(df.head(3).to_string(index=False))
        
        return output_file
    
    def _collect_experiment_data(self) -> List[Dict]:
        """Collect metadata from all experiment directories."""
        experiment_data = []
        
        # Iterate through biosample directories
        for biosample in os.listdir(self.base_path):
            biosample_path = os.path.join(self.base_path, biosample)
            if not os.path.isdir(biosample_path):
                continue
                
            # Iterate through assay directories
            for assay in os.listdir(biosample_path):
                assay_path = os.path.join(biosample_path, assay)
                if not os.path.isdir(assay_path):
                    continue
                
                # Check for metadata file
                metadata_file = os.path.join(assay_path, "file_metadata.json")
                if not os.path.exists(metadata_file):
                    continue
                
                try:
                    # Load metadata
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract data
                    exp_data = {
                        'biosample_name': biosample,
                        'assay_name': assay,
                        'bios_accession': self._extract_field(metadata, 'biosample'),
                        'exp_accession': self._extract_field(metadata, 'experiment'),
                        'file_accession': self._extract_field(metadata, 'accession'),
                        'assembly': self._extract_field(metadata, 'assembly'),
                        'read_length': self._extract_field(metadata, 'read_length'),
                        'run_type': self._extract_field(metadata, 'run_type'),
                        'sequencing_platform': self._extract_field(metadata, 'sequencing_platform'),
                        'lab': self._extract_field(metadata, 'lab'),
                        'depth': self._extract_depth(assay_path)
                    }
                    
                    # Clean up experiment accession (remove /experiments/ prefix)
                    if exp_data['exp_accession'] and exp_data['exp_accession'].startswith('/experiments/'):
                        exp_data['exp_accession'] = exp_data['exp_accession'].split('/')[-2]
                    
                    experiment_data.append(exp_data)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing {biosample}/{assay}: {e}")
                    continue
        
        return experiment_data
    
    def _extract_field(self, metadata: Dict, field_name: str) -> Optional[str]:
        """Extract field value from metadata using the indexed format."""
        if field_name in metadata and isinstance(metadata[field_name], dict):
            return metadata[field_name].get('2')
        return None
    
    def _extract_depth(self, assay_path: str) -> Optional[float]:
        """Extract sequencing depth from signal_DSF1_res25/metadata.json."""
        try:
            # Path to the DSF1 metadata file
            dsf_metadata_path = os.path.join(assay_path, "signal_DSF1_res25", "metadata.json")
            
            if not os.path.exists(dsf_metadata_path):
                self.logger.warning(f"DSF1 metadata not found: {dsf_metadata_path}")
                return None
            
            # Load the DSF metadata
            with open(dsf_metadata_path, 'r') as f:
                dsf_metadata = json.load(f)
            
            # Extract depth value
            depth = dsf_metadata.get('depth')
            if depth is not None:
                return float(depth)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting depth from {assay_path}: {e}")
            return None


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


@dataclass
class ValidationResult:
    """Results of dataset validation."""
    dataset_name: str
    total_experiments: int
    total_biosamples: int
    completed_experiments: int
    completed_biosamples: int
    completion_percentage: float
    biosample_completion: Dict[str, Dict[str, bool]]
    missing_experiments: List[str]
    available_experiments: List[str]


class CANDIDatasetValidator:
    """Validate completeness of CANDI datasets."""
    
    def __init__(self, base_path: str, resolution: int = 25):
        """
        Initialize validator.
        
        Args:
            base_path: Base path where datasets are stored
            resolution: Resolution for signal files
        """
        self.base_path = Path(base_path)
        self.resolution = resolution
        self.logger = logging.getLogger(__name__)
        
        # Required signal directories for validation
        self.required_signal_dirs = [
            f"peaks_res{resolution}",
            f"signal_BW_res{resolution}",
            f"signal_DSF1_res{resolution}",
            f"signal_DSF2_res{resolution}",
            f"signal_DSF4_res{resolution}",
            f"signal_DSF8_res{resolution}"
        ]
        
        # Main chromosomes to check
        self.main_chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    
    def validate_experiment(self, biosample: str, assay: str, dataset_path: Path) -> bool:
        """
        Validate if a single experiment is complete.
        
        Args:
            biosample: Biosample name
            assay: Assay name
            dataset_path: Path to dataset directory
            
        Returns:
            bool: True if experiment is complete
        """
        exp_path = dataset_path / biosample / assay
        
        if not exp_path.exists():
            return False
        
        # Check file_metadata.json exists
        metadata_file = exp_path / "file_metadata.json"
        if not metadata_file.exists():
            return False
        
        # Special handling for RNA-seq experiments
        if assay == "RNA-seq":
            # For RNA-seq, only check TSV file exists
            # We need to get the TSV accession from the download plan
            # Determine dataset type based on path
            dataset_type = "merged" if "MERGED" in str(dataset_path).upper() else "eic"
            loader = DownloadPlanLoader(dataset_type)
            all_tasks = loader.create_task_list()
            
            # Find the matching task
            matching_task = None
            for task in all_tasks:
                if task.celltype == biosample and task.assay == assay:
                    matching_task = task
                    break
            
            if matching_task and matching_task.tsv_accession:
                tsv_file = exp_path / f"{matching_task.tsv_accession}.tsv"
                return tsv_file.exists()
            else:
                return False
        else:
            # For other assays, check all required signal directories exist
            for signal_dir in self.required_signal_dirs:
                signal_path = exp_path / signal_dir
                if not signal_path.exists():
                    return False
                
                # Check that signal directory has files for main chromosomes
                expected_files = [f"{chrom}.npz" for chrom in self.main_chromosomes]
                existing_files = [f.name for f in signal_path.glob("*.npz")]
                
                # Require at least 80% of chromosomes to be present
                required_count = int(0.8 * len(expected_files))
                if len(existing_files) < required_count:
                    return False
        
        return True
    
    def validate_dataset(self, dataset_name: str, data_directory: str = None) -> ValidationResult:
        """
        Validate complete dataset.
        
        Args:
            dataset_name: 'eic' or 'merged'
            data_directory: Optional specific directory to validate. If None, uses default structure.
            
        Returns:
            ValidationResult: Comprehensive validation results
        """
        # Load expected experiments from download plan
        loader = DownloadPlanLoader(dataset_name)
        all_tasks = loader.create_task_list()
        
        # Use provided data directory or default structure
        if data_directory:
            dataset_path = Path(data_directory)
        else:
            # Default structure (for backward compatibility)
            if dataset_name.lower() == 'eic':
                dataset_path = self.base_path / "DATA_CANDI_EIC"
            elif dataset_name.lower() == 'merged':
                dataset_path = self.base_path / "DATA_CANDI_MERGED"
            elif dataset_name.lower() == 'eic_test':
                dataset_path = self.base_path / "DATA_CANDI_EIC_TEST"
            elif dataset_name.lower() == 'merged_test':
                dataset_path = self.base_path / "DATA_CANDI_MERGED_TEST"
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        
        total_experiments = len(all_tasks)
        total_biosamples = len(set(task.celltype for task in all_tasks))
        
        completed_experiments = 0
        completed_biosamples = set()
        missing_experiments = []
        available_experiments = []
        biosample_completion = defaultdict(dict)
        
        for task in all_tasks:
            is_complete = self.validate_experiment(task.celltype, task.assay, dataset_path)
            biosample_completion[task.celltype][task.assay] = is_complete
            
            experiment_id = f"{task.celltype}-{task.assay}"
            if is_complete:
                completed_experiments += 1
                completed_biosamples.add(task.celltype)
                available_experiments.append(experiment_id)
            else:
                missing_experiments.append(experiment_id)
        
        completion_percentage = (completed_experiments / total_experiments) * 100 if total_experiments > 0 else 0
        
        return ValidationResult(
            dataset_name=dataset_name,
            total_experiments=total_experiments,
            total_biosamples=total_biosamples,
            completed_experiments=completed_experiments,
            completed_biosamples=len(completed_biosamples),
            completion_percentage=completion_percentage,
            biosample_completion=dict(biosample_completion),
            missing_experiments=missing_experiments,
            available_experiments=available_experiments
        )
    
    def quick_validate(self, dataset_name: str, data_directory: str = None, verbose: bool = True) -> Dict[str, float]:
        """
        Quick validation of CANDI dataset completeness.
        
        Args:
            dataset_name: 'eic' or 'merged'
            data_directory: Optional specific directory to validate
            verbose: Print detailed results
            
        Returns:
            Dict with validation results
        """
        validation_result = self.validate_dataset(dataset_name, data_directory)
        
        results = {
            'experiment_completion_rate': validation_result.completion_percentage,
            'biosample_coverage_rate': (validation_result.completed_biosamples / validation_result.total_biosamples) * 100,
            'total_experiments': validation_result.total_experiments,
            'completed_experiments': validation_result.completed_experiments,
            'total_biosamples': validation_result.total_biosamples,
            'biosamples_with_data': validation_result.completed_biosamples
        }
        
        if verbose:
            print(f"\n=== {dataset_name.upper()} Dataset Validation Results ===")
            print(f"Experiments: {results['completed_experiments']}/{results['total_experiments']} "
                  f"({results['experiment_completion_rate']:.1f}% complete)")
            print(f"Biosamples: {results['biosamples_with_data']}/{results['total_biosamples']} "
                  f"({results['biosample_coverage_rate']:.1f}% coverage)")
            
            if validation_result.missing_experiments:
                print(f"\n❌ Missing experiments ({len(validation_result.missing_experiments)}):")
                for exp in validation_result.missing_experiments[:10]:  # Show first 10
                    print(f"  - {exp}")
                if len(validation_result.missing_experiments) > 10:
                    print(f"  ... and {len(validation_result.missing_experiments) - 10} more")
        
        return results
    
    def get_missing_experiments(self, dataset_name: str, data_directory: str = None) -> List[Dict]:
        """
        Get detailed list of missing experiments.
        
        Args:
            dataset_name: 'eic' or 'merged'
            data_directory: Optional specific directory to validate
            
        Returns:
            List of missing experiment details
        """
        validation_result = self.validate_dataset(dataset_name, data_directory)
        
        # Load tasks to get detailed information
        loader = DownloadPlanLoader(dataset_name)
        all_tasks = loader.create_task_list()
        
        missing_details = []
        for task in all_tasks:
            experiment_id = f"{task.celltype}-{task.assay}"
            if experiment_id in validation_result.missing_experiments:
                missing_details.append({
                    'task_id': task.task_id,
                    'celltype': task.celltype,
                    'assay': task.assay,
                    'experiment_id': experiment_id
                })
        
        return missing_details
    
    def analyze_patterns(self, missing_experiments: List[Dict]) -> Dict:
        """
        Analyze patterns in missing experiments.
        
        Args:
            missing_experiments: List of missing experiment details
            
        Returns:
            Dict with analysis results
        """
        if not missing_experiments:
            return {"total_missing": 0, "patterns": {}}
        
        # Count by celltype
        celltype_counts = defaultdict(int)
        assay_counts = defaultdict(int)
        
        for exp in missing_experiments:
            celltype_counts[exp['celltype']] += 1
            assay_counts[exp['assay']] += 1
        
        return {
            "total_missing": len(missing_experiments),
            "by_celltype": dict(celltype_counts),
            "by_assay": dict(assay_counts),
            "top_missing_celltypes": sorted(celltype_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_missing_assays": sorted(assay_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }





class EnhancedCANDIDownloadManager(CANDIDownloadManager):
    """Enhanced download manager with improved error handling for retry scenarios."""
    
    def __init__(self, base_path: str, resolution: int = 25, dsf_list: List[int] = [1,2,4,8]):
        super().__init__(base_path, resolution, dsf_list)
        self.chr_sizes_file = "data/hg38.chrom.sizes"
    
    def process_task(self, task: Task) -> Task:
        """Enhanced task processing with better error handling."""
        try:
            return self._process_task_enhanced(task)
        except Exception as e:
            self.logger.error(f"Enhanced processing failed for {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            return task
    
    def _process_task_enhanced(self, task: Task) -> Task:
        """Core enhanced processing logic."""
        exp_path = os.path.join(self.base_path, task.celltype, task.assay)
        os.makedirs(exp_path, exist_ok=True)
        
        # Enhanced BAM processing with better error handling
        if task.bam_accession:
            success = self._enhanced_bam_processing(task, exp_path)
            if not success:
                raise Exception("Enhanced BAM processing failed")
        
        # Enhanced BigBed processing with invalid interval handling
        if task.peaks_bigbed_accession:
            success = self._enhanced_bigbed_processing(task, exp_path)
            if not success:
                raise Exception("Enhanced BigBed processing failed")
        
        # Standard processing for other file types
        if task.tsv_accession:
            success = self._download_and_process_tsv(task, exp_path)
            if not success:
                raise Exception("TSV processing failed")
        
        if task.signal_bigwig_accession:
            success = self._download_and_process_bigwig(task, exp_path)
            if not success:
                raise Exception("BigWig processing failed")
        
        # Create enhanced metadata
        file_metadata = self._create_file_metadata(task)
        self._save_file_metadata(file_metadata, exp_path)
        
        # Final validation
        if self._is_task_completed(task, exp_path):
            task.status = TaskStatus.COMPLETED
            self.logger.info(f"Enhanced processing completed for {task.task_id}")
        else:
            raise Exception("Enhanced validation failed")
        
        return task
    
    def _enhanced_bam_processing(self, task: Task, exp_path: str) -> bool:
        """Enhanced BAM processing with better indexing validation."""
        try:
            bam_file = os.path.join(exp_path, f"{task.bam_accession}.bam")
            download_url = f"https://www.encodeproject.org/files/{task.bam_accession}/@@download/{task.bam_accession}.bam"
            
            # Check if already processed
            if self._are_dsf_signals_complete(exp_path):
                self.logger.info(f"DSF signals already exist for {task.task_id}")
                return True
            
            # Download with retry
            self.logger.info(f"Downloading BAM file for {task.task_id}")
            if not download_save(download_url, bam_file):
                raise Exception("BAM download failed")
            
            # Enhanced indexing with validation
            self.logger.info(f"Indexing BAM file for {task.task_id}")
            index_result = os.system(f"samtools index {bam_file}")
            if index_result != 0:
                # Retry indexing once
                self.logger.warning(f"First indexing attempt failed for {task.task_id}, retrying...")
                index_result = os.system(f"samtools index {bam_file}")
                if index_result != 0:
                    raise Exception(f"BAM indexing failed after retry: return code {index_result}")
            
            # Validate index file exists
            if not os.path.exists(f"{bam_file}.bai"):
                raise Exception("BAM index file was not created")
            
            # Process BAM to signals
            self.logger.info(f"Processing BAM to signals for {task.task_id}")
            bam_processor = BAM_TO_SIGNAL(
                bam_file=bam_file,
                chr_sizes_file=self.chr_sizes_file
            )
            bam_processor.full_preprocess()
            
            # Clean up BAM files
            os.remove(bam_file)
            if os.path.exists(f"{bam_file}.bai"):
                os.remove(f"{bam_file}.bai")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced BAM processing failed for {task.task_id}: {e}")
            # Clean up on failure
            if os.path.exists(bam_file):
                os.remove(bam_file)
            if os.path.exists(f"{bam_file}.bai"):
                os.remove(f"{bam_file}.bai")
            return False
    
    def _enhanced_bigbed_processing(self, task: Task, exp_path: str) -> bool:
        """Enhanced BigBed processing with invalid interval handling."""
        try:
            bigbed_file = os.path.join(exp_path, f"{task.peaks_bigbed_accession}.bigBed")
            download_url = f"https://www.encodeproject.org/files/{task.peaks_bigbed_accession}/@@download/{task.peaks_bigbed_accession}.bigBed"
            
            # Check if peaks already complete
            peaks_path = os.path.join(exp_path, f"peaks_res{self.resolution}")
            if self._is_peaks_complete(peaks_path):
                self.logger.info(f"Peaks already exist for {task.task_id}")
                return True
            
            # Download BigBed file
            self.logger.info(f"Downloading BigBed file for {task.task_id}")
            if not download_save(download_url, bigbed_file):
                raise Exception("BigBed download failed")
            
            # Enhanced processing with error handling per chromosome
            self.logger.info(f"Processing BigBed for {task.task_id}")
            try:
                binned_peaks = get_binned_bigBed_peaks(bigbed_file, resolution=self.resolution)
            except Exception as e:
                # If standard processing fails, try enhanced processing
                self.logger.warning(f"Standard BigBed processing failed for {task.task_id}: {e}")
                binned_peaks = self._enhanced_bigbed_binning(bigbed_file)
            
            # Remove old peaks directory if exists
            if os.path.exists(peaks_path):
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
            self.logger.error(f"Enhanced BigBed processing failed for {task.task_id}: {e}")
            if os.path.exists(bigbed_file):
                os.remove(bigbed_file)
            return False
    
    def _enhanced_bigbed_binning(self, bigbed_file: str) -> Dict[str, List]:
        """Enhanced BigBed binning with per-chromosome error handling."""
        import pyBigWig
        
        binned_peaks = {}
        main_chrs = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
        
        try:
            bb = pyBigWig.open(bigbed_file)
            
            for chr_name in main_chrs:
                try:
                    # Get chromosome length
                    chr_length = bb.chroms().get(chr_name, 0)
                    if chr_length == 0:
                        self.logger.warning(f"Chromosome {chr_name} not found in BigBed")
                        binned_peaks[chr_name] = []
                        continue
                    
                    # Process chromosome with error handling
                    try:
                        intervals = bb.entries(chr_name, 0, chr_length)
                        if intervals is None:
                            binned_peaks[chr_name] = []
                            continue
                        
                        # Create binned representation
                        num_bins = (chr_length + self.resolution - 1) // self.resolution
                        binned_data = [0] * num_bins
                        
                        for start, end, value in intervals:
                            # Validate interval bounds
                            if start >= end or start < 0 or end > chr_length:
                                self.logger.debug(f"Skipping invalid interval: {start}-{end} on {chr_name}")
                                continue
                            
                            start_bin = start // self.resolution
                            end_bin = min((end - 1) // self.resolution, num_bins - 1)
                            
                            for bin_idx in range(start_bin, end_bin + 1):
                                if 0 <= bin_idx < num_bins:
                                    binned_data[bin_idx] = 1
                        
                        binned_peaks[chr_name] = binned_data
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing chromosome {chr_name}: {e}")
                        binned_peaks[chr_name] = []
                        
                except Exception as e:
                    self.logger.warning(f"Error accessing chromosome {chr_name}: {e}")
                    binned_peaks[chr_name] = []
            
            bb.close()
            
        except Exception as e:
            self.logger.error(f"Error opening BigBed file: {e}")
            # Return empty data for all chromosomes
            for chr_name in main_chrs:
                binned_peaks[chr_name] = []
        
        return binned_peaks


def setup_detailed_logging(log_file: str = None) -> logging.Logger:
    """Set up detailed logging for pipeline operations."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def count_experiments_and_biosamples(all_tasks: List[Task]) -> Tuple[int, int]:
    """Count total experiments and unique biosamples."""
    biosamples = set()
    for task in all_tasks:
        biosamples.add(task.celltype)
    return len(all_tasks), len(biosamples)


def log_progress(logger: logging.Logger, completed_tasks: List[Task], all_tasks: List[Task], current_task: Task = None):
    """Log detailed progress including remaining experiments and biosamples."""
    total_experiments, total_biosamples = count_experiments_and_biosamples(all_tasks)
    
    # Count completed experiments and biosamples
    completed_experiments = len(completed_tasks)
    completed_biosamples = set()
    for task in completed_tasks:
        completed_biosamples.add(task.celltype)
    
    remaining_experiments = total_experiments - completed_experiments
    remaining_biosamples = total_biosamples - len(completed_biosamples)
    
    if current_task:
        logger.info(f"✅ COMPLETED: {current_task.celltype} - {current_task.assay}")
    
    logger.info(f"📊 PROGRESS SUMMARY:")
    logger.info(f"   🧬 Experiments: {completed_experiments}/{total_experiments} completed ({remaining_experiments} remaining)")
    logger.info(f"   🏷️  Biosamples: {len(completed_biosamples)}/{total_biosamples} completed ({remaining_biosamples} remaining)")
    logger.info(f"   📈 Progress: {(completed_experiments/total_experiments)*100:.1f}%")


def main():
    """Comprehensive command line interface for CANDI data pipeline."""
    parser = argparse.ArgumentParser(
        description="CANDI Data Pipeline - Unified Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  process         Download and process datasets
  validate        Validate dataset completeness  
  quick-validate  Quick validation check
  analyze-missing Analyze missing experiments
  retry           Retry failed experiments
  run-complete    Run complete dataset processing with detailed logging

Examples:
  # Process datasets
  python get_candi_data.py process eic /path/to/data
  python get_candi_data.py process merged /path/to/data --max-workers 8
  
  # Validation and analysis
  python get_candi_data.py validate eic /path/to/data
  python get_candi_data.py quick-validate merged /path/to/data
  python get_candi_data.py analyze-missing eic /path/to/data
  
  # Retry failed experiments
  python get_candi_data.py retry eic /path/to/data
  python get_candi_data.py retry merged /path/to/data
  
  # Complete processing with detailed logging
  python get_candi_data.py run-complete eic /path/to/data --max-workers 16
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Download and process datasets')
    process_parser.add_argument('dataset', choices=['eic', 'merged', 'eic_test', 'merged_test'], help='Dataset type')
    process_parser.add_argument('directory', help='Data directory')
    process_parser.add_argument('--resolution', default=25, type=int, help='Resolution in bp (default: 25)')
    process_parser.add_argument('--max-workers', type=int, help='Max parallel workers')
    process_parser.add_argument('--validate-only', action='store_true', help='Only validate, no download')
    process_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Comprehensive dataset validation')
    validate_parser.add_argument('dataset', choices=['eic', 'merged', 'eic_test', 'merged_test'], help='Dataset type')
    validate_parser.add_argument('directory', help='Data directory')
    validate_parser.add_argument('--resolution', default=25, type=int, help='Resolution in bp')
    validate_parser.add_argument('--save-csv', help='Save results to CSV file')
    
    # Quick validate command
    quick_validate_parser = subparsers.add_parser('quick-validate', help='Quick validation check')
    quick_validate_parser.add_argument('dataset', choices=['eic', 'merged', 'eic_test', 'merged_test'], help='Dataset type')
    quick_validate_parser.add_argument('directory', help='Data directory')
    quick_validate_parser.add_argument('--resolution', default=25, type=int, help='Resolution in bp')
    
    # Analyze missing command
    analyze_parser = subparsers.add_parser('analyze-missing', help='Analyze missing experiments')
    analyze_parser.add_argument('dataset', choices=['eic', 'merged', 'eic_test', 'merged_test'], help='Dataset type')
    analyze_parser.add_argument('directory', help='Data directory')
    analyze_parser.add_argument('--resolution', default=25, type=int, help='Resolution in bp')
    analyze_parser.add_argument('--save-json', help='Save results to JSON file')
    

    
    # Retry command
    retry_parser = subparsers.add_parser('retry', help='Retry failed experiments')
    retry_parser.add_argument('dataset', choices=['eic', 'merged', 'eic_test', 'merged_test'], help='Dataset type')
    retry_parser.add_argument('directory', help='Data directory')
    retry_parser.add_argument('--max-workers', default=6, type=int, help='Max parallel workers')
    retry_parser.add_argument('--log-file', help='Log file to analyze for failures')
    
    # Run complete command
    complete_parser = subparsers.add_parser('run-complete', help='Complete processing with detailed logging')
    complete_parser.add_argument('dataset', choices=['eic', 'merged', 'eic_test', 'merged_test'], help='Dataset type')
    complete_parser.add_argument('directory', help='Data directory')
    complete_parser.add_argument('--max-workers', default=16, type=int, help='Max parallel workers')
    complete_parser.add_argument('--log-file', help='Custom log file name')
    
    # Update metadata command
    update_metadata_parser = subparsers.add_parser('update-metadata', help='Update existing file_metadata.json files with new fields')
    update_metadata_parser.add_argument('dataset', choices=['eic', 'merged', 'eic_test', 'merged_test'], help='Dataset type')
    update_metadata_parser.add_argument('directory', help='Data directory')
    update_metadata_parser.add_argument('--max-workers', default=6, type=int, help='Max parallel workers')
    update_metadata_parser.add_argument('--backup', action='store_true', help='Create backups of existing metadata files')
    update_metadata_parser.add_argument('--force', action='store_true', help='Force update even if new fields already exist')
    update_metadata_parser.add_argument('--test', action='store_true', help='Test mode: only process 5 random experiments')
    
    # Export metadata command
    export_metadata_parser = subparsers.add_parser('export-metadata', help='Export metadata to CSV files')
    export_metadata_parser.add_argument('dataset', choices=['eic', 'merged', 'eic_test', 'merged_test'], help='Dataset type')
    export_metadata_parser.add_argument('directory', help='Data directory')
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.command == 'process':
            _handle_process_command(args)
        elif args.command == 'validate':
            _handle_validate_command(args)
        elif args.command == 'quick-validate':
            _handle_quick_validate_command(args)
        elif args.command == 'analyze-missing':
            _handle_analyze_missing_command(args)

        elif args.command == 'retry':
            _handle_retry_command(args)
        elif args.command == 'run-complete':
            _handle_run_complete_command(args)
        elif args.command == 'update-metadata':
            _handle_update_metadata_command(args)
        elif args.command == 'export-metadata':
            _handle_export_metadata_command(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _handle_process_command(args):
    """Handle the process command."""
    directory = os.path.abspath(args.directory)
    os.makedirs(directory, exist_ok=True)
    
    # Check download plans exist
    loader = DownloadPlanLoader(args.dataset)
    if not os.path.exists(loader.download_plan_file):
        raise FileNotFoundError(f"Download plan not found: {loader.download_plan_file}")
    
    # Initialize and run pipeline
    pipeline = CANDIDataPipeline(
        base_path=directory,
        resolution=args.resolution,
        max_workers=args.max_workers
    )
    
    result = pipeline.run_pipeline(
        dataset_name=args.dataset,
        download=not args.validate_only,
        validate_only=args.validate_only
    )
    
    print(f"\n✅ Processing completed!")
    if "total_tasks" in result:
        total = result["total_tasks"]
        valid = result["valid_tasks"]
        print(f"📊 Results: {valid}/{total} tasks completed successfully")


def _handle_validate_command(args):
    """Handle the validate command."""
    directory = os.path.abspath(args.directory)
    
    validator = CANDIDatasetValidator(directory, args.resolution)
    result = validator.validate_dataset(args.dataset, directory)
    
    print(f"\n=== {args.dataset.upper()} Dataset Validation ===")
    print(f"Total experiments: {result.total_experiments}")
    print(f"Completed experiments: {result.completed_experiments} ({result.completion_percentage:.1f}%)")
    print(f"Total biosamples: {result.total_biosamples}")
    print(f"Biosamples with data: {result.completed_biosamples}")
    
    if result.missing_experiments:
        print(f"\n❌ Missing experiments ({len(result.missing_experiments)}):")
        for exp in result.missing_experiments[:20]:  # Show first 20
            print(f"  - {exp}")
        if len(result.missing_experiments) > 20:
            print(f"  ... and {len(result.missing_experiments) - 20} more")
    
    if args.save_csv:
        # Save detailed results to CSV
        data = []
        for biosample, assays in result.biosample_completion.items():
            for assay, is_complete in assays.items():
                data.append({
                    'biosample': biosample,
                    'assay': assay,
                    'is_complete': is_complete,
                    'experiment_id': f"{biosample}-{assay}"
                })
        
        df = pd.DataFrame(data)
        df.to_csv(args.save_csv, index=False)
        print(f"\n💾 Results saved to {args.save_csv}")


def _handle_quick_validate_command(args):
    """Handle the quick-validate command."""
    directory = os.path.abspath(args.directory)
    
    validator = CANDIDatasetValidator(directory, args.resolution)
    results = validator.quick_validate(args.dataset, directory, verbose=True)
    
    print(f"\n📊 Quick Validation Summary:")
    print(f"Experiment completion: {results['experiment_completion_rate']:.1f}%")
    print(f"Biosample coverage: {results['biosample_coverage_rate']:.1f}%")


def _handle_analyze_missing_command(args):
    """Handle the analyze-missing command."""
    directory = os.path.abspath(args.directory)
    
    validator = CANDIDatasetValidator(directory, args.resolution)
    missing_experiments = validator.get_missing_experiments(args.dataset, directory)
    patterns = validator.analyze_patterns(missing_experiments)
    
    print(f"\n=== Missing Experiments Analysis for {args.dataset.upper()} ===")
    print(f"Total missing: {patterns['total_missing']}")
    
    if patterns['total_missing'] > 0:
        print(f"\n🔍 Top missing celltypes:")
        for celltype, count in patterns['top_missing_celltypes']:
            print(f"  {celltype}: {count} experiments")
        
        print(f"\n🔬 Top missing assays:")
        for assay, count in patterns['top_missing_assays']:
            print(f"  {assay}: {count} experiments")
    
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump({
                'missing_experiments': missing_experiments,
                'patterns': patterns
            }, f, indent=2)
        print(f"\n💾 Analysis saved to {args.save_json}")





def _handle_retry_command(args):
    """Handle the retry command."""
    directory = os.path.abspath(args.directory)
    
    # Setup logging for retry
    log_file = args.log_file or f"retry_{args.dataset}_processing.log"
    logger = setup_detailed_logging(log_file)
    
    logger.info(f"🔧 Starting {args.dataset.upper()} failed experiments retry")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Base path: {directory}")
    logger.info(f"Max workers: {args.max_workers}")
    
    # Get missing tasks (which are essentially "failed" tasks)
    loader = DownloadPlanLoader(args.dataset)
    missing_tasks = loader.get_missing_tasks(directory)
    
    if not missing_tasks:
        print(f"✅ No failed {args.dataset.upper()} experiments found to retry")
        return
    
    print(f"Found {len(missing_tasks)} failed {args.dataset.upper()} experiments to retry")
    logger.info(f"Found {len(missing_tasks)} failed experiments to retry")
    
    # Create enhanced download manager
    download_manager = EnhancedCANDIDownloadManager(base_path=directory)
    
    # Process retry tasks
    successful_retries = 0
    failed_retries = 0
    
    for task in missing_tasks:
        logger.info(f"Retrying {args.dataset.upper()} task: {task.task_id}")
        
        try:
            result_task = download_manager.process_task(task)
            if result_task.status == TaskStatus.COMPLETED:
                successful_retries += 1
                logger.info(f"✅ Successfully retried {args.dataset.upper()} task: {task.task_id}")
            else:
                failed_retries += 1
                error_msg = result_task.error_message or "Unknown error"
                logger.error(f"❌ {args.dataset.upper()} retry failed: {task.task_id} - {error_msg}")
        except Exception as e:
            failed_retries += 1
            logger.error(f"❌ Exception during {args.dataset.upper()} retry of {task.task_id}: {e}")
    
    # Final summary
    print(f"\n=== {args.dataset.upper()} RETRY SUMMARY ===")
    print(f"Total retry attempts: {len(missing_tasks)}")
    print(f"Successful retries: {successful_retries}")
    print(f"Failed retries: {failed_retries}")
    print(f"Success rate: {successful_retries/len(missing_tasks)*100:.1f}%")
    
    logger.info(f"\n=== {args.dataset.upper()} RETRY SUMMARY ===")
    logger.info(f"Total retry attempts: {len(missing_tasks)}")
    logger.info(f"Successful retries: {successful_retries}")
    logger.info(f"Failed retries: {failed_retries}")
    logger.info(f"Success rate: {successful_retries/len(missing_tasks)*100:.1f}%")
    
    if failed_retries > 0:
        print(f"⚠️  {failed_retries} experiments still failed after retry")
        logger.warning(f"⚠️  {failed_retries} experiments still failed after retry")


def _handle_run_complete_command(args):
    """Handle the run-complete command with detailed logging."""
    directory = os.path.abspath(args.directory)
    os.makedirs(directory, exist_ok=True)
    
    # Setup detailed logging
    log_file = args.log_file or f"{args.dataset}_complete_processing.log"
    logger = setup_detailed_logging(log_file)
    
    logger.info("=" * 80)
    logger.info(f"🎯 {args.dataset.upper()} DATASET COMPLETE PROCESSING STARTED")
    logger.info("=" * 80)
    logger.info(f"📁 Dataset: {args.dataset}")
    logger.info(f"📂 Base path: {directory}")
    logger.info(f"⚡ Max workers: {args.max_workers}")
    logger.info(f"📝 Log file: {log_file}")
    
    try:
        # Load tasks
        loader = DownloadPlanLoader(args.dataset)
        all_tasks = loader.create_task_list()
        missing_tasks = loader.get_missing_tasks(directory)
        
        total_experiments, total_biosamples = count_experiments_and_biosamples(all_tasks)
        
        logger.info(f"📋 DATASET OVERVIEW:")
        logger.info(f"   🧬 Total experiments: {total_experiments}")
        logger.info(f"   🏷️  Total biosamples: {total_biosamples}")
        logger.info(f"   📥 Tasks to process: {len(missing_tasks)}")
        
        if len(missing_tasks) == 0:
            print(f"✅ All {args.dataset.upper()} experiments already completed!")
            logger.info(f"✅ All experiments already completed!")
            return
        
        # Process with detailed progress logging
        download_manager = CANDIDownloadManager(directory)
        
        print(f"🚀 Starting complete {args.dataset.upper()} processing...")
        print(f"📊 Processing {len(missing_tasks)}/{total_experiments} experiments")
        print(f"⚡ Using {args.max_workers} parallel workers")
        print(f"📝 Logging to: {log_file}")
        
        # Use enhanced executor with detailed logging
        class DetailedParallelTaskExecutor(ParallelTaskExecutor):
            def __init__(self, download_manager, max_workers=None, logger=None, all_tasks=None):
                super().__init__(download_manager, max_workers)
                self.logger = logger or logging.getLogger(__name__)
                self.all_tasks = all_tasks or []
                self.completed_tasks = []
            
            def execute_tasks(self, tasks, show_progress=True):
                self.logger.info(f"🚀 Starting parallel execution of {len(tasks)} tasks")
                self.logger.info(f"⚡ Using {self.max_workers} parallel workers")
                
                # Process tasks with detailed logging
                processed_tasks = super().execute_tasks(tasks, show_progress)
                
                # Log progress after each completion
                self.completed_tasks.extend(processed_tasks)
                log_progress(self.logger, self.completed_tasks, self.all_tasks)
                
                return processed_tasks
        
        executor = DetailedParallelTaskExecutor(
            download_manager, 
            args.max_workers, 
            logger, 
            all_tasks
        )
        
        completed_tasks = executor.execute_tasks(missing_tasks, show_progress=True)
        
        # Final summary
        successful = len([t for t in completed_tasks if t.status == TaskStatus.COMPLETED])
        failed = len([t for t in completed_tasks if t.status == TaskStatus.FAILED])
        
        print(f"\n🎉 {args.dataset.upper()} COMPLETE PROCESSING FINISHED!")
        print(f"✅ Successful: {successful}/{len(missing_tasks)}")
        print(f"❌ Failed: {failed}/{len(missing_tasks)}")
        print(f"📊 Success rate: {successful/len(missing_tasks)*100:.1f}%")
        
        logger.info(f"\n🎉 {args.dataset.upper()} COMPLETE PROCESSING FINISHED!")
        logger.info(f"✅ Successful: {successful}/{len(missing_tasks)}")
        logger.info(f"❌ Failed: {failed}/{len(missing_tasks)}")
        logger.info(f"📊 Success rate: {successful/len(missing_tasks)*100:.1f}%")
        
        if failed > 0:
            failed_tasks = [t for t in completed_tasks if t.status == TaskStatus.FAILED]
            print(f"\n⚠️  Failed experiments:")
            for task in failed_tasks[:10]:  # Show first 10
                print(f"  - {task.task_id}: {task.error_message}")
                logger.error(f"FAILED: {task.task_id} - {task.error_message}")
            if len(failed_tasks) > 10:
                print(f"  ... and {len(failed_tasks) - 10} more (see log file)")
        
    except Exception as e:
        logger.error(f"❌ Complete processing failed: {e}")
        raise


def _handle_update_metadata_command(args):
    """Handle the update-metadata command."""
    directory = os.path.abspath(args.directory)
    
    print(f"=== Updating metadata for {args.dataset.upper()} dataset ===")
    print(f"Directory: {directory}")
    print(f"Max workers: {args.max_workers}")
    print(f"Create backups: {args.backup}")
    print(f"Force update: {args.force}")
    
    # Initialize metadata updater
    updater = MetadataUpdater(
        base_path=directory,
        dataset_name=args.dataset,
        max_workers=args.max_workers,
        create_backups=args.backup,
        force_update=args.force,
        test_mode=args.test,
        test_count=5
    )
    
    # Run metadata update
    results = updater.update_all_metadata()
    
    # Print results
    print(f"\n=== Metadata Update Results ===")
    print(f"Total experiments processed: {results['total']}")
    print(f"Successfully updated: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Already up-to-date: {results['up_to_date']}")
    
    # Calculate and display success score
    success_score = results['successful'] / results['total'] if results['total'] > 0 else 0
    print(f"Success score: {success_score:.2%}")
    
    if results['failed'] > 0:
        print(f"\n⚠️  Failed updates:")
        for failure in results['failures'][:10]:  # Show first 10
            print(f"  - {failure}")
        if len(results['failures']) > 10:
            print(f"  ... and {len(results['failures']) - 10} more")
    
    print(f"\n💾 Log file created in: {os.path.join(os.path.dirname(directory), 'log')}")


def _handle_export_metadata_command(args):
    """Handle the export-metadata command."""
    directory = os.path.abspath(args.directory)
    
    print(f"=== Exporting metadata for {args.dataset.upper()} dataset ===")
    print(f"Directory: {directory}")
    print(f"Output: ./data/{args.dataset}_metadata.csv")
    
    # Initialize metadata exporter
    exporter = MetadataCSVExporter(
        base_path=directory,
        dataset_name=args.dataset
    )
    
    # Export metadata to CSV
    output_file = exporter.export_to_csv()
    
    if output_file:
        print(f"\n✅ Metadata export completed successfully!")
        print(f"📁 Output file: {output_file}")
        
        # Show file info
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"📊 File size: {file_size / 1024:.1f} KB")
    else:
        print(f"\n❌ Metadata export failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
