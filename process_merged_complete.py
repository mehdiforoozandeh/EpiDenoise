#!/usr/bin/env python3
"""
Complete MERGED dataset processing script.
Process all experiments in the MERGED dataset with parallel execution and detailed logging.
"""

import sys
import os
import time
import logging
from pathlib import Path
import multiprocessing as mp

# Add current directory to Python path
sys.path.append('.')

from get_candi_data import DownloadPlanLoader, CANDIDownloadManager, ParallelTaskExecutor

def setup_logging():
    """Set up detailed logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('merged_complete_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def count_experiments_and_biosamples(all_tasks):
    """Count total experiments and unique biosamples"""
    biosamples = set()
    for task in all_tasks:
        biosamples.add(task.celltype)
    
    return len(all_tasks), len(biosamples)

def log_progress(logger, completed_tasks, all_tasks, current_task=None):
    """Log detailed progress including remaining experiments and biosamples"""
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

class DetailedParallelTaskExecutor(ParallelTaskExecutor):
    """Extended ParallelTaskExecutor with detailed progress logging"""
    
    def __init__(self, download_manager, max_workers=None, logger=None, all_tasks=None):
        super().__init__(download_manager, max_workers)
        self.logger = logger or logging.getLogger(__name__)
        self.all_tasks = all_tasks or []
        self.completed_tasks = []
    
    def execute_tasks(self, tasks, show_progress=True):
        """Execute tasks with detailed progress logging"""
        self.logger.info(f"🚀 Starting parallel execution of {len(tasks)} tasks")
        self.logger.info(f"⚡ Using {self.max_workers} parallel workers")
        
        total_experiments, total_biosamples = count_experiments_and_biosamples(self.all_tasks)
        self.logger.info(f"📋 DATASET OVERVIEW:")
        self.logger.info(f"   🧬 Total experiments: {total_experiments}")
        self.logger.info(f"   🏷️  Total biosamples: {total_biosamples}")
        
        # Process tasks
        processed_tasks = super().execute_tasks(tasks, show_progress)
        
        # Update completed tasks and log final progress
        self.completed_tasks.extend(processed_tasks)
        log_progress(self.logger, self.completed_tasks, self.all_tasks)
        
        return processed_tasks

def main():
    logger = setup_logging()
    
    # Configuration
    dataset = "merged"
    base_path = "/home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_MERGED"
    max_workers = 16  # 16 parallel experiments
    
    logger.info("=" * 80)
    logger.info("🎯 MERGED DATASET COMPLETE PROCESSING STARTED")
    logger.info("=" * 80)
    logger.info(f"📁 Dataset: {dataset}")
    logger.info(f"📂 Base path: {base_path}")
    logger.info(f"💻 Available CPUs: {mp.cpu_count()}")
    logger.info(f"⚡ Max workers: {max_workers}")
    logger.info(f"🕐 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Create base directory
        os.makedirs(base_path, exist_ok=True)
        
        # Load all tasks
        logger.info("📋 Loading download plan...")
        loader = DownloadPlanLoader(dataset)
        all_tasks = loader.create_task_list()
        
        total_experiments, total_biosamples = count_experiments_and_biosamples(all_tasks)
        logger.info(f"✅ Loaded {total_experiments} experiments across {total_biosamples} biosamples")
        
        # Check for already completed tasks
        logger.info("🔍 Checking for already completed experiments...")
        missing_tasks = loader.get_missing_tasks(base_path)
        
        already_completed = len(all_tasks) - len(missing_tasks)
        logger.info(f"📊 Status check:")
        logger.info(f"   ✅ Already completed: {already_completed} experiments")
        logger.info(f"   📥 Need processing: {len(missing_tasks)} experiments")
        
        if not missing_tasks:
            logger.info("🎉 All experiments already completed!")
            return
        
        # Create download manager and executor
        download_manager = CANDIDownloadManager(base_path, resolution=25)
        executor = DetailedParallelTaskExecutor(
            download_manager, 
            max_workers=max_workers,
            logger=logger,
            all_tasks=all_tasks
        )
        
        logger.info(f"⚙️  Initialized executor with {executor.max_workers} workers")
        
        # Process tasks
        start_time = time.time()
        logger.info("🚀 Starting parallel processing...")
        
        processed_tasks = executor.execute_tasks(missing_tasks, show_progress=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Final results
        successful_tasks = [t for t in processed_tasks if hasattr(t, 'status') and str(t.status) == 'TaskStatus.COMPLETED']
        failed_tasks = [t for t in processed_tasks if t not in successful_tasks]
        
        logger.info("=" * 80)
        logger.info("📊 FINAL RESULTS")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total processing time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
        logger.info(f"✅ Successful experiments: {len(successful_tasks)}")
        logger.info(f"❌ Failed experiments: {len(failed_tasks)}")
        logger.info(f"📈 Success rate: {len(successful_tasks)/len(missing_tasks)*100:.1f}%")
        
        if processing_time > 0:
            throughput = len(successful_tasks) / processing_time * 60
            logger.info(f"⚡ Throughput: {throughput:.2f} experiments/minute")
        
        # Log failed tasks if any
        if failed_tasks:
            logger.warning("❌ Failed experiments:")
            for task in failed_tasks:
                logger.warning(f"   - {task.celltype} - {task.assay}")
        
        # Calculate disk usage
        try:
            total_size = 0
            for root, dirs, files in os.walk(base_path):
                total_size += sum(os.path.getsize(os.path.join(root, f)) for f in files)
            
            total_size_gb = total_size / (1024**3)
            logger.info(f"💾 Total disk usage: {total_size_gb:.1f} GB")
        except Exception as e:
            logger.warning(f"Could not calculate disk usage: {e}")
        
        logger.info("=" * 80)
        logger.info("🎉 MERGED DATASET PROCESSING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"🕐 End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
