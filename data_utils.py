# data_utils.py
"""
Optimization utilities for CANDIDataHandler.
This module contains all optimization components for efficient data loading.
"""

import os
import sys
import time
import psutil
import logging
import threading
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict
import queue
import heapq
import gc
import weakref

# ========= CORE DATA STRUCTURES =========

class LoadStatus(Enum):
    """Status of a loading operation."""
    PENDING = "pending"
    LOADING = "loading"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

@dataclass
class LoadResult:
    """Result of a loading operation."""
    status: LoadStatus
    data: Optional[Dict[str, np.ndarray]] = None
    metadata: Optional[Dict[str, Any]] = None
    load_time: float = 0.0
    memory_used: float = 0.0
    error: Optional[str] = None

# ========= CHROMOSOME CACHE =========

class ChromosomeCache:
    """LRU cache for chromosome data."""
    
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    def __contains__(self, key):
        return key in self.cache
    
    def __getitem__(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
        else:
            self.misses += 1
            raise KeyError(key)
    
    def __setitem__(self, key, value):
        if key in self.cache:
            # Update existing
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def __len__(self):
        return len(self.cache)
    
    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_memory_usage(self):
        """Estimate memory usage in MB."""
        total_size = 0
        for key, value in self.cache.items():
            if isinstance(value, tuple):
                for item in value:
                    if hasattr(item, 'nbytes'):
                        total_size += item.nbytes
            elif hasattr(value, 'nbytes'):
                total_size += value.nbytes
        return total_size / (1024 * 1024)  # Convert to MB

# ========= CACHE SIZE MANAGER =========

class AdaptiveCacheSizeManager:
    """Manages cache size based on available memory."""
    
    def __init__(self, base_memory_mb=1000):
        self.base_memory_mb = base_memory_mb
        self.current_memory_mb = base_memory_mb
        self.usage_history = []
        
    def update_memory_usage(self, memory_mb):
        """Update memory usage and adjust cache size if needed."""
        self.usage_history.append(memory_mb)
        if len(self.usage_history) > 10:
            self.usage_history.pop(0)
        
        # Calculate average usage
        avg_usage = sum(self.usage_history) / len(self.usage_history)
        
        # Adjust cache size based on usage
        if avg_usage > self.current_memory_mb * 0.9:
            self.current_memory_mb = max(self.base_memory_mb * 0.5, self.current_memory_mb * 0.8)
        elif avg_usage < self.current_memory_mb * 0.5:
            self.current_memory_mb = min(self.base_memory_mb * 2, self.current_memory_mb * 1.2)
    
    def get_max_cache_size(self):
        """Get maximum cache size based on current memory usage."""
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        return min(int(available_memory * 0.3), self.current_memory_mb)

# ========= CACHE PRELOADER =========

class CachePreloader:
    """Preloads chromosome data into cache based on access patterns."""
    
    def __init__(self, cache, size_manager):
        self.cache = cache
        self.size_manager = size_manager
        self.preload_queue = queue.PriorityQueue()
        self.preload_thread = None
        self.stop_preloading = False
        
    def start_preloading(self):
        """Start background preloading thread."""
        if self.preload_thread is None or not self.preload_thread.is_alive():
            self.stop_preloading = False
            self.preload_thread = threading.Thread(target=self._preload_worker, daemon=True)
            self.preload_thread.start()
    
    def stop_preloading(self):
        """Stop background preloading thread."""
        self.stop_preloading = True
        if self.preload_thread and self.preload_thread.is_alive():
            self.preload_thread.join(timeout=1.0)
    
    def _preload_worker(self):
        """Background worker for preloading data."""
        while not self.stop_preloading:
            try:
                priority, chromosome = self.preload_queue.get(timeout=1.0)
                if chromosome not in self.cache:
                    # Preload chromosome data
                    self._preload_chromosome(chromosome)
                self.preload_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in preload worker: {e}")
    
    def _preload_chromosome(self, chromosome):
        """Preload data for a specific chromosome."""
        # This would be implemented based on the actual data loading logic
        pass
    
    def add_to_preload_queue(self, chromosome, priority=1):
        """Add chromosome to preload queue."""
        self.preload_queue.put((priority, chromosome))

# ========= BATCH LOADER =========

class BatchExperimentLoader:
    """Loads multiple experiments in batches for efficiency."""
    
    def __init__(self, batch_size=4):
        self.batch_size = batch_size
        self.load_stats = {}
    
    def load_experiments_batch(self, experiments, load_func):
        """Load a batch of experiments."""
        results = {}
        for i in range(0, len(experiments), self.batch_size):
            batch = experiments[i:i + self.batch_size]
            batch_results = self._load_batch(batch, load_func)
            results.update(batch_results)
        return results
    
    def _load_batch(self, batch, load_func):
        """Load a single batch of experiments."""
        results = {}
        for exp in batch:
            try:
                result = load_func(exp)
                if result is not None:
                    results[exp] = result
            except Exception as e:
                print(f"Error loading experiment {exp}: {e}")
        return results

# ========= MEMORY MAPPED LOADER =========

class MemoryMappedLoader:
    """Loads data using memory mapping for efficient access."""
    
    def __init__(self):
        self.mapped_files = {}
    
    def load_with_mmap(self, file_path):
        """Load file using memory mapping."""
        try:
            if file_path in self.mapped_files:
                return self.mapped_files[file_path]
            
            # Load with memory mapping
            data = np.load(file_path, mmap_mode='r')
            self.mapped_files[file_path] = data
            return data
        except Exception as e:
            print(f"Error loading {file_path} with mmap: {e}")
            return None
    
    def cleanup(self):
        """Clean up mapped files."""
        self.mapped_files.clear()

# ========= PARALLEL CHROMOSOME LOADER =========

class ParallelChromosomeLoader:
    """Loads chromosome data in parallel using multiple workers."""
    
    def __init__(self, max_workers=8):
        self.max_workers = max_workers
        self.load_stats = {}
    
    def load_chromosomes_parallel(self, chromosome_experiments, handler, bios_name, locus, method_type="DSF", **kwargs):
        """Load multiple chromosomes in parallel using real CANDIDataHandler methods."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chromosome loading tasks
            future_to_chromosome = {}
            for chromosome, experiments in chromosome_experiments.items():
                future = executor.submit(
                    self._load_chromosome, 
                    chromosome, experiments, handler, bios_name, locus, method_type, **kwargs
                )
                future_to_chromosome[future] = chromosome
            
            # Collect results
            for future in as_completed(future_to_chromosome):
                chromosome = future_to_chromosome[future]
                try:
                    result = future.result()
                    results[chromosome] = result
                except Exception as e:
                    results[chromosome] = LoadResult(
                        status=LoadStatus.FAILED,
                        error=str(e)
                    )
        
        return results
    
    def _load_chromosome(self, chromosome, experiments, handler, bios_name, locus, method_type="DSF", **kwargs):
        """Load a single chromosome with its experiments using real CANDIDataHandler methods."""
        start_time = time.time()
        
        try:
            # Use the real CANDIDataHandler methods instead of fake data
            if method_type == "DSF":
                result = handler.load_bios_DSF(bios_name, locus, **kwargs)
                if result is not None:
                    data, metadata = result
                else:
                    return LoadResult(
                        status=LoadStatus.FAILED,
                        error="Failed to load DSF data",
                        load_time=time.time() - start_time
                    )
            elif method_type == "BW":
                data = handler.load_bios_BW(bios_name, locus, **kwargs)
                metadata = {}
                if data is None:
                    return LoadResult(
                        status=LoadStatus.FAILED,
                        error="Failed to load BW data",
                        load_time=time.time() - start_time
                    )
            elif method_type == "Peaks":
                data = handler.load_bios_Peaks(bios_name, locus, **kwargs)
                metadata = {}
                if data is None:
                    return LoadResult(
                        status=LoadStatus.FAILED,
                        error="Failed to load Peaks data",
                        load_time=time.time() - start_time
                    )
            else:
                return LoadResult(
                    status=LoadStatus.FAILED,
                    error=f"Unknown method type: {method_type}",
                    load_time=time.time() - start_time
                )
            
            # Validate data integrity
            if not data or len(data) == 0:
                return LoadResult(
                    status=LoadStatus.FAILED,
                    error="Empty data loaded",
                    load_time=time.time() - start_time
                )
            
            # Check data shapes - ensure we have meaningful data
            for exp_name, exp_data in data.items():
                if hasattr(exp_data, 'shape') and exp_data.shape[0] == 0:
                    return LoadResult(
                        status=LoadStatus.FAILED,
                        error=f"Empty array for experiment {exp_name}",
                        load_time=time.time() - start_time
                    )
            
            load_time = time.time() - start_time
            
            return LoadResult(
                status=LoadStatus.COMPLETED,
                data=data,
                metadata=metadata,
                load_time=load_time
            )
            
        except Exception as e:
            return LoadResult(
                status=LoadStatus.FAILED,
                error=str(e),
                load_time=time.time() - start_time
            )

# ========= DDP COMPONENTS =========

class DDPChromosomeSampler:
    """Chromosome-aware sampler for DDP training."""
    
    def __init__(self):
        self.chromosome_weights = {}
        self.rank_assignments = {}
    
    def assign_chromosomes_to_ranks(self, chromosomes, num_ranks):
        """Assign chromosomes to different ranks for balanced loading."""
        assignments = {}
        for i, chromosome in enumerate(chromosomes):
            rank = i % num_ranks
            if rank not in assignments:
                assignments[rank] = []
            assignments[rank].append(chromosome)
        return assignments

class RankChromosomeAssigner:
    """Assigns chromosomes to specific ranks."""
    
    def __init__(self):
        self.rank_chromosomes = {}
    
    def assign_chromosomes(self, rank, chromosomes):
        """Assign chromosomes to a specific rank."""
        self.rank_chromosomes[rank] = chromosomes
    
    def get_chromosomes_for_rank(self, rank):
        """Get chromosomes assigned to a specific rank."""
        return self.rank_chromosomes.get(rank, [])

class ChromosomeBalancer:
    """Balances chromosome loading across ranks."""
    
    def __init__(self):
        self.load_times = {}
        self.balancing_weights = {}
    
    def update_load_time(self, chromosome, load_time):
        """Update load time for a chromosome."""
        self.load_times[chromosome] = load_time
    
    def get_balanced_assignment(self, chromosomes, num_ranks):
        """Get balanced chromosome assignment based on load times."""
        # Sort chromosomes by load time (heaviest first)
        sorted_chromosomes = sorted(
            chromosomes, 
            key=lambda x: self.load_times.get(x, 0), 
            reverse=True
        )
        
        # Assign to ranks in round-robin fashion
        assignments = [[] for _ in range(num_ranks)]
        for i, chromosome in enumerate(sorted_chromosomes):
            rank = i % num_ranks
            assignments[rank].append(chromosome)
        
        return assignments

# ========= MEMORY OPTIMIZATION =========

class ChromosomeMemoryPool:
    """Memory pool for chromosome data."""
    
    def __init__(self, max_pool_size=50):
        self.max_pool_size = max_pool_size
        self.pool = {}
        self.usage_count = {}
    
    def get_memory_block(self, size):
        """Get a memory block from the pool."""
        if size in self.pool and self.pool[size]:
            block = self.pool[size].pop()
            self.usage_count[id(block)] = 1
            return block
        else:
            return np.zeros(size, dtype=np.float32)
    
    def return_memory_block(self, block):
        """Return a memory block to the pool."""
        size = block.size
        if size not in self.pool:
            self.pool[size] = []
        
        if len(self.pool[size]) < self.max_pool_size:
            self.pool[size].append(block)
            if id(block) in self.usage_count:
                del self.usage_count[id(block)]
    
    def clear(self):
        """Clear the memory pool."""
        self.pool.clear()
        self.usage_count.clear()

class LazyChromosomeLoader:
    """Lazy loading for non-active chromosomes."""
    
    def __init__(self):
        self.loaded_chromosomes = {}
        self.loading_queue = queue.Queue()
        self.loading_thread = None
    
    def load_chromosome_lazy(self, chromosome, load_func):
        """Load chromosome data lazily."""
        if chromosome in self.loaded_chromosomes:
            return self.loaded_chromosomes[chromosome]
        
        # Add to loading queue
        self.loading_queue.put((chromosome, load_func))
        return None
    
    def start_lazy_loading(self):
        """Start background lazy loading thread."""
        if self.loading_thread is None or not self.loading_thread.is_alive():
            self.loading_thread = threading.Thread(target=self._lazy_load_worker, daemon=True)
            self.loading_thread.start()
    
    def _lazy_load_worker(self):
        """Background worker for lazy loading."""
        while True:
            try:
                chromosome, load_func = self.loading_queue.get(timeout=1.0)
                data = load_func(chromosome)
                if data is not None:
                    self.loaded_chromosomes[chromosome] = data
                self.loading_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in lazy loading: {e}")

class MemoryCleanupSystem:
    """System for cleaning up memory between operations."""
    
    def __init__(self):
        self.cleanup_callbacks = []
    
    def register_cleanup_callback(self, callback):
        """Register a cleanup callback."""
        self.cleanup_callbacks.append(callback)
    
    def cleanup_all(self):
        """Run all cleanup callbacks."""
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in cleanup callback: {e}")
        
        # Force garbage collection
        gc.collect()

# ========= PERFORMANCE MONITORING =========

class LoadingMetricsCollector:
    """Collects and stores loading performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'loading_times': {},
            'memory_usage': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def record_loading_time(self, chromosome, load_time):
        """Record loading time for a chromosome."""
        if chromosome not in self.metrics['loading_times']:
            self.metrics['loading_times'][chromosome] = []
        self.metrics['loading_times'][chromosome].append(load_time)
    
    def record_memory_usage(self, chromosome, memory_usage):
        """Record memory usage for a chromosome."""
        if chromosome not in self.metrics['memory_usage']:
            self.metrics['memory_usage'][chromosome] = []
        self.metrics['memory_usage'][chromosome].append(memory_usage)
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics['cache_misses'] += 1
    
    def get_all_metrics(self):
        """Get all collected metrics."""
        return self.metrics.copy()
    
    def get_average_loading_time(self, chromosome):
        """Get average loading time for a chromosome."""
        times = self.metrics['loading_times'].get(chromosome, [])
        return sum(times) / len(times) if times else 0.0
    
    def get_average_memory_usage(self, chromosome):
        """Get average memory usage for a chromosome."""
        usage = self.metrics['memory_usage'].get(chromosome, [])
        return sum(usage) / len(usage) if usage else 0.0

class MemoryTracker:
    """Tracks memory usage during operations."""
    
    def __init__(self):
        self.start_memory = 0
        self.peak_memory = 0
        self.current_memory = 0
    
    def start_tracking(self):
        """Start tracking memory usage."""
        self.start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        self.current_memory = self.start_memory
        self.peak_memory = self.start_memory
    
    def update_tracking(self):
        """Update current memory usage."""
        self.current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = max(self.peak_memory, self.current_memory)
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        self.update_tracking()
        return self.current_memory - self.start_memory
    
    def get_peak_memory(self):
        """Get peak memory usage in MB."""
        return self.peak_memory - self.start_memory
    
    def get_current_usage(self):
        """Get current total memory usage in MB."""
        return psutil.Process().memory_info().rss / (1024 * 1024)

class PerformanceComparisonTests:
    """Tests for comparing performance between different loading methods."""
    
    def __init__(self):
        self.test_results = {}
    
    def run_baseline_test(self, load_func, test_data):
        """Run baseline performance test."""
        start_time = time.time()
        result = load_func(test_data)
        load_time = time.time() - start_time
        
        self.test_results['baseline'] = {
            'load_time': load_time,
            'success': result is not None
        }
        return result
    
    def run_optimized_test(self, load_func, test_data):
        """Run optimized performance test."""
        start_time = time.time()
        result = load_func(test_data)
        load_time = time.time() - start_time
        
        self.test_results['optimized'] = {
            'load_time': load_time,
            'success': result is not None
        }
        return result
    
    def get_performance_improvement(self):
        """Get performance improvement percentage."""
        if 'baseline' not in self.test_results or 'optimized' not in self.test_results:
            return 0.0
        
        baseline_time = self.test_results['baseline']['load_time']
        optimized_time = self.test_results['optimized']['load_time']
        
        if baseline_time == 0:
            return 0.0
        
        improvement = (baseline_time - optimized_time) / baseline_time * 100
        return improvement
