from model import EmbedMetadata, ConvTower, DeconvTower, DualAttentionEncoderBlock, NegativeBinomialLayer, GaussianLayer, MONITOR_VALIDATION
from model import CANDI, CANDI_LOSS, CANDI, CANDI_UNET, CANDI_Decoder, CANDI_DNA_Encoder, PeakLayer
from _utils import exponential_linspace_int, negative_binomial_loss, Gaussian, NegativeBinomial, compute_perplexity, DataMasker
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
from data import CANDIDataHandler, CANDIIterableDataset

from torch import nn
import torch
from torchinfo import summary
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import tracemalloc, sys, argparse
from datetime import datetime
import os, random
import multiprocessing
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

import json
from pathlib import Path
import time
from tqdm import tqdm
import warnings
import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

##=========================================== Trainer =====================================================##

class CANDI_TRAINER(object):
    def __init__(self, model, dataset_params, training_params, device=None, rank=None, world_size=None):
        """
        Initialize CANDI trainer with model, dataset configuration, and training parameters.
        
        Args:
            model: CANDI model instance
            dataset_params: Dict with dataset configuration (base_path, resolution, etc.)
            training_params: Dict with training configuration (optimizer, lr, epochs, etc.)
            device: Device to use for training, auto-detected if None
            rank: Process rank for DDP (None for single-GPU)
            world_size: Total number of processes for DDP (None for single-GPU)
        """
        super(CANDI_TRAINER, self).__init__()

        # DDP setup
        self.rank = rank
        self.world_size = world_size
        self.is_ddp = (rank is not None and world_size is not None)
        self.is_main_process = (rank == 0) if self.is_ddp else True

        # Device setup
        if device is None:
            if self.is_ddp:
                # In DDP mode, use local rank as device
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if self.is_main_process:
            print(f"Training on device: {self.device}. DDP: {self.is_ddp}")

        # Model setup
        self.model = model.to(self.device)
        
        # Dataset setup - initialize CANDIIterableDataset from data.py
        self.dataset_params = dataset_params
        try:
            self.dataset = CANDIIterableDataset(**self.dataset_params)
            if self.is_main_process:
                print(f"Successfully initialized dataset with {len(self.dataset.aliases['experiment_aliases'])} assays")
        except Exception as e:
            if self.is_main_process:
                print(f"Warning: Failed to initialize dataset during __init__: {e}")
                print("Dataset will be initialized during train() call")
            self.dataset = None
        
        # Training configuration with defaults
        self.training_params = {
            'optimizer': 'adamax',
            'enable_validation': False,
            'DNA': True,
            "specific_ema_alpha": 0.005,
            'inner_epochs': 1,
            **training_params  # Override defaults with provided params
        }
        
        # Initialize optimizer and scheduler
        self._setup_optimizer_scheduler()
        
        # Initialize criterion
        self.criterion = CANDI_LOSS()
        
        # Mixed precision support
        self.use_mixed_precision = self.training_params.get('use_mixed_precision', True) and self.device.type == 'cuda'
        if self.use_mixed_precision:
            self.scaler = GradScaler('cuda')
            if self.is_main_process:
                print("Mixed precision training enabled")
        else:
            self.scaler = None
            if self.is_main_process:
                print("Mixed precision training disabled")
        
        # Flags
        self.enable_validation = self.training_params.get('enable_validation', False)
        
        # Initialize progress tracking
        self.progress_data = []
        self.batch_counter = 0
        self.progress_dir = training_params.get('progress_dir', './progress')
        self.progress_file = None  # Will be set when first batch is processed
        
        # Create progress directory if it doesn't exist
        if self.is_main_process:
            Path(self.progress_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize checkpoint tracking
        self.checkpoint_dir = None
        self.current_checkpoint_path = None
        self.last_table_lines = 0

    def _setup_optimizer_scheduler(self):
        """Setup optimizer and scheduler based on training parameters."""
        lr = self.training_params['learning_rate']
        
        # Setup optimizer
        if self.training_params['optimizer'].lower() == 'adamax':
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=lr)
        elif self.training_params['optimizer'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif self.training_params['optimizer'].lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        # Scheduler will be created in train() with actual batch counts
        self.scheduler = None
    
    def _setup(self):
        """
        Configure logging, optional validation, and wrap model in DDP if multi-GPU.
        """
        # Setup logging
        if self.is_main_process:
            print(f"Setting up CANDI trainer...")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Training parameters: {self.training_params}")
        
        # Wrap model in DDP if multi-GPU
        if self.is_ddp:
            if self.is_main_process:
                print(f"Wrapping model in DDP for {self.world_size} processes")
            
            # Wrap model with DDP
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                output_device=self.device.index if self.device.type == 'cuda' else None,
                find_unused_parameters=False)

            
        
        # Setup validation if enabled
        if self.enable_validation:
            if self.is_main_process:
                print("Validation is enabled")
            try:
                # Try to initialize MONITOR_VALIDATION from model.py
                from model import MONITOR_VALIDATION
                self.validator = MONITOR_VALIDATION(
                    data_path=self.dataset_params['base_path'],
                    context_length=self.training_params.get('context_length', 1200),
                    batch_size=max(4, self.training_params.get('batch_size', 25) * 4),  # Use larger batch for validation
                    must_have_chr_access=self.dataset_params.get('must_have_chr_access', False),
                    DNA=self.training_params.get('DNA', True),
                    eic=self.dataset_params.get('dataset_type') == 'eic',
                    device=self.device
                )
                if self.is_main_process:
                    print("Validation setup completed successfully")
            except Exception as e:
                if self.is_main_process:
                    print(f"Warning: Failed to setup MONITOR_VALIDATION: {e}")
                    print("Falling back to simplified validation")
                # Use simplified validation as fallback
                self.validator = "simplified"
                if self.is_main_process:
                    print("Simplified validation enabled as fallback")
        else:
            if self.is_main_process:
                print("Validation is disabled")
            self.validator = None
    
    def train(self):
        """
        Main training entry point. Sets up DataLoader and handles epoch loop.
        """        
        # Setup trainer (logging, DDP, validation)
        self._setup()
        
        # Initialize dataset if not already done
        if self.dataset is None:
            try:
                if self.is_main_process:
                    print("Initializing dataset...")
                self.dataset = CANDIIterableDataset(**self.dataset_params)
                if self.is_main_process:
                    print(f"Successfully initialized dataset with {len(self.dataset.aliases['experiment_aliases'])} assays")
            except Exception as e:
                error_msg = f"Failed to initialize dataset: {e}"
                if self.is_main_process:
                    print(f"Error: {error_msg}")
                raise RuntimeError(error_msg) from e
        
        # Calculate estimated batches per epoch for progress tracking
        estimated_batches_per_epoch = self._estimate_batches_per_epoch()
        
        # Setup cosine scheduler with actual batch counts
        if self.scheduler is None:
            self._setup_cosine_scheduler(estimated_batches_per_epoch)
        
        # Create DataLoader with dynamic num_workers based on CPU count
        # Reduce workers per process in DDP to avoid resource contention
        cpu_count = multiprocessing.cpu_count()
        if self.is_ddp:
            num_workers = min(cpu_count // self.world_size, 4)  # Divide workers among processes
        else:
            num_workers = min(cpu_count, 4)  # Cap at 8 for single process
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.training_params['batch_size'],
            num_workers=num_workers
        )
        
        if self.is_main_process:
            print(f"Using {num_workers} workers for data loading (CPU count: {cpu_count}, DDP: {self.is_ddp})")
        
        # Main training loop
        for epoch in range(self.training_params['epochs']):
            # Start epoch
            if self.is_main_process:
                print(f"\nEpoch {epoch+1}/{self.training_params['epochs']}")
            
            # Process batches from CANDIIterableDataset
            batch_count = 0
            for batch_idx, batch in enumerate(dataloader):
                # Validate batch structure
                if not self._validate_batch(batch):
                    if self.is_main_process:
                        print(f"Warning: Skipping invalid batch {batch_idx}")
                    continue
                
                batch_count += 1
                
                # Process the batch
                # try:
                batch_start_time = time.time()
                batch = self._move_batch_to_device(batch)
                
                # Process batch with masking, forward pass, and loss computation
                result_dict = self._process_batch(batch)
                
                if result_dict is None:  # Batch was skipped due to errors
                    continue
                
                # Extract loss and metrics
                loss_keys = ['total_loss', 'obs_count_loss', 'imp_count_loss', 'obs_pval_loss', 'imp_pval_loss', 'obs_peak_loss', 'imp_peak_loss']
                loss_dict = {k: result_dict[k] for k in loss_keys if k in result_dict}
                metrics = {k: v for k, v in result_dict.items() if k not in loss_keys}
                
                # Calculate batch processing time
                batch_processing_time = time.time() - batch_start_time
                    
                # Log batch info for first few batches (only in debug mode)
                if self.is_main_process and batch_idx < 3 and self.training_params.get('debug', False):
                    self._log_batch_info(batch, batch_idx, loss_dict)
                
                # Print batch progress
                if self.is_main_process:
                    try:
                        self._print_batch_log(metrics, loss_dict, batch_idx, epoch, batch_processing_time)
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Warning: Failed to print batch log: {e}")
                            print(f"Batch info: {batch_idx}, {epoch}, {batch_processing_time}")
                            print(f"Metrics: {metrics}")
                            print(f"Loss dict: {loss_dict}")
                    
                # Learning rate scheduling - cosine scheduler steps per batch
                if self.scheduler is not None:
                    # Check if we've exceeded the total steps
                    if hasattr(self, 'total_scheduler_steps') and self.current_scheduler_step >= self.total_scheduler_steps:
                        if self.is_main_process and batch_idx % 1000 == 0:  # Only print occasionally
                            print(f"Warning: Scheduler has completed all {self.total_scheduler_steps} steps. Continuing with final learning rate.")
                    else:
                        try:
                            self.scheduler.step()
                            if hasattr(self, 'current_scheduler_step'):
                                self.current_scheduler_step += 1
                        except (ZeroDivisionError, ValueError) as e:
                            if self.is_main_process:
                                print(f"Warning: Scheduler step failed: {e}. Continuing with current learning rate.")
                            # Continue training without stepping the scheduler
                    
            # Run validation at epoch end if enabled
            validation_summary = None
            if self.enable_validation and self.is_main_process:
                validation_summary, validation_metrics = self._validate()
                
            # End epoch
            if self.is_main_process:
                print(f"Epoch {epoch+1} complete - Processed {batch_count} batches")
                
            # Save progress at end of each epoch
            if self.is_main_process and self.progress_data:
                self._save_progress_to_csv(epoch, batch_count)
            
            # Save checkpoint after each epoch (only if saving is enabled)
            if hasattr(self, 'model_name') and self.model_name and not self.training_params.get('no_save', False):
                self._save_checkpoint(epoch, self.model_name)
                
            # Synchronize processes at epoch end if using DDP
            if self.is_ddp:
                dist.barrier()
        
        # Save final progress data
        if self.is_main_process and self.progress_data:
            self._save_progress_to_csv(epoch, batch_count)
            print(f"Final training progress saved with {len(self.progress_data)} total records")
        
        # Clean up final checkpoint since we'll save the final model
        if self.is_main_process and self.current_checkpoint_path and self.current_checkpoint_path.exists():
            self.current_checkpoint_path.unlink()
            print(f"🗑️  Removed final checkpoint: {self.current_checkpoint_path.name}")
        
        return self.model
    
    def _process_batch(self, batch):
        """
        Process a single batch: apply masking, forward pass, compute loss, and backward pass.
        
        Args:
            batch: Dictionary containing batch data from CANDIIterableDataset
            
        Returns:
            dict: Dictionary containing loss values, or None if batch should be skipped
        """
        # Extract data from batch and convert to correct types
        x_data = batch['x_data'].float()      # [B, L, F] - input signal data (convert to float)
        x_meta = batch['x_meta'].float()      # [B, 4, F] - input metadata (convert to float)
        x_avail = batch['x_avail']            # [B, F] - input availability
        x_dna = batch['x_dna'].float()        # [B, L*25, 4] - input DNA sequence (convert to float)
        
        y_data = batch['y_data'].float()      # [B, L, F] - target signal data (convert to float)
        y_meta = batch['y_meta'].float()      # [B, 4, F] - target metadata (convert to float)
        y_avail = batch['y_avail']            # [B, F] - target availability  
        y_pval = batch['y_pval'].float()      # [B, L, F] - target p-values (convert to float)
        y_peaks = batch['y_peaks'].float()    # [B, L, F] - target peak data (convert to float)
        y_dna = batch['y_dna'].float()        # [B, L*25, 4] - target DNA sequence (convert to float)

        control_data = batch['control_data'].float()   # [B, L, 1] - control signal data (convert to float)
        control_meta = batch['control_meta'].float()   # [B, 4, 1] - control metadata (convert to float)
        control_avail = batch['control_avail']         # [B, 1] - control availability
        
        # Apply masking to create imputation targets
        # Use the masker to create masked inputs
        if not hasattr(self, 'masker'):
            # Initialize masker if not already done
            token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}
            self.masker = DataMasker(token_dict["cloze_mask"])
        
        # Apply masking - this modifies x_data, x_meta, x_avail in place
        # The masker expects inputs in a specific format, so we need to handle batch dimension
        B, L, F = x_data.shape
        
        # Clone inputs for masking
        x_data_masked = x_data.clone()
        x_meta_masked = x_meta.clone() 
        x_avail_masked = x_avail.clone()
        
        # Get signal_dim from dataset_params or calculate from aliases
        if hasattr(self.dataset, 'signal_dim'):
            signal_dim = self.dataset.signal_dim 
        elif hasattr(self.dataset, 'aliases'):
            signal_dim = len(self.dataset.aliases['experiment_aliases'])
        else:
            # Fallback - get from dataset_params
            signal_dim = self.dataset_params.get('signal_dim', 35)  # Default to 35 for EIC
        
        # Determine how many features to mask based on availability
        num_available_per_sample = x_avail.sum(dim=1)  # [B] - number of available features per sample
        min_available = num_available_per_sample.min().item()
        
        # Let DataMasker handle the case where there are very few features
        # It will mask 0 features if there's only 1 available, which is fine
        num_mask = min(random.randint(1, max(1, signal_dim - 1)), signal_dim - 1)
        x_data_masked, x_meta_masked, x_avail_masked = self.masker.mask_assays(x_data_masked, x_meta_masked, x_avail_masked, num_mask)

        # Create masks for loss computation
        token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}
        masked_map = (x_data_masked == token_dict["cloze_mask"])  # Imputation targets
        observed_map = (x_data_masked != token_dict["missing_mask"]) & (x_data_masked != token_dict["cloze_mask"])  # Upsampling targets

        observed_map = observed_map.clone()
        masked_map = masked_map.clone()

        # Store num_mask for logging
        self.last_num_mask = num_mask

        x_data_masked = torch.cat([x_data_masked, control_data], dim=2)      # (B, L, F+1)
        x_meta_masked = torch.cat([x_meta_masked, control_meta], dim=2)      # (B, 4, F+1)
        x_avail_masked = torch.cat([x_avail_masked, control_avail], dim=1)   # (B, F+1)
        
        # Validate that we have observed regions (we need at least some data to train on)
        if not observed_map.any():
            if self.is_main_process:
                print("Warning: No observed regions found! Skipping batch...")
            return None
        
        # If no regions were masked, we can still do training (just no imputation loss)
        # This is fine - the model can learn from reconstruction loss on observed data
        has_masked_regions = masked_map.any()
        
        # Move masks to device
        masked_map = masked_map.to(self.device)
        observed_map = observed_map.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        try:
            # Forward pass through model with mixed precision
            if self.use_mixed_precision:
                with autocast('cuda'):
                    if self.training_params.get('DNA', False):
                        # Model expects DNA sequence
                        output_p, output_n, output_mu, output_var, output_peak = self.model(
                            x_data_masked, x_dna, x_meta_masked, y_meta
                        )
                    else:
                        raise ValueError("DNA must be True for CANDI_TRAINER")
            else:
                if self.training_params.get('DNA', False):
                    # Model expects DNA sequence
                    output_p, output_n, output_mu, output_var, output_peak = self.model(
                        x_data_masked, x_dna, x_meta_masked, y_meta
                    )
                else:
                    raise ValueError("DNA must be True for CANDI_TRAINER")
                    
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if self.is_main_process:
                    print(f"Warning: CUDA Out of Memory! Batch size: {B}, Sequence length: {L}, Features: {F}. Skipping batch and clearing cache...")
                torch.cuda.empty_cache()
                return None
            else:
                raise e
        
        # Validate model outputs before loss computation
        if torch.isnan(output_p).any() or torch.isnan(output_n).any() or \
           torch.isnan(output_mu).any() or torch.isnan(output_var).any() or \
           torch.isnan(output_peak).any():
            if self.is_main_process:
                print("Warning: NaN in model outputs! Skipping batch...")
            return None
        
        try:
            # Compute losses using CANDI_LOSS with mixed precision
            if self.use_mixed_precision:
                with autocast('cuda'):
                    if has_masked_regions:
                        # Normal case: compute all losses
                        obs_count_loss, imp_count_loss, obs_pval_loss, imp_pval_loss, obs_peak_loss, imp_peak_loss = self.criterion(
                            output_p, output_n, output_mu, output_var, output_peak,
                            y_data, y_pval, y_peaks, observed_map, masked_map
                        )
                        total_loss = obs_count_loss + obs_pval_loss + imp_pval_loss + imp_count_loss + obs_peak_loss + imp_peak_loss
                    else:
                        # No masked regions: only compute observed losses
                        obs_count_loss, _, obs_pval_loss, _, obs_peak_loss, _ = self.criterion(
                            output_p, output_n, output_mu, output_var, output_peak,
                            y_data, y_pval, y_peaks, observed_map, observed_map  # Use observed_map for both
                        )
                        imp_count_loss = torch.tensor(0.0, device=self.device)
                        imp_pval_loss = torch.tensor(0.0, device=self.device)
                        imp_peak_loss = torch.tensor(0.0, device=self.device)
                        total_loss = obs_count_loss + obs_pval_loss + obs_peak_loss
            else:
                if has_masked_regions:
                    # Normal case: compute all losses
                    obs_count_loss, imp_count_loss, obs_pval_loss, imp_pval_loss, obs_peak_loss, imp_peak_loss = self.criterion(
                        output_p, output_n, output_mu, output_var, output_peak,
                        y_data, y_pval, y_peaks, observed_map, masked_map
                    )
                    total_loss = obs_count_loss + obs_pval_loss + imp_pval_loss + imp_count_loss + obs_peak_loss + imp_peak_loss
                else:
                    # No masked regions: only compute observed losses
                    obs_count_loss, _, obs_pval_loss, _, obs_peak_loss, _ = self.criterion(
                        output_p, output_n, output_mu, output_var, output_peak,
                        y_data, y_pval, y_peaks, observed_map, observed_map  # Use observed_map for both
                    )
                    imp_count_loss = torch.tensor(0.0, device=self.device)
                    imp_pval_loss = torch.tensor(0.0, device=self.device)
                    imp_peak_loss = torch.tensor(0.0, device=self.device)
                    total_loss = obs_count_loss + obs_pval_loss + obs_peak_loss
            
            # Check for NaN losses with detailed debugging
            if torch.isnan(total_loss).sum() > 0:
                if self.is_main_process:
                    print("Warning: Encountered NaN loss! Skipping batch...")
                return None
            
            # Backward pass with mixed precision
            if self.use_mixed_precision:
                # Scale the loss and backward pass
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard backward pass
                total_loss = total_loss.float()
                total_loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                
                # Optimizer step
                self.optimizer.step()
            
            # Store gradient norm for logging
            self.grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if self.is_main_process:
                    print("Warning: CUDA Out of Memory during loss computation! Skipping batch and clearing cache...")
                torch.cuda.empty_cache()
                return None
            else:
                raise e
        
        # Compute metrics for monitoring
        if has_masked_regions:
            metrics = self._compute_metrics(
                output_p, output_n, output_mu, output_var, output_peak,
                y_data, y_pval, y_peaks, observed_map, masked_map
            )
        else:
            # Only compute observed metrics when no masking occurred
            metrics = self._compute_metrics(
                output_p, output_n, output_mu, output_var, output_peak,
                y_data, y_pval, y_peaks, observed_map, torch.zeros_like(observed_map)  # Empty mask for imputation metrics
            )
        
        # Return loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'obs_count_loss': obs_count_loss.item(),
            'imp_count_loss': imp_count_loss.item(), 
            'obs_pval_loss': obs_pval_loss.item(),
            'imp_pval_loss': imp_pval_loss.item(),
            'obs_peak_loss': obs_peak_loss.item(),
            'imp_peak_loss': imp_peak_loss.item()
        }
        
        # Update progress monitoring and check for LR adjustment
        self._update_progress_monitoring(metrics, loss_dict, self.training_params.get('specific_ema_alpha', 0.005))
        
        # Add metrics to return dictionary
        return_dict = {**loss_dict, **metrics}
        
        return return_dict
    
    def _estimate_batches_per_epoch(self):
        """
        Estimate the number of batches per epoch based on dataset parameters.
        
        The total number of samples per epoch is:
        num_biosamples × num_loci × num_dsf_factors × num_chromosomes
        
        Then divided by batch_size to get number of batches.
        
        Returns:
            int: Estimated number of batches per epoch, or None if cannot estimate
        """
        try:
            # Create a temporary dataset to get the actual counts
            temp_dataset = CANDIIterableDataset(**self.dataset_params)
            
            # Setup the data looper to get actual counts
            temp_dataset.setup_datalooper(
                m=self.dataset_params.get('m', 1000),
                context_length=self.dataset_params.get('context_length', 30000),
                bios_batchsize=1,  # CANDIIterableDataset always uses 1
                loci_batchsize=1,  # CANDIIterableDataset always uses 1
                loci_gen_strategy=self.dataset_params.get('loci_gen_strategy', 'random'),
                split=self.dataset_params.get('split', 'train'),
                shuffle_bios=self.dataset_params.get('shuffle_bios', True),
                dsf_list=self.dataset_params.get('dsf_list', [1, 2]),
                includes=self.dataset_params.get('includes'),
                excludes=self.dataset_params.get('excludes', []),
                must_have_chr_access=self.dataset_params.get('must_have_chr_access', False), 
                bios_min_exp_avail_threshold=self.dataset_params.get('bios_min_exp_avail_threshold', 0)
            )
            
            # Get the actual counts after setup
            num_biosamples = len(temp_dataset.navigation)
            num_loci = temp_dataset.num_regions  # This is the number of loci (m_regions)
            num_dsf_factors = len(temp_dataset.dsf_list)
            num_chromosomes = len(temp_dataset.loci.keys()) if hasattr(temp_dataset, 'loci') else 1
            
            # Account for DDP and DataLoader workers sharding
            total_samples = num_biosamples * num_loci * num_dsf_factors
            
            if self.is_ddp:
                # In DDP, samples are divided among processes
                world_size = self.world_size
                # Also account for DataLoader workers
                cpu_count = multiprocessing.cpu_count()
                num_workers = min(cpu_count // world_size, 4) if self.is_ddp else min(cpu_count, 4)
                total_consumers = num_workers * world_size
                total_samples = total_samples // total_consumers
            
            # Calculate batches
            batch_size = self.training_params.get('batch_size', 25)
            estimated_batches = max(1, total_samples // batch_size)

            self.estimated_batches_per_epoch = estimated_batches
            
            if self.is_main_process:
                print(f"Estimated batches per epoch: {estimated_batches} "
                      f"(samples: {total_samples}, batch_size: {batch_size})")
                print(f"Dataset composition: {num_biosamples} biosamples × "
                      f"{num_loci} loci × {num_dsf_factors} DSF factors = "
                      f"{num_biosamples * num_loci * num_dsf_factors} total samples")
            
            return estimated_batches
            
        except Exception as e:
            if self.is_main_process:
                print(f"Warning: Could not estimate batches per epoch: {e}")
            return None
    
    def _setup_cosine_scheduler(self, batches_per_epoch):
        """Setup cosine scheduler with actual batch counts."""
        epochs = self.training_params['epochs']
        inner_epochs = self.training_params['inner_epochs']
        
        # Calculate actual total steps
        num_total_steps = epochs * inner_epochs * batches_per_epoch
        warmup_steps = inner_epochs * batches_per_epoch
        
        # Ensure we have at least 1 step for the cosine annealing phase
        cosine_steps = max(1, num_total_steps - warmup_steps)
        
        if self.is_main_process:
            print(f"Setting up cosine scheduler: {num_total_steps} total steps, {warmup_steps} warmup steps, {cosine_steps} cosine steps")
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(self.optimizer, T_max=cosine_steps, eta_min=0.0)
            ],
            milestones=[warmup_steps]
        )
        
        # Store the total steps for tracking
        self.total_scheduler_steps = num_total_steps
        self.current_scheduler_step = 0
    
    def _validate(self):
        """
        Run validation if enabled, computing metrics on validation set.
        
        Returns:
            tuple: (validation_summary_string, validation_metrics_dict) or (None, None) if validation fails
        """
        if not self.enable_validation or self.validator is None:
            return None, None
            
        if not self.is_main_process:
            return None, None  # Only run validation on main process
            
        try:
            print("Running validation...")
                
            # Set model to evaluation mode
            self.model.eval()
            
            with torch.no_grad():
                if self.validator == "simplified":
                    # Simplified validation fallback
                    validation_summary = self._run_simplified_validation()
                    validation_metrics = {"validation_type": "simplified", "status": "completed"}
                else:
                    # Use the full MONITOR_VALIDATION system
                    validation_summary, validation_metrics = self.validator.get_validation(self.model)
                
            # Set model back to training mode
            self.model.train()
            
            if validation_summary and validation_metrics:
                print("Validation completed successfully")
                return validation_summary, validation_metrics
            else:
                print("Warning: Validation returned empty results")
                return None, None
                
        except Exception as e:
            print(f"Error: Validation failed: {e}. Continuing training without validation...")
            # Set model back to training mode in case of error
            self.model.train()
            return None, None
    
    def _compute_metrics(self, output_p, output_n, output_mu, output_var, output_peak, y_data, y_pval, y_peaks, observed_map, masked_map):
        """
        Compute metrics per feature for both observed and imputed predictions.
        
        Args:
            output_p, output_n: Model outputs for negative binomial parameters [B, L, F]
            output_mu, output_var: Model outputs for Gaussian parameters [B, L, F]
            output_peak: Model outputs for peak predictions [B, L, F]
            y_data: Target count data [B, L, F]
            y_pval: Target p-value data [B, L, F]
            y_peaks: Target peak data [B, L, F]
            observed_map: Boolean mask for observed (upsampling) targets [B, L, F]
            masked_map: Boolean mask for masked (imputation) targets [B, L, F]
            
        Returns:
            dict: Dictionary containing per-feature metrics
        """
        metrics = {}
        B, L, F = y_data.shape  # Batch size, sequence length, num features
        
        # === IMPUTED (MASKED) METRICS PER FEATURE ===
        if masked_map.any():
            imp_count_r2_per_feature = []
            imp_count_spearman_per_feature = []
            imp_count_pearson_per_feature = []
            imp_count_mse_per_feature = []
            
            imp_pval_r2_per_feature = []
            imp_pval_spearman_per_feature = []
            imp_pval_pearson_per_feature = []
            imp_pval_mse_per_feature = []
            imp_count_perplexity_per_feature = []
            imp_pval_perplexity_per_feature = []
            
            imp_peak_auc_per_feature = []
            
            # Compute metrics per feature (F dimension) and per sample (B dimension)
            for f in range(F):
                # Collect all masked data points for this feature across all samples
                all_imp_count_pred = []
                all_imp_count_true = []

                all_imp_pval_pred = []
                all_imp_pval_true = []

                all_imp_count_var = []
                all_imp_pval_var = []
                
                all_imp_peak_pred = []
                all_imp_peak_true = []
                
                # Iterate over each sample in the batch
                for b in range(B):
                    # Get masked positions for this sample and feature
                    sample_masked_map = masked_map[b, :, f]
                    
                    if sample_masked_map.any():
                        # Count predictions for this sample and feature
                        sample_output_p = output_p[b, :, f][sample_masked_map]
                        sample_output_n = output_n[b, :, f][sample_masked_map]
                        neg_bin_imp = NegativeBinomial(sample_output_p.cpu().detach(), sample_output_n.cpu().detach())
                        sample_count_pred = neg_bin_imp.expect().numpy()
                        sample_count_true = y_data[b, :, f][sample_masked_map].cpu().detach().numpy()
                        
                        # P-value predictions for this sample and feature
                        sample_pval_pred = output_mu[b, :, f][sample_masked_map].cpu().detach().numpy()
                        sample_pval_true = y_pval[b, :, f][sample_masked_map].cpu().detach().numpy()
                        sample_pval_var = output_var[b, :, f][sample_masked_map].cpu().detach().numpy()
                        
                        # Peak predictions for this sample and feature
                        sample_peak_pred = output_peak[b, :, f][sample_masked_map].cpu().detach().numpy()
                        sample_peak_true = y_peaks[b, :, f][sample_masked_map].cpu().detach().numpy()
                        
                        # Collect data points
                        all_imp_count_pred.extend(sample_count_pred)
                        all_imp_count_true.extend(sample_count_true)
                        all_imp_pval_pred.extend(sample_pval_pred)
                        all_imp_pval_true.extend(sample_pval_true)
                        all_imp_pval_var.extend(sample_pval_var)
                        all_imp_peak_pred.extend(sample_peak_pred)
                        all_imp_peak_true.extend(sample_peak_true)
                
                # Compute metrics if we have enough data points across all samples
                if len(all_imp_count_true) > 1:
                    all_imp_count_pred = np.array(all_imp_count_pred)
                    all_imp_count_true = np.array(all_imp_count_true)
                    
                    # Check for zero variance to avoid division by zero in r2_score
                    if np.var(all_imp_count_true) > 1e-10:  # Avoid division by zero
                        r2_val = r2_score(all_imp_count_true, all_imp_count_pred)
                        imp_count_r2_per_feature.append(r2_val)
                    else:
                        # If variance is zero, r2 is undefined, use a default value
                        imp_count_r2_per_feature.append(0.0)
                    
                    mse_val = ((all_imp_count_true - all_imp_count_pred)**2).mean()
                    imp_count_mse_per_feature.append(mse_val)
                    
                    # Spearman correlation
                    spearman_result = spearmanr(all_imp_count_true, all_imp_count_pred)
                    if not np.isnan(spearman_result.correlation):
                        imp_count_spearman_per_feature.append(spearman_result.correlation)
                    
                    # Pearson correlation
                    pearson_result = pearsonr(all_imp_count_true, all_imp_count_pred)
                    if not np.isnan(pearson_result[0]):
                        imp_count_pearson_per_feature.append(pearson_result[0])

                    # Compute perplexity for count predictions
                    # Reconstruct NegativeBinomial for perplexity calculation
                    # We need to collect the parameters again for perplexity
                    all_output_p = []
                    all_output_n = []
                    for b in range(B):
                        sample_masked_map = masked_map[b, :, f]
                        if sample_masked_map.any():
                            all_output_p.extend(output_p[b, :, f][sample_masked_map].cpu().detach().numpy())
                            all_output_n.extend(output_n[b, :, f][sample_masked_map].cpu().detach().numpy())
                    
                    if len(all_output_p) > 0:
                        neg_bin_imp = NegativeBinomial(torch.tensor(all_output_p), torch.tensor(all_output_n))
                        neg_bin_probs = neg_bin_imp.pmf(all_imp_count_true.astype(int))
                        perplexity = compute_perplexity(neg_bin_probs)
                        imp_count_perplexity_per_feature.append(perplexity.item())
                
                if len(all_imp_pval_true) > 1:
                    all_imp_pval_pred = np.array(all_imp_pval_pred)
                    all_imp_pval_true = np.array(all_imp_pval_true)
                    
                    # Check for zero variance to avoid division by zero in r2_score
                    if np.var(all_imp_pval_true) > 1e-10:  # Avoid division by zero
                        imp_pval_r2_per_feature.append(r2_score(all_imp_pval_true, all_imp_pval_pred))
                    else:
                        # If variance is zero, r2 is undefined, use a default value
                        imp_pval_r2_per_feature.append(0.0)
                    imp_pval_mse_per_feature.append(((all_imp_pval_true - all_imp_pval_pred)**2).mean())
                    
                    # Spearman correlation
                    spearman_result = spearmanr(all_imp_pval_true, all_imp_pval_pred)
                    if not np.isnan(spearman_result.correlation):
                        imp_pval_spearman_per_feature.append(spearman_result.correlation)
                    
                    # Pearson correlation
                    pearson_result = pearsonr(all_imp_pval_true, all_imp_pval_pred)
                    if not np.isnan(pearson_result[0]):
                        imp_pval_pearson_per_feature.append(pearson_result[0])
                    
                    # Compute perplexity for p-value predictions (using Gaussian likelihood)
                    # Use the custom Gaussian class from _utils.py
                    all_imp_pval_var = np.array(all_imp_pval_var)
                    gaussian_imp = Gaussian(all_imp_pval_pred, all_imp_pval_var)
                    gaussian_probs = gaussian_imp.pdf(all_imp_pval_true)
                    perplexity = compute_perplexity(gaussian_probs)
                    imp_pval_perplexity_per_feature.append(perplexity.item())
                
                # Compute AUC-ROC for peak predictions
                if len(all_imp_peak_true) > 1:
                    all_imp_peak_pred = np.array(all_imp_peak_pred)
                    all_imp_peak_true = np.array(all_imp_peak_true)
                    
                    # Check if we have both positive and negative samples
                    if len(np.unique(all_imp_peak_true)) > 1:
                        try:
                            auc_score = roc_auc_score(all_imp_peak_true, all_imp_peak_pred)
                            imp_peak_auc_per_feature.append(auc_score)
                        except ValueError:
                            # Handle edge cases where AUC cannot be computed
                            pass
            
            # Aggregate imputed metrics: median only
            if imp_count_r2_per_feature:
                imp_count_r2_arr = np.array(imp_count_r2_per_feature)
                metrics.update({
                    'imp_count_r2_median': np.median(imp_count_r2_arr)
                })
                
            if imp_count_spearman_per_feature:
                imp_count_spearman_arr = np.array(imp_count_spearman_per_feature)
                metrics.update({
                    'imp_count_spearman_median': np.median(imp_count_spearman_arr)
                })
                
            if imp_count_pearson_per_feature:
                imp_count_pearson_arr = np.array(imp_count_pearson_per_feature)
                metrics.update({
                    'imp_count_pearson_median': np.median(imp_count_pearson_arr)
                })
                
            if imp_pval_r2_per_feature:
                imp_pval_r2_arr = np.array(imp_pval_r2_per_feature)
                metrics.update({
                    'imp_pval_r2_median': np.median(imp_pval_r2_arr)
                })
                
            if imp_pval_spearman_per_feature:
                imp_pval_spearman_arr = np.array(imp_pval_spearman_per_feature)
                metrics.update({
                    'imp_pval_spearman_median': np.median(imp_pval_spearman_arr)
                })
                
            if imp_pval_pearson_per_feature:
                imp_pval_pearson_arr = np.array(imp_pval_pearson_per_feature)
                metrics.update({
                    'imp_pval_pearson_median': np.median(imp_pval_pearson_arr)
                })
            
            # Aggregate perplexity metrics for imputation
            if imp_count_perplexity_per_feature:
                imp_count_perplexity_arr = np.array(imp_count_perplexity_per_feature)
                metrics.update({
                    'imp_count_perplexity_median': np.median(imp_count_perplexity_arr)
                })
            
            if imp_pval_perplexity_per_feature:
                imp_pval_perplexity_arr = np.array(imp_pval_perplexity_per_feature)
                metrics.update({
                    'imp_pval_perplexity_median': np.median(imp_pval_perplexity_arr)
                })
            
            # Aggregate MSE metrics for imputation
            if imp_count_mse_per_feature:
                imp_count_mse_arr = np.array(imp_count_mse_per_feature)
                metrics.update({
                    'imp_count_mse_median': np.median(imp_count_mse_arr)
                })
            
            if imp_pval_mse_per_feature:
                imp_pval_mse_arr = np.array(imp_pval_mse_per_feature)
                metrics.update({
                    'imp_pval_mse_median': np.median(imp_pval_mse_arr)
                })
            
            # Aggregate AUC metrics for imputation
            if imp_peak_auc_per_feature:
                imp_peak_auc_arr = np.array(imp_peak_auc_per_feature)
                metrics.update({
                    'imp_peak_auc_median': np.median(imp_peak_auc_arr)
                })
            
        # === OBSERVED (UPSAMPLING) METRICS PER FEATURE ===
        if observed_map.any():
            obs_count_r2_per_feature = []
            obs_count_spearman_per_feature = []
            obs_count_pearson_per_feature = []
            obs_count_mse_per_feature = []
            
            obs_pval_r2_per_feature = []
            obs_pval_spearman_per_feature = []
            obs_pval_pearson_per_feature = []
            obs_pval_mse_per_feature = []
            obs_count_perplexity_per_feature = []
            obs_pval_perplexity_per_feature = []
            
            obs_peak_auc_per_feature = []
            
            # Compute metrics per feature (F dimension) and per sample (B dimension)
            for f in range(F):
                # Collect all observed data points for this feature across all samples
                all_obs_count_pred = []
                all_obs_count_true = []
                all_obs_pval_pred = []
                all_obs_pval_true = []
                all_obs_pval_var = []
                
                all_obs_peak_pred = []
                all_obs_peak_true = []
                
                # Iterate over each sample in the batch
                for b in range(B):
                    # Get observed positions for this sample and feature
                    sample_observed_map = observed_map[b, :, f]
                    
                    if sample_observed_map.any():
                        # Count predictions for this sample and feature
                        sample_output_p = output_p[b, :, f][sample_observed_map]
                        sample_output_n = output_n[b, :, f][sample_observed_map]
                        neg_bin_obs = NegativeBinomial(sample_output_p.cpu().detach(), sample_output_n.cpu().detach())
                        sample_count_pred = neg_bin_obs.expect().numpy()
                        sample_count_true = y_data[b, :, f][sample_observed_map].cpu().detach().numpy()
                        
                        # P-value predictions for this sample and feature
                        sample_pval_pred = output_mu[b, :, f][sample_observed_map].cpu().detach().numpy()
                        sample_pval_true = y_pval[b, :, f][sample_observed_map].cpu().detach().numpy()
                        sample_pval_var = output_var[b, :, f][sample_observed_map].cpu().detach().numpy()
                        
                        # Peak predictions for this sample and feature
                        sample_peak_pred = output_peak[b, :, f][sample_observed_map].cpu().detach().numpy()
                        sample_peak_true = y_peaks[b, :, f][sample_observed_map].cpu().detach().numpy()
                        
                        # Collect data points
                        all_obs_count_pred.extend(sample_count_pred)
                        all_obs_count_true.extend(sample_count_true)
                        all_obs_pval_pred.extend(sample_pval_pred)
                        all_obs_pval_true.extend(sample_pval_true)
                        all_obs_pval_var.extend(sample_pval_var)
                        all_obs_peak_pred.extend(sample_peak_pred)
                        all_obs_peak_true.extend(sample_peak_true)
                
                # Compute metrics if we have enough data points across all samples
                if len(all_obs_count_true) > 1:
                    all_obs_count_pred = np.array(all_obs_count_pred)
                    all_obs_count_true = np.array(all_obs_count_true)
                    
                    # Check for zero variance to avoid division by zero in r2_score
                    if np.var(all_obs_count_true) > 1e-10:  # Avoid division by zero
                        obs_count_r2_per_feature.append(r2_score(all_obs_count_true, all_obs_count_pred))
                    else:
                        # If variance is zero, r2 is undefined, use a default value
                        obs_count_r2_per_feature.append(0.0)
                    obs_count_mse_per_feature.append(((all_obs_count_true - all_obs_count_pred)**2).mean())
                    
                    # Spearman correlation
                    spearman_result = spearmanr(all_obs_count_true, all_obs_count_pred)
                    if not np.isnan(spearman_result.correlation):
                        obs_count_spearman_per_feature.append(spearman_result.correlation)
                    
                    # Pearson correlation
                    pearson_result = pearsonr(all_obs_count_true, all_obs_count_pred)
                    if not np.isnan(pearson_result[0]):
                        obs_count_pearson_per_feature.append(pearson_result[0])
                    
                    # Compute perplexity for observed count predictions
                    # Reconstruct NegativeBinomial for perplexity calculation
                    # We need to collect the parameters again for perplexity
                    all_output_p = []
                    all_output_n = []
                    for b in range(B):
                        sample_observed_map = observed_map[b, :, f]
                        if sample_observed_map.any():
                            all_output_p.extend(output_p[b, :, f][sample_observed_map].cpu().detach().numpy())
                            all_output_n.extend(output_n[b, :, f][sample_observed_map].cpu().detach().numpy())
                    
                    if len(all_output_p) > 0:
                        neg_bin_obs = NegativeBinomial(torch.tensor(all_output_p), torch.tensor(all_output_n))
                        neg_bin_probs = neg_bin_obs.pmf(all_obs_count_true.astype(int))
                        perplexity = compute_perplexity(neg_bin_probs)
                        obs_count_perplexity_per_feature.append(perplexity.item())
                
                if len(all_obs_pval_true) > 1:
                    all_obs_pval_pred = np.array(all_obs_pval_pred)
                    all_obs_pval_true = np.array(all_obs_pval_true)
                    
                    # Check for zero variance to avoid division by zero in r2_score
                    if np.var(all_obs_pval_true) > 1e-10:  # Avoid division by zero
                        obs_pval_r2_per_feature.append(r2_score(all_obs_pval_true, all_obs_pval_pred))
                    else:
                        # If variance is zero, r2 is undefined, use a default value
                        obs_pval_r2_per_feature.append(0.0)
                    obs_pval_mse_per_feature.append(((all_obs_pval_true - all_obs_pval_pred)**2).mean())
                    
                    # Spearman correlation
                    spearman_result = spearmanr(all_obs_pval_true, all_obs_pval_pred)
                    if not np.isnan(spearman_result.correlation):
                        obs_pval_spearman_per_feature.append(spearman_result.correlation)
                    
                    # Pearson correlation
                    pearson_result = pearsonr(all_obs_pval_true, all_obs_pval_pred)
                    if not np.isnan(pearson_result[0]):
                        obs_pval_pearson_per_feature.append(pearson_result[0])
                    
                    # Compute perplexity for observed p-value predictions (using Gaussian likelihood)
                    # Use the custom Gaussian class from _utils.py
                    all_obs_pval_var = np.array(all_obs_pval_var)
                    gaussian_obs = Gaussian(all_obs_pval_pred, all_obs_pval_var)
                    gaussian_probs = gaussian_obs.pdf(all_obs_pval_true)
                    perplexity = compute_perplexity(gaussian_probs)
                    obs_pval_perplexity_per_feature.append(perplexity.item())
                
                # Compute AUC-ROC for observed peak predictions
                if len(all_obs_peak_true) > 1:
                    all_obs_peak_pred = np.array(all_obs_peak_pred)
                    all_obs_peak_true = np.array(all_obs_peak_true)
                    
                    # Check if we have both positive and negative samples
                    if len(np.unique(all_obs_peak_true)) > 1:
                        try:
                            auc_score = roc_auc_score(all_obs_peak_true, all_obs_peak_pred)
                            obs_peak_auc_per_feature.append(auc_score)
                        except ValueError:
                            # Handle edge cases where AUC cannot be computed
                            pass
            
            # Aggregate observed metrics: median only
            if obs_count_r2_per_feature:
                obs_count_r2_arr = np.array(obs_count_r2_per_feature)
                metrics.update({
                    'obs_count_r2_median': np.median(obs_count_r2_arr)
                })
                
            if obs_count_spearman_per_feature:
                obs_count_spearman_arr = np.array(obs_count_spearman_per_feature)
                metrics.update({
                    'obs_count_spearman_median': np.median(obs_count_spearman_arr)
                })
                
            if obs_count_pearson_per_feature:
                obs_count_pearson_arr = np.array(obs_count_pearson_per_feature)
                metrics.update({
                    'obs_count_pearson_median': np.median(obs_count_pearson_arr)
                })
                
            if obs_pval_r2_per_feature:
                obs_pval_r2_arr = np.array(obs_pval_r2_per_feature)
                metrics.update({
                    'obs_pval_r2_median': np.median(obs_pval_r2_arr)
                })
                
            if obs_pval_spearman_per_feature:
                obs_pval_spearman_arr = np.array(obs_pval_spearman_per_feature)
                metrics.update({
                    'obs_pval_spearman_median': np.median(obs_pval_spearman_arr)
                })
                
            if obs_pval_pearson_per_feature:
                obs_pval_pearson_arr = np.array(obs_pval_pearson_per_feature)
                metrics.update({
                    'obs_pval_pearson_median': np.median(obs_pval_pearson_arr)
                })
            
            # Aggregate perplexity metrics for observed (upsampling)
            if obs_count_perplexity_per_feature:
                obs_count_perplexity_arr = np.array(obs_count_perplexity_per_feature)
                metrics.update({
                    'obs_count_perplexity_median': np.median(obs_count_perplexity_arr)
                })
            
            if obs_pval_perplexity_per_feature:
                obs_pval_perplexity_arr = np.array(obs_pval_perplexity_per_feature)
                metrics.update({
                    'obs_pval_perplexity_median': np.median(obs_pval_perplexity_arr)
                })
            
            # Aggregate MSE metrics for observed (upsampling)
            if obs_count_mse_per_feature:
                obs_count_mse_arr = np.array(obs_count_mse_per_feature)
                metrics.update({
                    'obs_count_mse_median': np.median(obs_count_mse_arr)
                })
            
            if obs_pval_mse_per_feature:
                obs_pval_mse_arr = np.array(obs_pval_mse_per_feature)
                metrics.update({
                    'obs_pval_mse_median': np.median(obs_pval_mse_arr)
                })
            
            # Aggregate AUC metrics for observed (upsampling)
            if obs_peak_auc_per_feature:
                obs_peak_auc_arr = np.array(obs_peak_auc_per_feature)
                metrics.update({
                    'obs_peak_auc_median': np.median(obs_peak_auc_arr)
                })
           
        return metrics
    
    def _print_batch_log(self, metrics, loss_dict, batch_idx, epoch, batch_processing_time=None):
        """Print batch logging similar to old_train_candi.py format."""
        if not metrics or not loss_dict:
            return
            
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        lr_printstatement = f"LR {current_lr:.2e}"
        
        # Get gradient norm if available
        grad_norm = 0.0
        if hasattr(self, 'grad_norm'):
            grad_norm = self.grad_norm
        
        # Get number of masked features (approximate)
        num_mask = "N/A"
        if hasattr(self, 'last_num_mask'):
            num_mask = self.last_num_mask
        
        # Format batch processing time
        batch_time_str = "N/A"
        batch_time = 0.0
        if batch_processing_time is not None:
            batch_time_str = f"{batch_processing_time:.2f}s"
            batch_time = batch_processing_time
        
        # Record progress for CSV logging
        self._record_progress(epoch, batch_idx, metrics, loss_dict, grad_norm, num_mask, batch_time, current_lr)
        
        # Improved and aesthetic print statement with aligned columns and section headers
        sep = " | "
        metrics_table = [
            # Header
            f"{'='*84}",
            f" Epoch: {epoch+1:<4}   Batch: {batch_idx:<4}/{self.estimated_batches_per_epoch}  ({100.0 * batch_idx / self.estimated_batches_per_epoch:.1f}%)".ljust(82),
            f"{'-'*84}",
            # Section: Losses
            f"{'Type':<8} {'nbNLL':>10} {'gNLL':>10} {'peakLoss':>12}",
            f"{'-'*84}",
            f"{'Imp':<8} "
            f"{loss_dict.get('imp_count_loss', 0.0):>10.2f} "
            f"{loss_dict.get('imp_pval_loss', 0.0):>10.2f} "
            f"{loss_dict.get('imp_peak_loss', 0.0):>12.2f}",
            f"{'Obs':<8}"
            f"{loss_dict.get('obs_count_loss', 0.0):>10.2f} "
            f"{loss_dict.get('obs_pval_loss', 0.0):>10.2f} "
            f"{loss_dict.get('obs_peak_loss', 0.0):>12.2f}",
            f"{'-'*84}",
            # Section: R2
            f"{'':<8} {'Count R2':>10} {'Pval R2':>10}",
            f"{'Imp':<8} "
            f"{metrics.get('imp_count_r2_median', 0.0):>10.2f} "
            f"{metrics.get('imp_pval_r2_median', 0.0):>10.2f}",
            f"{'Obs':<8} "
            f"{metrics.get('obs_count_r2_median', 0.0):>10.2f} "
            f"{metrics.get('obs_pval_r2_median', 0.0):>10.2f}",
            f"{'-'*84}",
            # Section: Spearman
            f"{'':<8} {'Count SRCC':>10} {'Pval SRCC':>10}",
            f"{'Imp':<8} "
            f"{metrics.get('imp_count_spearman_median', 0.0):>10.2f} "
            f"{metrics.get('imp_pval_spearman_median', 0.0):>10.2f}",
            f"{'Obs':<8} "
            f"{metrics.get('obs_count_spearman_median', 0.0):>10.2f} "
            f"{metrics.get('obs_pval_spearman_median', 0.0):>10.2f}",
            f"{'-'*84}",
            # Section: Pearson
            f"{'':<8} {'Count PCC':>10} {'Pval PCC':>10}",
            f"{'Imp':<8} "
            f"{metrics.get('imp_count_pearson_median', 0.0):>10.2f} "
            f"{metrics.get('imp_pval_pearson_median', 0.0):>10.2f}",
            f"{'Obs':<8} "
            f"{metrics.get('obs_count_pearson_median', 0.0):>10.2f} "
            f"{metrics.get('obs_pval_pearson_median', 0.0):>10.2f}",
            f"{'-'*84}",
            # Section: MSE
            f"{'':<8} {'Count MSE':>10} {'Pval MSE':>10}",
            f"{'Imp':<8} "
            f"{metrics.get('imp_count_mse_median', 0.0):>10.2f} "
            f"{metrics.get('imp_pval_mse_median', 0.0):>10.2f}",
            f"{'Obs':<8} "
            f"{metrics.get('obs_count_mse_median', 0.0):>10.2f} "
            f"{metrics.get('obs_pval_mse_median', 0.0):>10.2f}",
            f"{'-'*84}",
            # Section: Peak AUC
            f"{'':<8} {'Peak AUC':>10}",
            f"{'Imp':<8} {metrics.get('imp_peak_auc_median', 0.0):>10.2f}",
            f"{'Obs':<8} {metrics.get('obs_peak_auc_median', 0.0):>10.2f}",
            f"{'-'*84}",
        ]
        
        # Add EMA values (as additional nicely formatted section)
        if hasattr(self, 'specific_ema') and self.specific_ema:
            metrics_table.append("EMA (Exponential Moving Average):")
            metrics_table.append(f"{'':<8} {'Count_Loss':>10} {'Obs_Count_Loss':>15} {'Pval_Loss':>15} {'Obs_Pval_Loss':>15}")
            metrics_table.append(
                f"{'EMA':<8} "
                f"{self.specific_ema.get('imp_count_loss', 0.0):>10.2f} "
                f"{self.specific_ema.get('obs_count_loss', 0.0):>15.2f} "
                f"{self.specific_ema.get('imp_pval_loss', 0.0):>15.2f} "
                f"{self.specific_ema.get('obs_pval_loss', 0.0):>15.2f}"
            )
            metrics_table.append(f"{'':<8} {'Pval_R2':>10} {'Pval_SRCC':>12} {'Pval_PCC':>12} {'Obs_Peak_AUC':>14}")
            metrics_table.append(
                f"{'EMA':<8} "
                f"{self.specific_ema.get('imp_pval_r2_median', 0.0):>10.2f} "
                f"{self.specific_ema.get('imp_pval_spearman_median', 0.0):>12.2f} "
                f"{self.specific_ema.get('imp_pval_pearson_median', 0.0):>12.2f} "
                f"{self.specific_ema.get('obs_peak_auc_median', 0.0):>14.2f}"
            )
            metrics_table.append(f"{'':<8} {'Count_R2':>10} {'Count_SRCC':>12} {'Count_PCC':>12} {'Peak_AUC':>14}")
            metrics_table.append(
                f"{'EMA':<8} "
                f"{self.specific_ema.get('imp_count_r2_median', 0.0):>10.2f} "
                f"{self.specific_ema.get('imp_count_spearman_median', 0.0):>12.2f} "
                f"{self.specific_ema.get('imp_count_pearson_median', 0.0):>12.2f} "
                f"{self.specific_ema.get('imp_peak_auc_median', 0.0):>14.2f}"
            )
            metrics_table.append(f"{'-'*84}")

        # Add training dynamics and environment
        metrics_table.append(
            f"Gradient Norm: {grad_norm:.2f}   Num Masked: {num_mask}   {lr_printstatement}   Batch time: {batch_time_str}"
        )
        metrics_table.append(f"{'='*84}")

        # Then, before printing the metrics table (around line 1301), add this:
        # Clear previous table by moving cursor up and clearing lines
        # if hasattr(self, 'last_table_lines') and self.last_table_lines > 0:
        #     # Move cursor up and clear each line
        #     print('\033[F\033[K' * self.last_table_lines, end='')
        
        # Print organized metrics
        table_output = "\n".join(metrics_table)
        print(table_output)
        print("\n")

        # Store the number of lines for next time
        self.last_table_lines = len(metrics_table)
    
    def _save_progress_to_csv(self, epoch, batch_idx):
        """Save progress data to CSV file every 100 batches."""
        if not self.progress_data:
            return
        
        # Create filename if not already set (only once)
        if self.progress_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.progress_file = Path(self.progress_dir) / f"training_progress_{timestamp}.csv"
            if self.is_main_process:
                print(f"Progress will be saved to: {self.progress_file}")
            
        # Create DataFrame from progress data
        df = pd.DataFrame(self.progress_data)
        
        # Save to CSV (overwrite existing file)
        df.to_csv(self.progress_file, index=False)
        
        if self.is_main_process:
            print(f"Progress updated in {self.progress_file} ({len(df)} records)")
    
    def _record_progress(self, epoch, batch_idx, metrics, loss_dict, grad_norm, num_mask, batch_time, lr):
        """Record all progress metrics for CSV logging."""
        # Get EMA values if available
        ema_values = {}
        if hasattr(self, 'specific_ema') and self.specific_ema:
            ema_values = {
                'EMA_Imp_Pval_R2': self.specific_ema.get('imp_pval_r2_median', 0.0),
                'EMA_Imp_Pval_SRCC': self.specific_ema.get('imp_pval_spearman_median', 0.0),
                'EMA_Imp_Pval_PCC': self.specific_ema.get('imp_pval_pearson_median', 0.0),
                'EMA_Imp_Count_R2': self.specific_ema.get('imp_count_r2_median', 0.0),
                'EMA_Imp_Count_SRCC': self.specific_ema.get('imp_count_spearman_median', 0.0),
                'EMA_Imp_Count_PCC': self.specific_ema.get('imp_count_pearson_median', 0.0),
                'EMA_Imp_Count_Loss': self.specific_ema.get('imp_count_loss', 0.0),
                'EMA_Obs_Count_Loss': self.specific_ema.get('obs_count_loss', 0.0),
                'EMA_Imp_Pval_Loss': self.specific_ema.get('imp_pval_loss', 0.0),
                'EMA_Obs_Pval_Loss': self.specific_ema.get('obs_pval_loss', 0.0),
                'EMA_Imp_Peak_AUC': self.specific_ema.get('imp_peak_auc_median', 0.0),
                'EMA_Obs_Peak_AUC': self.specific_ema.get('obs_peak_auc_median', 0.0)
            }
        
        # Create record with all metrics
        record = {
            'epoch': epoch + 1,
            'batch_idx': batch_idx,
            'timestamp': datetime.now().isoformat(),
            'learning_rate': lr,
            'gradient_norm': grad_norm,
            'num_mask': num_mask,
            'batch_time': batch_time,
            
            # Loss values
            'imp_count_loss': loss_dict.get('imp_count_loss', 0.0),
            'obs_count_loss': loss_dict.get('obs_count_loss', 0.0),
            'imp_pval_loss': loss_dict.get('imp_pval_loss', 0.0),
            'obs_pval_loss': loss_dict.get('obs_pval_loss', 0.0),
            'imp_peak_loss': loss_dict.get('imp_peak_loss', 0.0),
            'obs_peak_loss': loss_dict.get('obs_peak_loss', 0.0),
            'total_loss': loss_dict.get('total_loss', 0.0),
            
            # Imputation metrics
            'imp_count_r2_median': metrics.get('imp_count_r2_median', 0.0),
            'imp_count_spearman_median': metrics.get('imp_count_spearman_median', 0.0),
            'imp_count_pearson_median': metrics.get('imp_count_pearson_median', 0.0),
            'imp_count_mse_median': metrics.get('imp_count_mse_median', 0.0),
            'imp_count_perplexity_median': metrics.get('imp_count_perplexity_median', 0.0),
            
            'imp_pval_r2_median': metrics.get('imp_pval_r2_median', 0.0),
            'imp_pval_spearman_median': metrics.get('imp_pval_spearman_median', 0.0),
            'imp_pval_pearson_median': metrics.get('imp_pval_pearson_median', 0.0),
            'imp_pval_mse_median': metrics.get('imp_pval_mse_median', 0.0),
            'imp_pval_perplexity_median': metrics.get('imp_pval_perplexity_median', 0.0),
            
            'imp_peak_auc_median': metrics.get('imp_peak_auc_median', 0.0),
            
            # Upsampling metrics
            'obs_count_r2_median': metrics.get('obs_count_r2_median', 0.0),
            'obs_count_spearman_median': metrics.get('obs_count_spearman_median', 0.0),
            'obs_count_pearson_median': metrics.get('obs_count_pearson_median', 0.0),
            'obs_count_mse_median': metrics.get('obs_count_mse_median', 0.0),
            'obs_count_perplexity_median': metrics.get('obs_count_perplexity_median', 0.0),
            
            'obs_pval_r2_median': metrics.get('obs_pval_r2_median', 0.0),
            'obs_pval_spearman_median': metrics.get('obs_pval_spearman_median', 0.0),
            'obs_pval_pearson_median': metrics.get('obs_pval_pearson_median', 0.0),
            'obs_pval_mse_median': metrics.get('obs_pval_mse_median', 0.0),
            'obs_pval_perplexity_median': metrics.get('obs_pval_perplexity_median', 0.0),
            
            'obs_peak_auc_median': metrics.get('obs_peak_auc_median', 0.0),
            
            # EMA values
            **ema_values
        }
        
        # Add to progress data
        self.progress_data.append(record)
        
        # Save to CSV every 100 batches or every batch if it's the first few batches
        if batch_idx % 100 == 0 or batch_idx < 10:
            self._save_progress_to_csv(epoch, batch_idx)
    
    def _update_progress_monitoring(self, metrics, loss_dict, specific_ema_alpha = 0.005):
        """
        Update progress monitoring with EMA tracking and check for learning rate adjustment.
        
        Args:
            metrics: Dictionary of computed metrics
            loss_dict: Dictionary of loss values
        """
            
        # Initialize specific EMA tracking for requested metrics (alpha=0.005)
        if not hasattr(self, 'specific_ema'):
            self.specific_ema = {}
            self.specific_ema_alpha = specific_ema_alpha
        
        
        # Add loss values (negated for monitoring increasing trends)
        loss_keys = ["imp_count_loss", "obs_count_loss", "imp_pval_loss", "obs_pval_loss", "imp_peak_loss", "obs_peak_loss"]
        
        # Update specific EMA tracking for requested metrics
        specific_metrics = [
            "imp_pval_r2_median", "imp_count_r2_median",
            "imp_pval_spearman_median", "imp_count_spearman_median", 
            "imp_pval_pearson_median", "imp_count_pearson_median",
            "imp_peak_auc_median", "obs_peak_auc_median"
        ]
        
        for key in specific_metrics + loss_keys:
            if key in metrics:
                if key not in metrics:
                    continue
                value = metrics[key]
                if key not in self.specific_ema:
                    self.specific_ema[key] = value
                else:
                    self.specific_ema[key] = self.specific_ema_alpha * value + (1 - self.specific_ema_alpha) * self.specific_ema[key]
            else:
                if key not in loss_dict:
                    continue
                value = loss_dict[key]
                if key not in self.specific_ema:
                    self.specific_ema[key] = value
                else:
                    self.specific_ema[key] = self.specific_ema_alpha * value + (1 - self.specific_ema_alpha) * self.specific_ema[key]
    
    def _log_metrics(self, metrics, loss_dict, batch_idx):
        """
        Log per-feature metrics with IQR, median, min, max, and EMA values.
        
        Args:
            metrics: Dictionary of computed per-feature metrics with aggregated statistics
            loss_dict: Dictionary of loss values  
            batch_idx: Current batch index
        """
        if not self.is_main_process:
            return
            
        # Log current batch per-feature metrics (IQR, median, min, max)
        if metrics:
            print(f"Batch {batch_idx} Per-Feature Metrics (Aggregated):")
            
            # Group metrics by type for better readability
            metric_types = ['imp_count_r2', 'imp_count_spearman', 'imp_count_pearson',
                           'imp_pval_r2', 'imp_pval_spearman', 'imp_pval_pearson',
                           'obs_count_r2', 'obs_count_spearman', 'obs_count_pearson',
                           'obs_pval_r2', 'obs_pval_spearman', 'obs_pval_pearson']
            
            for metric_type in metric_types:
                # Check if we have all aggregation statistics for this metric
                median_key = f"{metric_type}_median"
                iqr_key = f"{metric_type}_iqr"
                min_key = f"{metric_type}_min"
                max_key = f"{metric_type}_max"
                
                if all(key in metrics for key in [median_key, iqr_key, min_key, max_key]):
                    median_val = metrics[median_key]
                    iqr_val = metrics[iqr_key]
                    min_val = metrics[min_key]
                    max_val = metrics[max_key]
                    
                    if not any(np.isnan([median_val, iqr_val, min_val, max_val])):
                        print(f"  {metric_type}: median={median_val:.4f}, IQR={iqr_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
        
        # Log EMA values if available (using median values for EMA tracking)
        if hasattr(self, 'prog_mon_ema') and len(self.prog_mon_ema) > 0:
            print(f"EMA Metrics (based on median values):")
            for key, value in self.prog_mon_ema.items():
                if 'loss' in key:
                    # Show original loss value (un-negated)
                    print(f"  EMA_{key}: {-1*value:.4f}")
                else:
                    print(f"  EMA_{key}: {value:.4f}")
    
    def _validate_batch(self, batch):
        """
        Validate that the batch has the expected structure from CANDIIterableDataset.
        
        Args:
            batch: Dictionary containing batch data from CANDIIterableDataset
            
        Returns:
            bool: True if batch is valid, False otherwise
        """
        if not isinstance(batch, dict):
            return False
            
        # Expected keys from CANDIIterableDataset
        expected_keys = {'x_data', 'x_meta', 'x_avail', 'x_dna', 
                        'y_data', 'y_meta', 'y_avail', 'y_pval', 'y_peaks', 'y_dna'}
        
        if not all(key in batch for key in expected_keys):
            missing_keys = expected_keys - set(batch.keys())
            if self.is_main_process:
                print(f"Missing keys in batch: {missing_keys}")
            return False
            
        # Check that all values are tensors
        for key, value in batch.items():
            if key != 'sample_id' and not isinstance(value, torch.Tensor):
                if self.is_main_process:
                    print(f"Non-tensor value for key {key}: {type(value)}")
                return False
                
        return True
    
    def _move_batch_to_device(self, batch):
        """
        Move all tensor values in the batch to the training device.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            dict: Batch with tensors moved to device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value  # Keep non-tensors as is (e.g., sample_id)
        return device_batch
    
    def _log_batch_info(self, batch, batch_idx, loss_dict=None):
        """
        Log information about the batch for debugging.
        
        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of the current batch
            loss_dict: Optional dictionary containing loss values
        """
        print(f"Batch {batch_idx} info:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}, device {value.device}")
            else:
                print(f"  {key}: {type(value)} - {value}")
        
        if loss_dict is not None:
            print(f"  Losses:")
            for loss_name, loss_value in loss_dict.items():
                print(f"    {loss_name}: {loss_value:.4f}")
        
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB" if torch.cuda.is_available() else "  CPU mode")
    
    def _save_checkpoint(self, epoch, model_name):
        """
        Save model checkpoint after each epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            model_name: Name of the model for checkpoint naming
        """
        if not self.is_main_process:
            return
            
        # Set up checkpoint directory if not already done
        if self.checkpoint_dir is None:
            self.checkpoint_dir = Path(self.progress_dir) / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove previous checkpoint if it exists
        if self.current_checkpoint_path and self.current_checkpoint_path.exists():
            self.current_checkpoint_path.unlink()
            if self.is_main_process:
                print(f"🗑️  Removed previous checkpoint: {self.current_checkpoint_path.name}")
        
        # Create new checkpoint path
        checkpoint_name = f"{model_name}_epoch_{epoch+1}.pt"
        self.current_checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save model state dict
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), self.current_checkpoint_path)
        
        if self.is_main_process:
            print(f"💾 Epoch {epoch+1} checkpoint saved: {self.current_checkpoint_path.name}")

##=========================================== Loader =====================================================##

class CANDI_LOADER(object):
    def __init__(self, model_path, hyper_parameters, DNA=False):
        self.model_path = model_path
        self.hyper_parameters = hyper_parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DNA = DNA

    def load_CANDI(self):
        signal_dim = self.hyper_parameters["signal_dim"]
        dropout = self.hyper_parameters["dropout"]
        nhead = self.hyper_parameters["nhead"]
        n_sab_layers = self.hyper_parameters["n_sab_layers"]
        metadata_embedding_dim = self.hyper_parameters["metadata_embedding_dim"]
        context_length = self.hyper_parameters["context_length"]

        n_cnn_layers = self.hyper_parameters["n_cnn_layers"]
        conv_kernel_size = self.hyper_parameters["conv_kernel_size"]
        pool_size = self.hyper_parameters["pool_size"]
        separate_decoders = self.hyper_parameters["separate_decoders"]
        
        if self.DNA:
            if self.hyper_parameters["unet"]:
                model = CANDI_UNET(
                    signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                    n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length, 
                    separate_decoders=separate_decoders)
            else:
                model = CANDI(
                    signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                    n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length, 
                    separate_decoders=separate_decoders)
        else:
            model = CANDI(
                signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
                n_sab_layers, pool_size=pool_size, dropout=dropout, context_length=context_length,
                separate_decoders=separate_decoders)

        model.load_state_dict(torch.load(self.model_path, map_location=self.device)) 

        model = model.to(self.device)
        return model

##=========================================== DDP Utilities ==============================================##

def init_distributed():
    """Initialize distributed training. Returns rank, world_size, local_rank."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Check if the requested device exists
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for DDP training")
        
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"Requested local_rank {local_rank} but only {torch.cuda.device_count()} CUDA devices available")
        
        # Initialize the process group
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return None, None, None

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def check_gpu_availability():
    """Check GPU availability and provide guidance for DDP setup."""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system.")
        print("   For DDP training, you need CUDA-enabled GPUs.")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"🔍 Found {gpu_count} CUDA device(s):")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"   GPU {i}: {gpu_name}")
    
    return gpu_count > 0

##=========================================== CLI Interface ===============================================##

def create_argument_parser():
    """Create and configure the argument parser with organized argument groups."""
    parser = argparse.ArgumentParser(
        description="CANDI: Context-Aware Neural Data Imputation - Modern Training Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic EIC training with default settings
            python train.py --eic --epochs 10 --batch-size 16
            
            # Multi-GPU training with mixed precision
            python train.py --eic --ddp --mixed-precision --epochs 20 --batch-size 32
            
            # Custom model architecture with suffix
            python train.py --merged --nhead 12 --n-sab-layers 6 --expansion-factor 4 --name-suffix "experiment1"
            
            # Training with validation and checkpointing
            python train.py --eic --enable-validation --save-dir ./models --checkpoint-freq 5
            
            # U-Net model with custom loci strategy
            python train.py --eic --unet --loci-gen full_chr --num-loci 1000 --name-suffix "unet_test"
                    """)
    
    # === DATA CONFIGURATION ===
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--eic', action='store_true',
                           help='Use EIC dataset (default: merged dataset)')
    data_group.add_argument('--merged', action='store_true',
                           help='Use merged dataset (default if neither --eic nor --merged specified)')
    data_group.add_argument('--data-path', type=str, 
                           default="/home/mforooz/projects/def-maxwl/mforooz/",
                           help='Base path to the datasets')
    data_group.add_argument('--num-loci', '-m', type=int, default=5000,
                           help='Number of genomic loci to generate for training')
    data_group.add_argument('--context-length', type=int, default=1200,
                           help='Context length for genomic windows (in bins)')
    data_group.add_argument('--loci-gen', type=str, default='full_chr', 
                           choices=['random', 'ccre', 'full_chr', 'gw'],
                           help='Strategy for generating genomic loci')
    data_group.add_argument('--must-have-chr-access', action='store_true',
                           help='Require chromosome access for all experiments')
    data_group.add_argument('--min-avail', type=int, default=3,
                           help='Minimum number of available experiments per biosample')
    
    # === MODEL ARCHITECTURE ===
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--nhead', type=int, default=9,
                            help='Number of attention heads in transformer')
    model_group.add_argument('--n-sab-layers', type=int, default=4,
                            help='Number of self-attention blocks')
    model_group.add_argument('--n-cnn-layers', type=int, default=3,
                            help='Number of CNN layers in encoder/decoder')
    model_group.add_argument('--conv-kernel-size', type=int, default=3,
                            help='Convolution kernel size')
    model_group.add_argument('--pool-size', type=int, default=2,
                            help='Pooling size for CNN layers')
    model_group.add_argument('--expansion-factor', type=int, default=3,
                            help='Channel expansion factor for CNN layers')
    model_group.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout rate')
    model_group.add_argument('--pos-enc', type=str, default='relative',
                            choices=['relative', 'absolute'],
                            help='Type of positional encoding')
    model_group.add_argument('--separate-decoders', action='store_true', default=True,
                            help='Use separate decoders for count and p-value prediction')
    model_group.add_argument('--shared-decoders', action='store_true',
                            help='Use shared decoder (overrides --separate-decoders)')
    model_group.add_argument('--unet', action='store_true',
                            help='Use U-Net skip connections')
    
    # === TRAINING CONFIGURATION ===
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument('--epochs', type=int, default=10,
                               help='Number of training epochs')
    training_group.add_argument('--batch-size', type=int, default=25,
                               help='Training batch size')
    training_group.add_argument('--learning-rate', '--lr', type=float, default=1e-3,
                               help='Initial learning rate')
    training_group.add_argument('--optimizer', type=str, default='adamax',
                               choices=['adamax', 'adam', 'adamw', 'sgd'],
                               help='Optimizer type')
    # Scheduler is always cosine (linear warmup + cosine annealing)
    training_group.add_argument('--inner-epochs', type=int, default=1,
                               help='Number of inner epochs per batch')
    training_group.add_argument('--enable-validation', action='store_true',
                               help='Enable validation during training')
    
    # === SYSTEM CONFIGURATION ===
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument('--device', type=str, default=None,
                             help='Device to use (cuda:0, cpu, etc.). Auto-detect if not specified')
    system_group.add_argument('--mixed-precision', action='store_true', default=True,
                             help='Enable mixed precision training (default: True on CUDA)')
    system_group.add_argument('--no-mixed-precision', action='store_true',
                             help='Disable mixed precision training')
    system_group.add_argument('--ddp', action='store_true',
                             help='Enable Distributed Data Parallel training')
    system_group.add_argument('--rank', type=int, default=None,
                             help='Process rank for DDP (auto-detected if not specified)')
    system_group.add_argument('--world-size', type=int, default=None,
                             help='World size for DDP (auto-detected if not specified)')
    system_group.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')
    system_group.add_argument('--check-gpus', action='store_true',
                             help='Check GPU availability and exit')
    
    # === MODEL SAVING/LOADING ===
    io_group = parser.add_argument_group('Model I/O')
    io_group.add_argument('--save-dir', type=str, default='./models',
                         help='Directory to save trained models')
    io_group.add_argument('--progress-dir', type=str, default='./progress',
                         help='Directory to save training progress CSV files')
    io_group.add_argument('--checkpoint', type=str, default=None,
                         help='Path to checkpoint to resume training from')
    io_group.add_argument('--checkpoint-freq', type=int, default=5,
                         help='Save checkpoint every N epochs')
    io_group.add_argument('--model-name', type=str, default=None,
                         help='Custom model name (auto-generated if not specified)')
    io_group.add_argument('--name-suffix', type=str, default=None,
                         help='Suffix to append to auto-generated model name (format: YYYYMMDD_HHMMSS_CANDI[UNET]_dataset_lociStrategy_numLoci_suffix)')
    io_group.add_argument('--no-save', action='store_true',
                         help='Do not save the trained model')
    
    # === CONFIGURATION FILE ===
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', type=str, default=None,
                             help='Path to YAML/JSON configuration file')
    config_group.add_argument('--save-config', type=str, default=None,
                             help='Save current configuration to file')
    
    # === ADVANCED OPTIONS ===
    advanced_group = parser.add_argument_group('Advanced Options')
    advanced_group.add_argument('--dsf-list', type=int, nargs='+', default=[1, 2],
                               help='Downsampling factors to use')
    advanced_group.add_argument('--specific_ema_alpha', type=float, default=0.005,
                               help='Alpha for specific EMA tracking')
    advanced_group.add_argument('--debug', action='store_true',
                               help='Enable debug mode with extra logging')
    
    return parser

def load_config_file(config_path):
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

def save_config_file(config_dict, config_path):
    """Save configuration to YAML or JSON file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            except ImportError:
                # Fallback to JSON if PyYAML not available
                config_path = config_path.with_suffix('.json')
                with open(config_path, 'w') as jf:
                    json.dump(config_dict, jf, indent=2)
        else:
            json.dump(config_dict, f, indent=2)

def validate_arguments(args):
    """Validate argument combinations and set defaults."""
    errors = []
    
    # Dataset selection
    if args.eic and args.merged:
        errors.append("Cannot specify both --eic and --merged")
    elif not args.eic and not args.merged:
        args.merged = True  # Default to merged
    
    # Mixed precision
    if args.no_mixed_precision:
        args.mixed_precision = False
    
    
    # Decoder configuration
    if args.shared_decoders:
        args.separate_decoders = False
    
    # DDP validation
    if args.ddp:
        if args.rank is None:
            args.rank = int(os.environ.get('RANK', 0))
        if args.world_size is None:
            args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if args.world_size <= 1:
            print("Warning: DDP requested but world_size <= 1. Disabling DDP.")
            args.ddp = False
        elif not torch.cuda.is_available():
            print("Warning: DDP requested but CUDA is not available. Disabling DDP.")
            args.ddp = False
        elif args.world_size > torch.cuda.device_count():
            print(f"Warning: DDP requested {args.world_size} processes but only {torch.cuda.device_count()} CUDA devices available. Disabling DDP.")
            args.ddp = False
    
    # Path validation
    data_path = Path(args.data_path)
    if not data_path.exists():
        errors.append(f"Data path does not exist: {data_path}")
    
    # Model architecture validation
    if args.nhead <= 0:
        errors.append("Number of attention heads must be positive")
    if args.n_sab_layers <= 0:
        errors.append("Number of SAB layers must be positive")
    if args.context_length <= 0:
        errors.append("Context length must be positive")
    
    if errors:
        raise ValueError("Argument validation failed:\n" + "\n".join(f"  - {err}" for err in errors))
    
    return args

def create_model_from_args(args, signal_dim, num_sequencing_platforms=10, num_runtypes=4):
    """Create CANDI model based on CLI arguments."""
    metadata_embedding_dim = signal_dim * 4
    
    if args.unet:
        model = CANDI_UNET(
            signal_dim=signal_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            conv_kernel_size=args.conv_kernel_size,
            n_cnn_layers=args.n_cnn_layers,
            nhead=args.nhead,
            n_sab_layers=args.n_sab_layers,
            pool_size=args.pool_size,
            dropout=args.dropout,
            context_length=args.context_length,
            pos_enc=args.pos_enc,
            expansion_factor=args.expansion_factor,
            separate_decoders=args.separate_decoders,
            num_sequencing_platforms=num_sequencing_platforms,
            num_runtypes=num_runtypes
        )
    else:
        model = CANDI(
            signal_dim=signal_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            conv_kernel_size=args.conv_kernel_size,
            n_cnn_layers=args.n_cnn_layers,
            nhead=args.nhead,
            n_sab_layers=args.n_sab_layers,
            pool_size=args.pool_size,
            dropout=args.dropout,
            context_length=args.context_length,
            pos_enc=args.pos_enc,
            expansion_factor=args.expansion_factor,
            separate_decoders=args.separate_decoders,
            num_sequencing_platforms=num_sequencing_platforms,
            num_runtypes=num_runtypes
        )
    
    return model

def setup_device(args):
    """Setup device based on arguments and availability."""
    if args.device is not None:
        device = torch.device(args.device)
    else:
        if args.ddp and 'LOCAL_RANK' in os.environ:
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return device

def generate_model_name(args, timestamp=None):
    """Generate a descriptive model name based on configuration."""
    if args.model_name:
        return args.model_name
    
    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    dataset_type = "eic" if args.eic else "merged"
    arch_type = "CANDI_UNET" if args.unet else "CANDI"
    loci_strategy = args.loci_gen
    num_loci = args.num_loci
    
    name_parts = [
        timestamp,
        arch_type,
        dataset_type,
        f"{loci_strategy}_{num_loci}loci"
    ]
    
    # Add suffix if provided
    if args.name_suffix:
        name_parts.append(args.name_suffix)
    
    return "_".join(name_parts)

def print_training_summary(args, model, device):
    """Print a summary of the training configuration."""

    print("=" * 80)
    print("🚀 CANDI Training Configuration")
    print("=" * 80)

    # Dataset info
    dataset_type = "EIC" if args.eic else "Merged"
    print(f"📊 Dataset: {dataset_type} ({args.data_path})")
    print(f"   Loci: {args.num_loci}, Context: {args.context_length}, Strategy: {args.loci_gen}")

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch_type = "U-Net" if args.unet else "Standard"
    decoder_type = "Shared" if not args.separate_decoders else "Separate"

    print(f"🏗️  Model: CANDI-{arch_type} with {decoder_type} Decoders")
    print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"   Architecture: {args.nhead} heads, {args.n_sab_layers} SAB layers, {args.n_cnn_layers} CNN layers")

    # Print model summary
    print("\n📝 Model Summary:")
    try:
        from torchinfo import summary
        summary(model)
    except ImportError:
        print("torchinfo not installed. Printing model using print(model):")
        print(model)
    except Exception as e:
        print(f"Could not print model summary: {e}")
        print(model)

    # Training info
    print(f"🎯 Training: {args.epochs} epochs, batch size {args.batch_size}")
    print(f"   Optimizer: {args.optimizer.upper()}, LR: {args.learning_rate}, Scheduler: Cosine (Linear Warmup + Cosine Annealing)")
    print(f"   Masking: Random masking")

    # System info
    print(f"💻 System: {device}")
    if args.ddp:
        print(f"   DDP: Rank {args.rank}/{args.world_size}")
    if args.mixed_precision:
        print(f"   Mixed Precision: Enabled")

    # I/O info
    if not args.no_save:
        print(f"💾 Output: {args.save_dir}")
        print(f"   Checkpoints: Every {args.checkpoint_freq} epochs")
    print("=" * 80)

def main():
    """Main CLI entry point for CANDI training."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration file if specified
    if args.config:
        try:
            config = load_config_file(args.config)
            # Update args with config values (CLI args take precedence)
            for key, value in config.items():
                key_attr = key.replace('-', '_')
                if not hasattr(args, key_attr) or getattr(args, key_attr) == parser.get_default(key_attr):
                    setattr(args, key_attr, value)
        except Exception as e:
            print(f"❌ Error loading configuration file: {e}")
            return 1
    
    # Save configuration if requested
    if args.save_config:
        config_dict = {k.replace('_', '-'): v for k, v in vars(args).items() if v is not None}
        try:
            save_config_file(config_dict, args.save_config)
            print(f"✅ Configuration saved to: {args.save_config}")
            return 0
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
            return 1
    
    # Handle GPU check option
    if args.check_gpus:
        check_gpu_availability()
        return 0
    
    # Validate arguments
    try:
        args = validate_arguments(args)
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = setup_device(args)
    
    # Initialize DDP if requested
    if args.ddp:
        try:
            rank, world_size, local_rank = init_distributed()
            if rank is not None:
                args.rank, args.world_size = rank, world_size
            else:
                print("❌ Failed to initialize DDP: Environment variables not set properly")
                print("Make sure to run with torchrun: torchrun --nproc_per_node=N train.py ...")
                return 1
        except Exception as e:
            print(f"❌ Failed to initialize DDP: {e}")
            check_gpu_availability()
            print("Falling back to single-GPU training...")
            args.ddp = False
            args.rank = None
            args.world_size = None
    
    # try:
    # Create dataset parameters
    dataset_type = "eic" if args.eic else "merged"
    base_path = args.data_path
    if not base_path.endswith('/'):
        base_path += '/'
    
    if args.eic:
        data_path = base_path + "DATA_CANDI_EIC/"
    else:
        data_path = base_path + "DATA_CANDI_MERGED/"
    
    dataset_params = {
        'base_path': data_path,
        'dataset_type': dataset_type,
        'm': args.num_loci,
        'context_length': args.context_length * 25, 
        'split': 'train',
        'loci_gen_strategy': args.loci_gen,
        'dsf_list': args.dsf_list,
        'DNA': True,
        'must_have_chr_access': args.must_have_chr_access,
        'bios_min_exp_avail_threshold': args.min_avail,
        'shuffle_bios': True
    }
    
    # Create training parameters
    training_params = {
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'inner_epochs': args.inner_epochs,
        'enable_validation': args.enable_validation,
        'use_mixed_precision': args.mixed_precision,
        'specific_ema_alpha': args.specific_ema_alpha,
        'progress_dir': args.progress_dir,
        'debug': args.debug,
        'DNA': True,
        'no_save': args.no_save
    }

    # Create temporary dataset to get signal_dim and metadata information
    temp_dataset = CANDIIterableDataset(**dataset_params)
    signal_dim = len(temp_dataset.aliases['experiment_aliases'])
    
    # Get metadata information from the dataset
    num_sequencing_platforms = temp_dataset.num_sequencing_platforms
    num_runtypes = 4  # Based on the mapping in EmbedMetadata: 0, 1, 2 (missing), 3 (cloze_masked)
    
    # Store signal_dim and metadata info for later use in training
    dataset_params['signal_dim'] = signal_dim
    dataset_params['num_sequencing_platforms'] = num_sequencing_platforms
    dataset_params['num_runtypes'] = num_runtypes
    
    # Create model with metadata information
    model = create_model_from_args(args, signal_dim, num_sequencing_platforms, num_runtypes)
    
    # Load checkpoint if specified
    if args.checkpoint:
        if Path(args.checkpoint).exists():
            print(f"📂 Loading checkpoint: {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        else:
            print(f"⚠️ Checkpoint not found: {args.checkpoint}") 
    
    # Create trainer
    trainer = CANDI_TRAINER(
        model=model,
        dataset_params=dataset_params,
        training_params=training_params,
        device=device,
        rank=args.rank if args.ddp else None,
        world_size=args.world_size if args.ddp else None
    )
    
    # Generate timestamp once and use it consistently for model naming
    # This ensures all files (progress CSV, checkpoints, final model) go to the same directory
    training_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = generate_model_name(args, training_timestamp)
    print(f"🏷️  Model name: {model_name}")
    
    # Create model directory and save config at start of training
    if not args.no_save and (not args.ddp or args.rank == 0):
        model_dir = Path(args.save_dir) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config at start of training
        config_dict = {k.replace('_', '-'): v for k, v in vars(args).items() if v is not None}
        config_dict['model_parameters'] = sum(p.numel() for p in model.parameters())
        config_dict['signal_dim'] = signal_dim
        config_dict['num_sequencing_platforms'] = num_sequencing_platforms
        config_dict['num_runtypes'] = num_runtypes
        
        config_path = model_dir / f"{model_name}_config.json"
        save_config_file(config_dict, config_path)
        print(f"📝 Configuration saved to: {config_path}")
        
        # Update trainer's progress_dir to use the model directory
        trainer.progress_dir = str(model_dir)
        trainer.progress_file = None  # Reset progress file to use new directory
        trainer.model_name = model_name  # Set model name for checkpoint saving
    
    # Start training
    start_time = time.time()
    
    # Print training start
    print_training_summary(args, model, device)
    
    trained_model = trainer.train()
    
    end_time = time.time()
    training_duration = end_time - start_time
    
    # Print training completion
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining Complete!")
    print(f"Duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print("=" * 80)
    
    # Debug information for model saving
    print(f"🔍 Model saving debug info:")
    print(f"   args.no_save: {args.no_save}")
    print(f"   args.ddp: {args.ddp}")
    print(f"   args.rank: {args.rank}")
    print(f"   Condition result: {not args.no_save and (not args.ddp or args.rank == 0)}")
    
    # Save model if requested (with fallback for safety)
    should_save = not args.no_save and (not args.ddp or args.rank == 0)
    print(f"💾 Model saving decision: {should_save}")
    
    if should_save:
        model_dir = Path(args.save_dir) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{model_name}.pt"
        
        print(f"💾 Saving trained model to: {model_path}")
        try:
            # Handle DDP model unwrapping
            model_to_save = trained_model.module if hasattr(trained_model, 'module') else trained_model
            torch.save(model_to_save.state_dict(), model_path)
            print(f"✅ Model successfully saved to: {model_path}")
            
            # Verify the file was actually created
            if model_path.exists():
                file_size = model_path.stat().st_size
                print(f"✅ Model file verified: {file_size:,} bytes")
            else:
                print(f"❌ Model file was not created!")
                return 1
                
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            # Try to save to a fallback location
            fallback_path = Path(args.save_dir) / f"fallback_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            try:
                model_to_save = trained_model.module if hasattr(trained_model, 'module') else trained_model
                torch.save(model_to_save.state_dict(), fallback_path)
                print(f"🆘 Fallback model saved to: {fallback_path}")
            except Exception as e2:
                print(f"❌ Fallback save also failed: {e2}")
                return 1
        
        # Update config with training duration
        config_path = model_dir / f"{model_name}_config.json"
        try:
            if config_path.exists():
                # Load existing config and update it
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config_dict['training_duration'] = training_duration
                config_dict['model_parameters'] = sum(p.numel() for p in model_to_save.parameters())
                save_config_file(config_dict, config_path)
                print(f"✅ Config updated with training duration: {config_path}")
            else:
                # Create new config if it doesn't exist
                config_dict = {k.replace('_', '-'): v for k, v in vars(args).items() if v is not None}
                config_dict['training_duration'] = training_duration
                config_dict['model_parameters'] = sum(p.numel() for p in model_to_save.parameters())
                save_config_file(config_dict, config_path)
                print(f"✅ Config created: {config_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not update config file: {e}")
    else:
        if args.no_save:
            print("📝 Model saving disabled (--no-save flag)")
        elif args.ddp and args.rank != 0:
            print(f"📝 Skipping model save on non-main process (rank {args.rank})")
        else:
            print("📝 Model saving skipped for unknown reason")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 Training Session Complete")
    print("=" * 80)
    
    return 0
        
    # except KeyboardInterrupt:
    #     print("\n⚠️ Training interrupted by user")
    #     return 130
    # except Exception as e:
    #     print(f"❌ Training failed: {e}")
    #     if args.debug:
    #         import traceback
    #         traceback.print_exc()
    #     return 1
    # finally:
    #     # Cleanup DDP
    #     if args.ddp:
    #         cleanup_distributed()

if __name__ == "__main__":
    sys.exit(main())