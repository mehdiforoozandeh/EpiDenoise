# Implementation Plan

- [x] 1. Set up project structure and initialize CANDI_TRAINER skeleton
  - Create train.py and define CANDI_TRAINER class with __init__ method to initialize model, optimizer, scheduler, and dataset.
  - defaults: optimizer=adamax, learning rate=1e-3, context length=1200, scheduler=cosine annealing, epochs=10.
  - Add basic train() method stub that sets up DataLoader.
  - _Requirements: 1.1, 2.1, 3.1_

- [x] 2. Implement setup and DDP support
  - Add _setup() to configure logging, optional validation, and wrap model in DDP if multi-GPU.
  - Write unit tests for single vs multi-GPU initialization.
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 3. Integrate data loading with CANDIIterableDataset
  - In __init__, instantiate CANDIIterableDataset with params.
  - In train(), create DataLoader and iterate over it, yielding batches.
  - Add error handling for loading failures.
  - Write integration tests with small EIC dataset subset.
  - _Requirements: 2.1, 2.2, 4.1_

- [x] 4. Implement batch processing and loss computation
  - Add _process_batch() for masking, forward pass, loss calc using CANDI_LOSS.
  - Handle NaN losses by skipping backward pass.
  - Write unit tests with sample batches.
  - _Requirements: 1.1, 1.3, 4.1, 4.2_

- [x] 5. Add metrics computation and LR adjustment
  - Implement _compute_metrics() for R2, Spearman correlation, Pearson correlation, etc.
  - Add _adjust_lr() for progress monitoring and learning rate scheduling.
  - Handle tensor shapes: batch input (B, L, F) → outputs (B, L, F) where B=batch_size, L=sequence_length, F=num_features.
  - Compute metrics per feature: calculate R2, correlations independently for each of the F features.
  - Aggregate and log: print IQR, median, min, max metrics across all F features per batch for monitoring.
  - also print EMA of metrics for better tracking of progress.
  - Integrate metric computation and LR adjustment into main train() loop.
  - _Requirements: 1.2, 4.2_

- [x] 6. Implement optional validation
  - Add _validate() that runs if enabled, computes metrics on validation set.
  - Handle failures with fallbacks.
  - Add to train() at epoch ends.
  - _Requirements: 1.1, 4.1, 4.2_

- [x] 7. Final integration and regression testing
  - Wire all helpers into train() loop.
  - Add mixed precision support and OOM handling.
  - Run full training on EIC subset, compare with original PRETRAIN.
  - _Requirements: 1.1, 2.2, 3.2, 4.1_

- [x] 8. Create improved CLI interface for CANDI training
  - Design and implement a modern CLI using the refactored CANDI_TRAINER.
  - Improve upon the old CLI with better argument organization, validation, and help text.
  - Add new features: mixed precision toggle, DDP support, validation options, model saving/loading.
  - Support all training configurations: EIC/merged datasets, different optimizers, schedulers, architectures.
  - Include comprehensive error handling and user-friendly output.
  - Add configuration file support for complex training setups.
  - _Requirements: 1.1, 2.1, 3.1_

- [x] 9. Implement clean and beautiful training logs with verbosity levels
  - Fix warnings in the codebase by addressing root causes.
  - Remove all batch pointer update prints from data.py that clutter the output.
  - Create 3 verbosity levels (0, 1, 2) with aesthetic and organized logging:
    - **Verbosity 0**: Only show tqdm progress bar for epoch with fraction of batches processed + imp_pval_r2 metric
    - **Verbosity 1**: Show epoch progress bar + EMA of metrics (median, IQR, min, max) every 5 batches
    - **Verbosity 2**: Show epoch progress bar + current batch metrics (median, IQR, min, max) + EMA metrics after each batch
  - Implement TrainingLogger class to handle all logging with clean formatting and colors.
  - Integrate tqdm for progress tracking with custom format and metric display.
  - Add EMA tracking for all metric aggregations (median, IQR, min, max) across metric types.
  - Ensure logging works properly in both single-GPU and DDP modes.
  - Add --verbosity/-v CLI argument to control logging level.
  - _Requirements: 1.2, 4.2_
