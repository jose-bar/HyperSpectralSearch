"""
Memory-efficient data loader for mass spectrometry data.
"""

import os
import numpy as np
import pandas as pd
from typing import Iterator, Dict, Any, List, Optional, Tuple
import logging
from collections import deque
import gc

from checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

class SpectraDataLoader:
    """Memory-efficient data loader with checkpoint support."""
    
    def __init__(self, 
                 batch_size: int = 1000,
                 max_memory_gb: float = 4.0,
                 verbose: bool = True,
                 checkpoint_dir: str = "./checkpoints",
                 enable_checkpointing: bool = True):
        """
        Initialize the data loader.
        
        Args:
            batch_size: Number of spectra per batch
            max_memory_gb: Maximum memory usage in GB
            verbose: Whether to print progress
            checkpoint_dir: Directory for checkpoints
            enable_checkpointing: Whether to enable checkpointing
        """
        self.batch_size = batch_size
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.verbose = verbose
        self.enable_checkpointing = enable_checkpointing
        self._memory_usage = 0
        
        if enable_checkpointing:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        else:
            self.checkpoint_manager = None
    
    def load_and_process_dataset_with_checkpointing(
        self, 
        input_filepath: str, 
        pipeline,
        job_name: str = "spectral_search",
        resume_checkpoint_id: str = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
        """
        Load and process dataset with checkpoint support.
        
        Args:
            input_filepath: Path to MGF file(s)
            pipeline: SpectraSearchPipeline instance
            job_name: Name for this processing job
            resume_checkpoint_id: ID of checkpoint to resume from
            
        Yields:
            Tuple of (encoded_hvs, batch_indices, batch_meta_df)
        """
        import glob
        
        # Get list of MGF files
        if os.path.isdir(input_filepath):
            mgf_files = sorted(glob.glob(os.path.join(input_filepath, '*.mgf')))
        else:
            mgf_files = [input_filepath]
        
        if not mgf_files:
            raise FileNotFoundError(f"No MGF files found in {input_filepath}")
        
        # Handle checkpoint resumption
        start_batch_idx = 0
        total_spectra = 0
        processed_files = []
        
        if resume_checkpoint_id and self.checkpoint_manager:
            # Load existing checkpoint
            _, _, metadata = self.checkpoint_manager.load_checkpoint(resume_checkpoint_id)
            processed_files = metadata['processed_files']
            start_batch_idx = metadata['processed_batches']
            total_spectra = metadata['total_spectra']
            
            # Filter out already processed files
            mgf_files = [f for f in mgf_files if f not in processed_files]
            
            logger.info(f"Resuming from checkpoint {resume_checkpoint_id}")
            logger.info(f"Already processed: {start_batch_idx} batches, {total_spectra} spectra")
            logger.info(f"Remaining files: {len(mgf_files)}")
            
            # Set checkpoint prefix for continued saving
            self.checkpoint_manager.checkpoint_prefix = os.path.join(
                self.checkpoint_manager.checkpoint_dir, resume_checkpoint_id
            )
            self.checkpoint_manager.metadata = metadata
        
        elif self.checkpoint_manager:
            # Initialize new checkpoint
            checkpoint_id = self.checkpoint_manager.initialize_checkpoint(
                job_name, len(mgf_files), pipeline.config
            )
            logger.info(f"Created new checkpoint: {checkpoint_id}")
        
        # Process files
        batch = []
        batch_count = start_batch_idx
        
        for mgf_file in mgf_files:
            if self.verbose:
                logger.info(f"Processing file: {mgf_file}")
            
            for spectrum in self.load_mgf_lazy(mgf_file):
                batch.append(spectrum)
                
                if len(batch) >= self.batch_size:
                    # Process batch
                    processed_spectra, spectra_meta = self.process_batch(batch, pipeline)
                    
                    if processed_spectra:
                        # Convert to arrays
                        spectra_mz = np.array([s[0] for s in processed_spectra], dtype=np.float32)
                        spectra_intensity = np.array([s[1] for s in processed_spectra], dtype=np.float32)
                        
                        # Encode spectra
                        encoded_hvs = pipeline._encode_spectra(
                            spectra_mz, spectra_intensity, 
                            pipeline.bin_len, batch_size=len(processed_spectra)
                        )
                        
                        # Create DataFrame for metadata
                        batch_meta_df = pd.DataFrame(spectra_meta)
                        
                        # Track indices for this batch
                        batch_indices = np.arange(total_spectra, total_spectra + len(batch_meta_df))
                        total_spectra += len(batch_meta_df)
                        
                        # Save checkpoint if enabled
                        if self.checkpoint_manager:
                            processed_files.append(mgf_file)
                            self.checkpoint_manager.save_batch_checkpoint(
                                batch_count, encoded_hvs, batch_meta_df, 
                                list(set(processed_files))
                            )
                        
                        yield encoded_hvs, batch_indices, batch_meta_df
                        
                        batch_count += 1
                        if self.verbose and batch_count % 10 == 0:
                            logger.info(f"Processed {batch_count} batches, {total_spectra} spectra")
                    
                    # Clear batch and force garbage collection
                    batch = []
                    gc.collect()
        
        # Process remaining spectra
        if batch:
            processed_spectra, spectra_meta = self.process_batch(batch, pipeline)
            
            if processed_spectra:
                spectra_mz = np.array([s[0] for s in processed_spectra], dtype=np.float32)
                spectra_intensity = np.array([s[1] for s in processed_spectra], dtype=np.float32)
                
                encoded_hvs = pipeline._encode_spectra(
                    spectra_mz, spectra_intensity, 
                    pipeline.bin_len, batch_size=len(processed_spectra)
                )
                
                batch_meta_df = pd.DataFrame(spectra_meta)
                batch_indices = np.arange(total_spectra, total_spectra + len(batch_meta_df))
                
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_batch_checkpoint(
                        batch_count, encoded_hvs, batch_meta_df, mgf_files
                    )
                
                yield encoded_hvs, batch_indices, batch_meta_df
        
        # Mark checkpoint as complete
        if self.checkpoint_manager:
            self.checkpoint_manager.metadata['status'] = 'completed'
            self.checkpoint_manager._save_metadata()
        
        logger.info(f"Finished processing. Total spectra: {total_spectra + len(batch_meta_df)}")
