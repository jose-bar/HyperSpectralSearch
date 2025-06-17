# checkpoint_manager.py
import os
import json
import pickle
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manage checkpoints for interrupted processing."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_prefix = None
        self.metadata = {}
        
    def initialize_checkpoint(self, job_name: str, total_files: int, 
                            config: Dict[str, Any]) -> str:
        """
        Initialize a new checkpoint session.
        
        Args:
            job_name: Name for this processing job
            total_files: Total number of files to process
            config: Configuration parameters
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"{job_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, checkpoint_id)
        
        self.metadata = {
            'checkpoint_id': checkpoint_id,
            'job_name': job_name,
            'total_files': total_files,
            'processed_files': [],
            'processed_batches': 0,
            'total_spectra': 0,
            'config': config,
            'status': 'in_progress',
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        self._save_metadata()
        logger.info(f"Initialized checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def save_batch_checkpoint(self, batch_idx: int, 
                            encoded_hvs: np.ndarray,
                            batch_meta_df: pd.DataFrame,
                            processed_files: list):
        """
        Save checkpoint for a processed batch.
        
        Args:
            batch_idx: Batch index
            encoded_hvs: Encoded hypervectors for this batch
            batch_meta_df: Metadata for this batch
            processed_files: List of files processed so far
        """
        # Save HVs
        hvs_path = f"{self.checkpoint_prefix}_batch_{batch_idx}_hvs.npy"
        np.save(hvs_path, encoded_hvs)
        
        # Save metadata
        meta_path = f"{self.checkpoint_prefix}_batch_{batch_idx}_meta.parquet"
        batch_meta_df.to_parquet(meta_path, compression='snappy')
        
        # Update checkpoint metadata
        self.metadata['processed_batches'] = batch_idx + 1
        self.metadata['total_spectra'] += len(batch_meta_df)
        self.metadata['processed_files'] = processed_files
        self.metadata['last_updated'] = datetime.now().isoformat()
        self._save_metadata()
        
        logger.info(f"Saved checkpoint for batch {batch_idx}")
    
    def load_checkpoint(self, checkpoint_id: str) -> Tuple[Optional[np.ndarray], 
                                                          Optional[pd.DataFrame], 
                                                          Dict[str, Any]]:
        """
        Load checkpoint data.
        
        Args:
            checkpoint_id: ID of checkpoint to load
            
        Returns:
            Tuple of (combined_hvs, combined_meta_df, metadata)
        """
        checkpoint_prefix = os.path.join(self.checkpoint_dir, checkpoint_id)
        metadata_path = f"{checkpoint_prefix}_metadata.json"
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load all batches
        all_hvs = []
        all_meta = []
        
        for batch_idx in range(metadata['processed_batches']):
            hvs_path = f"{checkpoint_prefix}_batch_{batch_idx}_hvs.npy"
            meta_path = f"{checkpoint_prefix}_batch_{batch_idx}_meta.parquet"
            
            if os.path.exists(hvs_path) and os.path.exists(meta_path):
                all_hvs.append(np.load(hvs_path))
                all_meta.append(pd.read_parquet(meta_path))
        
        if all_hvs and all_meta:
            combined_hvs = np.vstack(all_hvs)
            combined_meta = pd.concat(all_meta, ignore_index=True)
            logger.info(f"Loaded checkpoint {checkpoint_id}: "
                       f"{len(combined_meta)} spectra from {metadata['processed_batches']} batches")
            return combined_hvs, combined_meta, metadata
        else:
            return None, None, metadata
    
    def cleanup_checkpoint(self, checkpoint_id: str):
        """Remove checkpoint files after successful completion."""
        checkpoint_prefix = os.path.join(self.checkpoint_dir, checkpoint_id)
        
        # Remove batch files
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(os.path.basename(checkpoint_prefix)):
                os.remove(os.path.join(self.checkpoint_dir, file))
        
        logger.info(f"Cleaned up checkpoint {checkpoint_id}")
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        metadata_path = f"{self.checkpoint_prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('_metadata.json'):
                with open(os.path.join(self.checkpoint_dir, file), 'r') as f:
                    metadata = json.load(f)
                    checkpoints.append({
                        'id': metadata['checkpoint_id'],
                        'job_name': metadata['job_name'],
                        'status': metadata['status'],
                        'progress': f"{metadata['processed_batches']} batches, "
                                  f"{metadata['total_spectra']} spectra",
                        'created': metadata['created_at'],
                        'updated': metadata['last_updated']
                    })
        return checkpoints
