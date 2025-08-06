"""
Core module for the HyperSpectral Search Pipeline.

This module provides the main functionality for searching mass spectrometry data
using binary hypervectors and FAISS indexing.
"""

import numpy as np
import pandas as pd
import faiss
import os
import time
import logging
import glob
from typing import List, Dict, Tuple, Optional, Union, Any
import joblib
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpectraSearchPipeline:
    """Main pipeline for encoding mass spectra into hypervectors and searching."""
    
    def __init__(self, config=None, index_path=None, output_dir=None):
        """
        Initialize the search pipeline.
        
        Args:
            config: Configuration parameters for spectra processing
            index_path: Path to a saved FAISS index
            output_dir: Directory to save results
        """
        self.config = config or self._default_config()
        self.output_dir = output_dir
        self.index_path = index_path
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Initialize data structures
        self.spectra_meta_df = None
        self.spectra_hvs = None
        self.faiss_index = None
        self.lv_hvs = None
        self.id_hvs = None
        self.bin_len = None
        self.min_mz = None
        self.max_mz = None
    
    def _default_config(self):
        """Create default configuration parameters."""
        return {
            # Preprocessing parameters
            'min_peaks': 5,
            'min_mz_range': 250.0,
            'min_mz': 101.0,
            'max_mz': 1500.0,
            'remove_precursor_tol': 1.5,
            'min_intensity': 0.01,
            'max_peaks_used': 50,
            'scaling': 'off',
            
            # HD parameters
            'hd_dim': 2048,
            'hd_Q': 16,
            'hd_id_flip_factor': 2.0,
            'fragment_tol': 0.05,
            
            # Search parameters
            'precursor_tol': 0.05,
            'hamming_threshold': 205,
            'min_matched_peaks': 6
        }
    
    def preprocess_dataset(self, input_filepath):
        """
        Process dataset and encode into hypervectors.
        
        Args:
            input_filepath: Path to the MGF file(s)
        """
        logger.info(f"Processing dataset from {input_filepath}")
        
        # Load and process spectra
        spectra_list = self._load_mgf_files(input_filepath)
        
        # Process spectra
        processed_spectra, spectra_meta = self._process_spectra(spectra_list)
        
        # Convert to DataFrame
        self.spectra_meta_df = pd.DataFrame(spectra_meta)
        
        # Extract mz and intensity arrays
        spectra_mz = np.array([s[0] for s in processed_spectra], dtype=np.float32)
        spectra_intensity = np.array([s[1] for s in processed_spectra], dtype=np.float32)
        
        # Get dimension parameters
        self.bin_len, self.min_mz, self.max_mz = self._get_dim(
            self.config['min_mz'], self.config['max_mz'], self.config['fragment_tol'])
        
        # Generate LV-ID hypervectors
        self.lv_hvs, self.id_hvs = self._gen_lv_id_hvs(
            self.config['hd_dim'], self.config['hd_Q'], self.bin_len, self.config['hd_id_flip_factor'])
        
        # Encode spectra
        logger.info("Encoding spectra into hypervectors")
        self.spectra_hvs = self._encode_spectra(
            spectra_mz, spectra_intensity, self.bin_len)
        
        logger.info(f"Processed {len(self.spectra_meta_df)} spectra")
    
    def build_index(self, index_type='flat'):
        """
        Build FAISS binary index from hypervectors.
        
        Args:
            index_type: Type of FAISS index to build ('flat', 'ivf', 'hnsw')
        """
        logger.info(f"Building {index_type} FAISS index")
        
        if self.spectra_hvs is None:
            raise ValueError("No hypervectors available. Call preprocess_dataset() first.")
        
        # Convert hypervectors to FAISS binary format
        binary_vectors = self._convert_hv_to_faiss_binary(self.spectra_hvs)
        
        # Get the dimension in bits
        n_bytes = binary_vectors.shape[1]
        d_bits = n_bytes * 8
        
        # Create FAISS binary index based on the specified type
        if index_type == 'flat':
            self.faiss_index = faiss.IndexBinaryFlat(d_bits)
            
        elif index_type == 'ivf':
            # Create quantizer
            quantizer = faiss.IndexBinaryFlat(d_bits)
            
            # Number of clusters (voronoi cells)
            nlist = min(100, max(1, len(binary_vectors) // 100))
            
            # Create IVF index
            self.faiss_index = faiss.IndexBinaryIVF(quantizer, d_bits, nlist)
            
            # Need to train IVF index
            logger.info(f"Training IVF index with {len(binary_vectors)} vectors and {nlist} clusters")
            self.faiss_index.train(binary_vectors)
            
        elif index_type == 'hnsw':
            # Create HNSW index
            M = 16  # Number of connections per layer
            self.faiss_index = faiss.IndexBinaryHNSW(d_bits, M)
            
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add vectors to the index
        logger.info(f"Adding {len(binary_vectors)} vectors to index")
        self.faiss_index.add(binary_vectors)
        
        return self.faiss_index
    
    def save_index(self, prefix="spectra_index"):
        """
        Save the index and all necessary metadata for searching.
        
        Args:
            prefix: Prefix for saved files
            
        Returns:
            Path to the saved index directory
        """
        if not self.output_dir:
            raise ValueError("No output directory specified. Please set output_dir when initializing.")
        
        if self.faiss_index is None:
            raise ValueError("No index available. Call build_index() first.")
        
        # Create paths for saving
        index_path = os.path.join(self.output_dir, f"{prefix}.bin")
        metadata_path = os.path.join(self.output_dir, f"{prefix}_metadata.joblib")
        
        # Save the FAISS index
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index_binary(self.faiss_index, index_path)
        
        # Save metadata (DataFrame and hypervectors)
        logger.info(f"Saving metadata to {metadata_path}")
        metadata = {
            'spectra_meta_df': self.spectra_meta_df,
            'lv_hvs': self.lv_hvs,
            'id_hvs': self.id_hvs,
            'bin_len': self.bin_len,
            'min_mz': self.min_mz,
            'max_mz': self.max_mz,
            'config': self.config
        }
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Index and metadata saved to {self.output_dir}")
        return self.output_dir
    
    def load_index(self):
        """
        Load index and metadata from disk.
        """
        if not self.index_path:
            raise ValueError("No index path specified")
        
        # Define paths
        index_dir = os.path.dirname(self.index_path)
        prefix = os.path.basename(self.index_path).split('.')[0]
        metadata_path = os.path.join(index_dir, f"{prefix}_metadata.joblib")
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {self.index_path}")
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found at {self.index_path}")
        self.faiss_index = faiss.read_index_binary(self.index_path)
        
        # Load metadata
        logger.info(f"Loading metadata from {metadata_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        metadata = joblib.load(metadata_path)
        self.spectra_meta_df = metadata['spectra_meta_df']
        self.lv_hvs = metadata['lv_hvs']
        self.id_hvs = metadata['id_hvs']
        self.bin_len = metadata['bin_len']
        self.min_mz = metadata['min_mz']
        self.max_mz = metadata['max_mz']
        if 'config' in metadata:
            self.config = metadata['config']
        
        logger.info(f"Successfully loaded index with {len(self.spectra_meta_df)} spectra")
    
    def process_query_mgf(self, query_mgf_path, k=10, hamming_threshold=None, precursor_tol=None):
        """
        Process all spectra in a query MGF file and search for matches.
        
        Args:
            query_mgf_path: Path to query MGF file
            k: Number of results to return for each query
            hamming_threshold: Maximum Hamming distance for a match
            precursor_tol: Precursor mass tolerance in Da
                
        Returns:
            Dictionary with search results summary and details
        """
        logger.info(f"Processing query MGF file: {query_mgf_path}")
        
        # Use provided parameters or defaults from config
        hamming_threshold = hamming_threshold or self.config['hamming_threshold']
        precursor_tol = precursor_tol or self.config['precursor_tol']
        
        # Check if index is loaded
        if self.faiss_index is None:
            raise ValueError("No index loaded. Call load_index() first.")
        
        # Load query spectra
        query_spectra = self._load_mgf_file(query_mgf_path)
        logger.info(f"Loaded {len(query_spectra)} spectra from {query_mgf_path}")
        
        # Process each query
        results = []
        valid_spectra = 0
        invalid_spectra = 0
        
        # Initialize results summary
        results_summary = {
            'total_queries': 0,
            'total_matches': 0,
            'queries': []
        }
        
        for i, query_spectrum in enumerate(query_spectra):
            if i % 20 == 0 or i == len(query_spectra) - 1:
                logger.info(f"Processing query {i+1}/{len(query_spectra)}")
            
            # Process the query spectrum
            processed_spectrum = self._preprocess_spectrum(query_spectrum)
            if processed_spectrum:
                valid_spectra += 1
                
                # Search in the index
                query_result = self.search(processed_spectrum, query_spectrum, k)
                
                # Apply hamming distance and precursor mass filtering
                matches = []
                for match in query_result['results']:
                    if match['hamming_distance'] <= hamming_threshold and \
                       abs(match['precursor_mz'] - query_spectrum['precursor_mz']) <= precursor_tol:
                        matches.append(match)
                
                # Update results with filtered matches
                query_info = query_result['query']
                if matches:
                    results_summary['total_matches'] += len(matches)
                    results_summary['queries'].append({
                        'query_info': query_info,
                        'matches': matches
                    })
                else:
                    results_summary['queries'].append({
                        'query_info': query_info,
                        'matches': []
                    })
                
            else:
                invalid_spectra += 1
        
        results_summary['total_queries'] = valid_spectra
        logger.info(f"Query processing complete: {valid_spectra} valid spectra, {invalid_spectra} invalid spectra")
        logger.info(f"Total matches: {results_summary['total_matches']}")
        
        return results_summary
    
    def search(self, processed_spectrum, raw_spectrum, k=10):
        """
        Search for similar spectra to the query spectrum.
        
        Args:
            processed_spectrum: Processed query spectrum
            raw_spectrum: Raw query spectrum data
            k: Number of results to return
            
        Returns:
            Search results with query info and matches
        """
        # Encode the query spectrum
        query_hv = self._encode_single_spectrum(
            processed_spectrum[0], processed_spectrum[1])
        
        # Convert to FAISS binary format
        query_binary = self._convert_hv_to_faiss_binary(np.array([query_hv]))
        
        # Set search parameters for IVF index if applicable
        if hasattr(self.faiss_index, 'nprobe'):
            # For IVF index, set nprobe (number of clusters to visit)
            self.faiss_index.nprobe = 10
        
        # Search the index
        distances, indices = self.faiss_index.search(query_binary, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.spectra_meta_df):
                spectrum_info = self.spectra_meta_df.iloc[idx].to_dict()
                spectrum_info['hamming_distance'] = int(distances[0][i])
                results.append(spectrum_info)
        
        return {
            'query': {
                'precursor_mz': raw_spectrum['precursor_mz'],
                'precursor_charge': raw_spectrum['precursor_charge'],
                'identifier': raw_spectrum['identifier'],
                'scan': raw_spectrum['scan']
            },
            'results': results
        }

    # Internal helper methods
    def _load_mgf_files(self, input_filepath):
        """Load MGF files from the input filepath."""
        # Check if input is a directory or a file
        if os.path.isdir(input_filepath):
            mgf_files = glob.glob(os.path.join(input_filepath, '*.mgf'))
        else:
            mgf_files = [input_filepath]
        
        if not mgf_files:
            raise FileNotFoundError(f"No MGF files found in {input_filepath}")
        
        logger.info(f"Found {len(mgf_files)} MGF files")
        
        # Load all MGF files
        all_spectra = []
        for mgf_file in mgf_files:
            logger.info(f"Loading MGF file: {mgf_file}")
            spectra = self._load_mgf_file(mgf_file)
            all_spectra.extend(spectra)
        
        logger.info(f"Loaded {len(all_spectra)} spectra in total")
        return all_spectra
    
    def _load_mgf_file(self, mgf_file):
        """Load a single MGF file."""
        spectra = []
        current_spectrum = None
        
        with open(mgf_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line == "BEGIN IONS":
                    current_spectrum = {
                        'peaks': [],
                        'identifier': '',
                        'precursor_mz': 0.0,
                        'precursor_charge': 0,
                        'scan': 0,
                        'retention_time': 0.0,
                        'source_file': os.path.basename(mgf_file)
                    }
                
                elif line == "END IONS":
                    if current_spectrum:
                        spectra.append(current_spectrum)
                        current_spectrum = None
                
                elif current_spectrum is not None:
                    # Handle known metadata fields first
                    if line.startswith("TITLE="):
                        current_spectrum['identifier'] = line[6:]
                    elif line.startswith("PEPMASS="):
                        parts = line.split('=')[1].split()
                        current_spectrum['precursor_mz'] = float(parts[0])
                        if len(parts) > 1:
                            current_spectrum['precursor_intensity'] = float(parts[1])
                    elif line.startswith("CHARGE="):
                        charge_str = line.split('=')[1].strip()
                        current_spectrum['precursor_charge'] = int(charge_str.rstrip('+'))
                    elif line.startswith("SCANS="):
                        current_spectrum['scan'] = int(line.split('=')[1])
                    elif line.startswith("SCAN="):  # Handle both SCANS and SCAN
                        if current_spectrum['scan'] == 0:  # Only use if SCANS wasn't set
                            current_spectrum['scan'] = int(line.split('=')[1])
                    elif line.startswith("RTINSECONDS="):
                        current_spectrum['retention_time'] = float(line.split('=')[1])
                    
                    # Handle other common metadata fields (ignore them)
                    elif any(line.startswith(prefix) for prefix in [
                        "MSLEVEL=", "COLLISION_ENERGY=", "FILENAME=", "SEQ=", 
                        "PROTEIN=", "SCORE=", "FDR=", "PROVENANCE_", "DATASET="
                    ]):
                        # Skip these metadata lines
                        continue
                    
                    # Try to parse as peak data (m/z intensity pairs)
                    elif ' ' in line or '\t' in line:
                        # Split by whitespace (handles both spaces and tabs)
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                # Try to parse the first two parts as numbers
                                mz = float(parts[0])
                                intensity = float(parts[1])
                                current_spectrum['peaks'].append((mz, intensity))
                            except ValueError:
                                # If parsing fails, it's probably metadata we don't recognize
                                # Log it for debugging but don't crash
                                logger.debug(f"Could not parse as peak data: {line}")
                                continue
        
        logger.info(f"Loaded {len(spectra)} spectra from {mgf_file}")
        return spectra

    
    def _process_spectra(self, spectra_list):
        """Process a list of spectra."""
        processed_spectra = []
        spectra_meta = []
        
        for spectrum in spectra_list:
            processed_spectrum = self._preprocess_spectrum(spectrum)
            
            if processed_spectrum:
                processed_spectra.append(processed_spectrum)
                
                spectra_meta.append({
                    'precursor_mz': spectrum['precursor_mz'],
                    'precursor_charge': spectrum['precursor_charge'],
                    'identifier': spectrum['identifier'],
                    'scan': spectrum['scan'],
                    'retention_time': spectrum['retention_time'],
                    'source_file': spectrum.get('source_file', '')
                })
        
        logger.info(f"Processed {len(processed_spectra)} valid spectra")
        return processed_spectra, spectra_meta
    
    def _preprocess_spectrum(self, spectrum):
        """Preprocess a single spectrum."""
        # Check if spectrum has peaks
        if not spectrum['peaks']:
            logger.debug(f"Spectrum {spectrum['identifier']} has no peaks")
            return None
        
        # Extract peaks
        peaks = sorted(spectrum['peaks'], key=lambda x: x[0])
        mz = np.array([p[0] for p in peaks], dtype=np.float32)
        intensity = np.array([p[1] for p in peaks], dtype=np.float32)
        
        # Filter by m/z range
        valid_idx = (mz >= self.config['min_mz']) & (mz <= self.config['max_mz'])
        mz = mz[valid_idx]
        intensity = intensity[valid_idx]
        
        if len(mz) < self.config['min_peaks']:
            logger.debug(f"Spectrum {spectrum['identifier']} has fewer than {self.config['min_peaks']} peaks after m/z filtering")
            return None
        
        # Check m/z range
        mz_range = mz[-1] - mz[0]
        if mz_range < self.config['min_mz_range']:
            logger.debug(f"Spectrum {spectrum['identifier']} has m/z range {mz_range} < {self.config['min_mz_range']}")
            return None
        
        # Remove precursor peak
        if self.config['remove_precursor_tol'] > 0:
            precursor_mz = spectrum['precursor_mz']
            valid_idx = np.abs(mz - precursor_mz) > self.config['remove_precursor_tol']
            mz = mz[valid_idx]
            intensity = intensity[valid_idx]
        
        if len(mz) < self.config['min_peaks']:
            logger.debug(f"Spectrum {spectrum['identifier']} has fewer than {self.config['min_peaks']} peaks after precursor removal")
            return None
        
        # Normalize intensities
        max_intensity = np.max(intensity)
        intensity = intensity / max_intensity
        
        # Filter by intensity
        valid_idx = intensity >= self.config['min_intensity']
        mz = mz[valid_idx]
        intensity = intensity[valid_idx]
        
        if len(mz) < self.config['min_peaks']:
            logger.debug(f"Spectrum {spectrum['identifier']} has fewer than {self.config['min_peaks']} peaks after intensity filtering")
            return None
        
        # Keep top N peaks
        if self.config['max_peaks_used'] > 0 and len(mz) > self.config['max_peaks_used']:
            top_idx = np.argsort(-intensity)[:self.config['max_peaks_used']]
            mz = mz[top_idx]
            intensity = intensity[top_idx]
        
        # Apply scaling
        if self.config['scaling'] == 'root':
            intensity = np.sqrt(intensity)
        elif self.config['scaling'] == 'log':
            intensity = np.log1p(intensity) / np.log(2)
        
        # Normalize again
        intensity = intensity / np.linalg.norm(intensity)
        
        # Quantize mz to bin indices
        bin_indices = np.floor((mz - self.config['min_mz']) / self.config['fragment_tol']).astype(np.int32)
        
        # Pad to fixed size
        if self.config['max_peaks_used'] > 0:
            padded_mz = np.ones(self.config['max_peaks_used'], dtype=np.int32) * -1
            padded_intensity = np.ones(self.config['max_peaks_used'], dtype=np.float32) * -1
            
            size = min(len(bin_indices), self.config['max_peaks_used'])
            padded_mz[:size] = bin_indices[:size]
            padded_intensity[:size] = intensity[:size]
            
            return padded_mz, padded_intensity
        else:
            return bin_indices, intensity
    
    def _get_dim(self, min_mz, max_mz, bin_size):
        """Compute the number of bins over the given mass range for the given bin size."""
        start_dim = min_mz - min_mz % bin_size
        end_dim = max_mz + bin_size - max_mz % bin_size
        return int(round((end_dim - start_dim) / bin_size)), start_dim, end_dim
    
    def _gen_lvs(self, D, Q):
        """Generate level vectors."""
        base = np.ones(D, dtype=np.float32)
        base[:D//2] = -1.0
        l0 = np.random.permutation(base)
        levels = []
        for i in range(Q+1):
            flip = int(int(i/float(Q) * D) / 2)
            li = np.copy(l0)
            li[:flip] = l0[:flip] * -1
            levels.append(li)
        return np.array(levels, dtype=np.float32)
    
    def _gen_idhvs(self, D, totalFeatures, flip_factor):
        """Generate ID hypervectors."""
        nFlip = int(D//flip_factor)
        mu, sigma = 0, 1
        bases = np.random.normal(mu, sigma, D)
        
        generated_hvs = [bases.copy()]
        for _ in range(totalFeatures-1):
            idx_to_flip = np.random.randint(0, D, size=nFlip)
            bases[idx_to_flip] *= (-1)
            generated_hvs.append(bases.copy())
        
        return np.array(generated_hvs, dtype=np.float32)
    
    def _gen_lv_id_hvs(self, D, Q, bin_len, id_flip_factor):
        """Generate level and ID hypervectors."""
        lv_id_hvs_file = f'lv_id_hvs_D_{D}_Q_{Q}_bin_{bin_len}_flip_{id_flip_factor}.npz'
        if os.path.exists(lv_id_hvs_file):
            logger.info(f"Loading existing {lv_id_hvs_file} file for HD")
            data = np.load(lv_id_hvs_file)
            lv_hvs, id_hvs = data['lv_hvs'], data['id_hvs']
        else:
            logger.info("Generating new level and ID hypervectors")
            lv_hvs = self._gen_lvs(D, Q)
            id_hvs = self._gen_idhvs(D, bin_len, id_flip_factor)
            np.savez(lv_id_hvs_file, lv_hvs=lv_hvs, id_hvs=id_hvs)
        
        return lv_hvs, id_hvs
    
    def _encode_single_spectrum(self, mz, intensity):
        """Encode a single spectrum into a hypervector."""
        D = self.config['hd_dim']
        Q = self.config['hd_Q']
        
        # Initialize output vector
        encoded_hv = np.zeros(D, dtype=np.float32)
        
        # For each peak, combine level and ID vectors
        for i in range(len(mz)):
            if mz[i] != -1 and intensity[i] != -1:
                bin_idx = int(mz[i])
                if bin_idx < self.bin_len:
                    # Get level index (quantize intensity)
                    level_idx = min(int(intensity[i] * Q), Q)
                    
                    # Add contribution from this peak
                    encoded_hv += self.lv_hvs[level_idx] * self.id_hvs[bin_idx]
        
        # Binarize using majority rule
        binary_hv = np.ones(D, dtype=np.float32)
        binary_hv[encoded_hv <= 0] = -1
        
        # Convert to binary
        final_hv = np.zeros(D, dtype=np.float32)
        final_hv[binary_hv > 0] = 1
        
        # Pack into uint32
        packed_hv = self._bit_packing(final_hv.reshape(1, -1), 1, D)[0]
        
        return packed_hv
    
    def _encode_spectra(self, spectra_mz, spectra_intensity, bin_len, batch_size=1000):
        """Encode multiple spectra into hypervectors."""
        num_spectra = spectra_mz.shape[0]
        D = self.config['hd_dim']
        pack_len = (D + 32 - 1) // 32
        
        # Initialize output array
        encoded_spectra = np.zeros((num_spectra, pack_len), dtype=np.uint32)
        
        # Process in batches
        for i in range(0, num_spectra, batch_size):
            if i % (10 * batch_size) == 0:
                logger.info(f"Encoding batch {i//batch_size + 1}/{(num_spectra+batch_size-1)//batch_size}")
            
            end = min(i + batch_size, num_spectra)
            batch_size_actual = end - i
            
            for j in range(batch_size_actual):
                # Fix: Remove bin_len parameter here
                encoded_spectra[i+j] = self._encode_single_spectrum(
                    spectra_mz[i+j], spectra_intensity[i+j]
                )
        
        return encoded_spectra
    
    def _bit_packing(self, vecs, N, D):
        """Vectorized bit packing using numpy operations."""
        pack_len = (D + 32 - 1) // 32
        packed_vecs = np.zeros((N, pack_len), dtype=np.uint32)
        
        # Reshape for efficient processing
        vecs_reshaped = vecs.reshape(N, -1, 32) if D % 32 == 0 else \
                        np.pad(vecs, ((0, 0), (0, 32 - D % 32)), constant_values=0).reshape(N, -1, 32)
        
        # Vectorized bit packing
        for i in range(vecs_reshaped.shape[1]):
            bits = vecs_reshaped[:, i, :]
            powers = 2 ** np.arange(31, -1, -1, dtype=np.uint32)
            packed_vecs[:, i] = np.dot(bits, powers)
        
        return packed_vecs[:, :pack_len]
    
    def _convert_hv_to_faiss_binary(self, hypervectors):
        """Convert hypervectors to FAISS binary format."""
        n_vectors, n_uint32 = hypervectors.shape
        D = n_uint32 * 32  # Original hypervector dimension
        n_bytes = (D + 7) // 8  # Number of bytes needed for FAISS
        
        binary_vectors = np.zeros((n_vectors, n_bytes), dtype=np.uint8)
        
        for i in range(n_vectors):
            for j in range(n_uint32):
                val = hypervectors[i, j]
                byte_offset = j * 4  # Each uint32 gives 4 bytes
                
                # Extract the 4 bytes from each uint32
                for b in range(min(4, n_bytes - byte_offset)):
                    binary_vectors[i, byte_offset + b] = (val >> (b * 8)) & 0xFF
        
        return binary_vectors
    
    def preprocess_dataset_streaming(self, input_filepath, batch_size=1000, max_memory_gb=4.0):
        """
        Process dataset using memory-efficient streaming.
        
        Args:
            input_filepath: Path to MGF file(s)
            batch_size: Number of spectra per batch
            max_memory_gb: Maximum memory to use in GB
        """
        from data_loader import SpectraDataLoader
        
        logger.info(f"Processing dataset from {input_filepath} using streaming")
        
        # Initialize data loader
        loader = SpectraDataLoader(
            batch_size=batch_size,
            max_memory_gb=max_memory_gb,
            verbose=True
        )
        
        # Get dimension parameters first
        self.bin_len, self.min_mz, self.max_mz = self._get_dim(
            self.config['min_mz'], self.config['max_mz'], self.config['fragment_tol'])
        
        # Generate LV-ID hypervectors
        self.lv_hvs, self.id_hvs = self._gen_lv_id_hvs(
            self.config['hd_dim'], self.config['hd_Q'], self.bin_len, self.config['hd_id_flip_factor'])
        
        # Process in batches
        all_hvs = []
        all_meta = []
        
        for encoded_hvs, batch_indices, batch_meta_df in loader.load_and_process_dataset(input_filepath, self):
            all_hvs.append(encoded_hvs)
            all_meta.append(batch_meta_df)
            
            # Optional: save intermediate results to disk if needed
            if len(all_hvs) % 10 == 0:
                logger.info(f"Processed {len(all_meta)} batches")
        
        # Combine all results
        self.spectra_hvs = np.vstack(all_hvs)
        self.spectra_meta_df = pd.concat(all_meta, ignore_index=True)
        
        logger.info(f"Processed {len(self.spectra_meta_df)} spectra total")

    def build_index_streaming(self, index_type='flat', chunk_size=100000):
        """
        Build FAISS index with streaming to handle large datasets.
        
        Args:
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            chunk_size: Number of vectors to add at a time
        """
        logger.info(f"Building {index_type} FAISS index with streaming")
        
        if self.spectra_hvs is None:
            raise ValueError("No hypervectors available. Call preprocess_dataset() first.")
        
        # Convert hypervectors to FAISS binary format
        binary_vectors = self._convert_hv_to_faiss_binary(self.spectra_hvs)
        
        # Get the dimension in bits
        n_bytes = binary_vectors.shape[1]
        d_bits = n_bytes * 8
        
        # Create FAISS binary index
        if index_type == 'flat':
            self.faiss_index = faiss.IndexBinaryFlat(d_bits)
        elif index_type == 'ivf':
            quantizer = faiss.IndexBinaryFlat(d_bits)
            nlist = min(100, max(1, len(binary_vectors) // 100))
            self.faiss_index = faiss.IndexBinaryIVF(quantizer, d_bits, nlist)
            
            # Train IVF index with a subset
            train_size = min(100000, len(binary_vectors))
            logger.info(f"Training IVF index with {train_size} vectors")
            self.faiss_index.train(binary_vectors[:train_size])
        elif index_type == 'hnsw':
            # Create HNSW index
            M = 16  # Number of connections per layer
            self.faiss_index = faiss.IndexBinaryHNSW(d_bits, M)
            # HNSW doesn't support batch adding for binary indices bruh
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add vectors in chunks to avoid memory issues
        n_vectors = len(binary_vectors)
        for i in range(0, n_vectors, chunk_size):
            end = min(i + chunk_size, n_vectors)
            logger.info(f"Adding vectors {i} to {end}")
            self.faiss_index.add(binary_vectors[i:end])
        
        logger.info(f"Index built with {self.faiss_index.ntotal} vectors")
        
    def preprocess_dataset_streaming_with_checkpoint(
        self, 
        input_filepath, 
        batch_size=1000, 
        max_memory_gb=4.0,
        enable_checkpointing=True,
        resume_checkpoint_id=None
    ):
        """
        Process dataset using memory-efficient streaming with checkpoint support.
        
        Args:
            input_filepath: Path to MGF file(s)
            batch_size: Number of spectra per batch
            max_memory_gb: Maximum memory to use in GB
            enable_checkpointing: Whether to save checkpoints
            resume_checkpoint_id: Checkpoint ID to resume from
        """
        from data_loader import SpectraDataLoader
        
        logger.info(f"Processing dataset from {input_filepath} using streaming with checkpointing")
        
        # Initialize data loader
        loader = SpectraDataLoader(
            batch_size=batch_size,
            max_memory_gb=max_memory_gb,
            verbose=True,
            enable_checkpointing=enable_checkpointing
        )
        
        # Get dimension parameters first
        self.bin_len, self.min_mz, self.max_mz = self._get_dim(
            self.config['min_mz'], self.config['max_mz'], self.config['fragment_tol'])
        
        # Generate LV-ID hypervectors
        self.lv_hvs, self.id_hvs = self._gen_lv_id_hvs(
            self.config['hd_dim'], self.config['hd_Q'], self.bin_len, self.config['hd_id_flip_factor'])
        
        # Check if resuming from checkpoint
        if resume_checkpoint_id:
            logger.info(f"Attempting to resume from checkpoint: {resume_checkpoint_id}")
            # Load existing data
            existing_hvs, existing_meta, _ = loader.checkpoint_manager.load_checkpoint(resume_checkpoint_id)
            if existing_hvs is not None:
                all_hvs = [existing_hvs]
                all_meta = [existing_meta]
            else:
                all_hvs = []
                all_meta = []
        else:
            all_hvs = []
            all_meta = []
        
        # Process in batches
        for encoded_hvs, batch_indices, batch_meta_df in loader.load_and_process_dataset_with_checkpointing(
            input_filepath, self, resume_checkpoint_id=resume_checkpoint_id
        ):
            all_hvs.append(encoded_hvs)
            all_meta.append(batch_meta_df)
        
        # Combine all results
        self.spectra_hvs = np.vstack(all_hvs)
        self.spectra_meta_df = pd.concat(all_meta, ignore_index=True)
        
        logger.info(f"Processed {len(self.spectra_meta_df)} spectra total")