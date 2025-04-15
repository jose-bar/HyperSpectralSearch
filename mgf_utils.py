"""
Utility functions for working with MGF files and mass spectrometry data.
"""

import os
import logging
import glob
from typing import List, Dict, Tuple, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_mgf_file(mgf_file: str) -> List[Dict[str, Any]]:
    """
    Load a single MGF file.
    
    Args:
        mgf_file: Path to the MGF file
        
    Returns:
        List of spectra dictionaries
    """
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
                elif line.startswith("RTINSECONDS="):
                    current_spectrum['retention_time'] = float(line.split('=')[1])
                elif ' ' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        mz, intensity = float(parts[0]), float(parts[1])
                        current_spectrum['peaks'].append((mz, intensity))
    
    logger.info(f"Loaded {len(spectra)} spectra from {mgf_file}")
    return spectra

def load_mgf_files(input_filepath: str) -> List[Dict[str, Any]]:
    """
    Load MGF files from the input filepath.
    
    Args:
        input_filepath: Path to the MGF file(s)
        
    Returns:
        List of spectra dictionaries
    """
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
        spectra = load_mgf_file(mgf_file)
        all_spectra.extend(spectra)
    
    logger.info(f"Loaded {len(all_spectra)} spectra in total")
    return all_spectra

def write_mgf_file(spectra: List[Dict[str, Any]], output_file: str) -> None:
    """
    Write spectra to an MGF file.
    
    Args:
        spectra: List of spectrum dictionaries
        output_file: Path to output MGF file
    """
    with open(output_file, 'w') as f:
        for spectrum in spectra:
            f.write("BEGIN IONS\n")
            
            # Write metadata
            f.write(f"TITLE={spectrum.get('identifier', '')}\n")
            f.write(f"PEPMASS={spectrum.get('precursor_mz', 0.0)}")
            if 'precursor_intensity' in spectrum:
                f.write(f" {spectrum['precursor_intensity']}")
            f.write("\n")
            
            if 'precursor_charge' in spectrum:
                f.write(f"CHARGE={spectrum['precursor_charge']}+\n")
            
            if 'scan' in spectrum:
                f.write(f"SCANS={spectrum['scan']}\n")
            
            if 'retention_time' in spectrum:
                f.write(f"RTINSECONDS={spectrum['retention_time']}\n")
            
            # Write peaks
            for mz, intensity in spectrum.get('peaks', []):
                f.write(f"{mz} {intensity}\n")
            
            f.write("END IONS\n\n")
    
    logger.info(f"Wrote {len(spectra)} spectra to {output_file}")

def filter_spectra_by_precursor(spectra: List[Dict[str, Any]], 
                               min_mz: Optional[float] = None, 
                               max_mz: Optional[float] = None,
                               min_charge: Optional[int] = None,
                               max_charge: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Filter spectra by precursor m/z and charge.
    
    Args:
        spectra: List of spectrum dictionaries
        min_mz: Minimum precursor m/z
        max_mz: Maximum precursor m/z
        min_charge: Minimum precursor charge
        max_charge: Maximum precursor charge
        
    Returns:
        Filtered list of spectra
    """
    filtered = []
    for spectrum in spectra:
        mz = spectrum.get('precursor_mz', 0.0)
        charge = spectrum.get('precursor_charge', 0)
        
        if min_mz is not None and mz < min_mz:
            continue
        if max_mz is not None and mz > max_mz:
            continue
        if min_charge is not None and charge < min_charge:
            continue
        if max_charge is not None and charge > max_charge:
            continue
        
        filtered.append(spectrum)
    
    logger.info(f"Filtered {len(spectra)} spectra to {len(filtered)} spectra")
    return filtered

def merge_mgf_files(input_files: List[str], output_file: str) -> None:
    """
    Merge multiple MGF files into a single MGF file.
    
    Args:
        input_files: List of input MGF files
        output_file: Path to output MGF file
    """
    all_spectra = []
    for input_file in input_files:
        spectra = load_mgf_file(input_file)
        all_spectra.extend(spectra)
    
    write_mgf_file(all_spectra, output_file)
    logger.info(f"Merged {len(input_files)} MGF files with {len(all_spectra)} spectra to {output_file}")