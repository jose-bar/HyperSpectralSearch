"""
SPTXT file loader for HyperSpectral Search Pipeline.

This module provides functionality to load and parse SPTXT spectral library files
and convert them to the internal format used by HyperSpectral.
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional

# Set up logging
logger = logging.getLogger(__name__)


class SPTXTLoader:
    """Loader for SPTXT spectral library files."""
    
    @staticmethod
    def parse_comment_field(comment: str) -> Dict[str, Any]:
        """
        Parse the comment field to extract metadata.
        
        Args:
            comment: Comment string from SPTXT entry
            
        Returns:
            Dictionary of parsed metadata
        """
        metadata = {}
        
        # Use regex to find key=value pairs
        # This handles various formats including quoted values
        pattern = r'(\w+)=([^\s]+(?:\s+[^\s=]+)*?)(?=\s+\w+=|$)'
        matches = re.findall(pattern, comment)
        
        for key, value in matches:
            # Clean up the value
            value = value.strip()
            
            # Try to convert to appropriate type
            if key in ['NAA', 'NMC', 'NTT', 'NumPeaks']:
                try:
                    metadata[key] = int(value)
                except ValueError:
                    metadata[key] = value
            elif key in ['Prob', 'FracUnassigned', 'Parent', 'AvePrecursorMz']:
                try:
                    metadata[key] = float(value)
                except ValueError:
                    metadata[key] = value
            else:
                metadata[key] = value
        
        return metadata
    
    @staticmethod
    def extract_sequence_and_modifications(full_name: str) -> Tuple[str, Dict[int, str]]:
        """
        Extract peptide sequence and modifications from the full name.
        
        Args:
            full_name: Full name field from SPTXT (e.g., "EGAC[Carbamidomethyl]PGLC[Carbamidomethyl]NSNGR/2 (HCD)")
            
        Returns:
            Tuple of (clean_sequence, modifications_dict)
        """
        # Remove fragmentation info (e.g., " (HCD)")
        if ' (' in full_name:
            full_name = full_name.split(' (')[0]
        
        # Extract charge if present
        if '/' in full_name:
            sequence_part = full_name.split('/')[0]
        else:
            sequence_part = full_name
        
        # Parse modifications
        modifications = {}
        clean_sequence = ""
        
        # Pattern to match modifications like C[Carbamidomethyl] or M[Oxidation]
        mod_pattern = r'([A-Z])\[([^\]]+)\]'
        
        # Find all modifications and their positions
        current_pos = 0
        last_end = 0
        
        for match in re.finditer(mod_pattern, sequence_part):
            # Add unmodified sequence before this modification
            clean_sequence += sequence_part[last_end:match.start()]
            
            # Add the modified amino acid
            aa = match.group(1)
            mod = match.group(2)
            clean_sequence += aa
            
            # Store modification position (1-based)
            position = len(clean_sequence)
            modifications[position] = f"{aa}[{mod}]"
            
            last_end = match.end()
        
        # Add any remaining sequence
        clean_sequence += sequence_part[last_end:]
        
        return clean_sequence, modifications
    
    @staticmethod
    def load_sptxt_file(filepath: str, include_annotations: bool = False) -> List[Dict[str, Any]]:
        """
        Load and parse an SPTXT file.
        
        Args:
            filepath: Path to the SPTXT file
            include_annotations: Whether to include peak annotations
            
        Returns:
            List of spectrum dictionaries compatible with HyperSpectral format
        """
        spectra = []
        current_spectrum = None
        reading_peaks = False
        
        logger.info(f"Loading SPTXT file: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check for new entry
                if line.startswith('Name:'):
                    # Save previous entry if exists
                    if current_spectrum and current_spectrum['peaks']:
                        spectra.append(current_spectrum)
                    
                    # Start new entry
                    name = line.split(':', 1)[1].strip()
                    
                    # Extract charge from name (format: SEQUENCE/charge)
                    charge = 0
                    if '/' in name:
                        try:
                            charge = int(name.split('/')[-1])
                            peptide_sequence = name.split('/')[0]
                        except ValueError:
                            charge = 0
                            peptide_sequence = name
                    else:
                        peptide_sequence = name
                    
                    current_spectrum = {
                        'peaks': [],
                        'identifier': name,
                        'precursor_mz': 0.0,
                        'precursor_charge': charge,
                        'scan': 0,
                        'retention_time': 0.0,
                        'source_file': os.path.basename(filepath),
                        'peptide_sequence': peptide_sequence,
                        'lib_id': '',
                        'mw': 0.0,
                        'metadata': {}
                    }
                    
                    if include_annotations:
                        current_spectrum['peak_annotations'] = []
                    
                    reading_peaks = False
                
                elif current_spectrum:
                    if line.startswith('LibID:'):
                        lib_id = line.split(':', 1)[1].strip()
                        current_spectrum['lib_id'] = lib_id
                        # Use LibID as scan number if available
                        try:
                            current_spectrum['scan'] = int(lib_id)
                        except ValueError:
                            pass
                    
                    elif line.startswith('MW:'):
                        try:
                            current_spectrum['mw'] = float(line.split(':', 1)[1].strip())
                        except ValueError:
                            logger.warning(f"Could not parse MW at line {line_num}: {line}")
                    
                    elif line.startswith('PrecursorMZ:'):
                        try:
                            current_spectrum['precursor_mz'] = float(line.split(':', 1)[1].strip())
                        except ValueError:
                            logger.warning(f"Could not parse PrecursorMZ at line {line_num}: {line}")
                    
                    elif line.startswith('Status:'):
                        current_spectrum['metadata']['status'] = line.split(':', 1)[1].strip()
                    
                    elif line.startswith('FullName:'):
                        full_name = line.split(':', 1)[1].strip()
                        current_spectrum['metadata']['full_name'] = full_name
                        
                        # Extract sequence and modifications
                        clean_seq, mods = SPTXTLoader.extract_sequence_and_modifications(full_name)
                        if clean_seq:
                            current_spectrum['peptide_sequence'] = clean_seq
                        if mods:
                            current_spectrum['metadata']['modifications'] = mods
                    
                    elif line.startswith('Comment:'):
                        comment = line.split(':', 1)[1].strip()
                        current_spectrum['metadata']['comment'] = comment
                        
                        # Parse comment field for additional metadata
                        parsed_comment = SPTXTLoader.parse_comment_field(comment)
                        current_spectrum['metadata'].update(parsed_comment)
                        
                        # Extract probability if present
                        if 'Prob' in parsed_comment:
                            current_spectrum['metadata']['probability'] = parsed_comment['Prob']
                    
                    elif line.startswith('NumPeaks:'):
                        try:
                            current_spectrum['metadata']['num_peaks'] = int(line.split(':', 1)[1].strip())
                        except ValueError:
                            logger.warning(f"Could not parse NumPeaks at line {line_num}: {line}")
                        reading_peaks = True
                    
                    elif reading_peaks:
                        # Parse peak line: m/z intensity annotation
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            try:
                                mz = float(parts[0])
                                intensity = float(parts[1])
                                
                                # Add peak
                                current_spectrum['peaks'].append((mz, intensity))
                                
                                # Add annotation if requested and available
                                if include_annotations and len(parts) >= 3:
                                    annotation = parts[2]
                                    current_spectrum['peak_annotations'].append(annotation)
                                
                            except ValueError:
                                logger.warning(f"Could not parse peak at line {line_num}: {line}")
                                continue
        
        # Don't forget the last entry
        if current_spectrum and current_spectrum['peaks']:
            spectra.append(current_spectrum)
        
        logger.info(f"Loaded {len(spectra)} spectra from SPTXT file")
        return spectra
    
    @staticmethod
    def load_sptxt_files(input_path: str, include_annotations: bool = False) -> List[Dict[str, Any]]:
        """
        Load SPTXT files from a file or directory.
        
        Args:
            input_path: Path to SPTXT file or directory containing SPTXT files
            include_annotations: Whether to include peak annotations
            
        Returns:
            List of spectrum dictionaries
        """
        import glob
        
        all_spectra = []
        
        if os.path.isfile(input_path):
            # Single file
            if input_path.lower().endswith(('.sptxt', '.splib', '.spl')):
                all_spectra = SPTXTLoader.load_sptxt_file(input_path, include_annotations)
            else:
                logger.warning(f"File {input_path} does not appear to be an SPTXT file")
        
        elif os.path.isdir(input_path):
            # Directory of files
            sptxt_files = []
            for ext in ['*.sptxt', '*.splib', '*.spl', '*.SPTXT', '*.SPLIB', '*.SPL']:
                sptxt_files.extend(glob.glob(os.path.join(input_path, ext)))
            
            if not sptxt_files:
                logger.warning(f"No SPTXT files found in {input_path}")
            else:
                logger.info(f"Found {len(sptxt_files)} SPTXT files")
                for sptxt_file in sptxt_files:
                    spectra = SPTXTLoader.load_sptxt_file(sptxt_file, include_annotations)
                    all_spectra.extend(spectra)
        
        else:
            raise FileNotFoundError(f"Path {input_path} does not exist")
        
        logger.info(f"Loaded {len(all_spectra)} spectra total from SPTXT files")
        return all_spectra


def convert_sptxt_to_mgf_format(sptxt_spectra: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert SPTXT spectra to MGF format for compatibility.
    
    Args:
        sptxt_spectra: List of spectra from SPTXT files
        
    Returns:
        List of spectra in MGF-compatible format
    """
    mgf_spectra = []
    
    for spectrum in sptxt_spectra:
        # The spectrum is already in a compatible format
        # Just ensure all required fields are present
        mgf_spectrum = {
            'peaks': spectrum.get('peaks', []),
            'identifier': spectrum.get('identifier', ''),
            'precursor_mz': spectrum.get('precursor_mz', 0.0),
            'precursor_charge': spectrum.get('precursor_charge', 0),
            'scan': spectrum.get('scan', 0),
            'retention_time': spectrum.get('retention_time', 0.0),
            'source_file': spectrum.get('source_file', ''),
        }
        
        # Add any additional metadata
        if 'metadata' in spectrum:
            mgf_spectrum['metadata'] = spectrum['metadata']
        
        if 'peptide_sequence' in spectrum:
            mgf_spectrum['peptide_sequence'] = spectrum['peptide_sequence']
        
        mgf_spectra.append(mgf_spectrum)
    
    return mgf_spectra
