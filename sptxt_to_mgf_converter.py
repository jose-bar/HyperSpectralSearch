#!/usr/bin/env python3
"""
SPTXT to MGF Converter
Converts spectral library files from SPTXT format to MGF format.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SpectrumEntry:
    """Class to hold spectrum data from SPTXT file"""
    
    def __init__(self):
        self.name = ""
        self.lib_id = ""
        self.mw = 0.0
        self.precursor_mz = 0.0
        self.charge = 1
        self.status = ""
        self.full_name = ""
        self.comment = ""
        self.num_peaks = 0
        self.peaks = []  # List of (m/z, intensity) tuples
        
    def extract_charge(self):
        """Extract charge state from the name field"""
        # Name format: PEPTIDESEQUENCE/charge
        if '/' in self.name:
            try:
                self.charge = int(self.name.split('/')[-1])
            except ValueError:
                self.charge = 1
    
    def parse_comment(self):
        """Parse comment field to extract additional metadata"""
        metadata = {}
        
        # Split by spaces but preserve quoted strings
        parts = re.findall(r'(\w+)=([^\s]+)', self.comment)
        for key, value in parts:
            metadata[key] = value
            
        return metadata
    
    def get_peptide_sequence(self):
        """Extract peptide sequence from name"""
        if '/' in self.name:
            return self.name.split('/')[0]
        return self.name


def parse_sptxt(filepath: str) -> List[SpectrumEntry]:
    """
    Parse SPTXT file and return list of SpectrumEntry objects
    
    Args:
        filepath: Path to the SPTXT file
        
    Returns:
        List of SpectrumEntry objects
    """
    entries = []
    current_entry = None
    reading_peaks = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for new entry
            if line.startswith('Name:'):
                # Save previous entry if exists
                if current_entry and current_entry.peaks:
                    entries.append(current_entry)
                
                # Start new entry
                current_entry = SpectrumEntry()
                current_entry.name = line.split(':', 1)[1].strip()
                current_entry.extract_charge()
                reading_peaks = False
                
            elif current_entry:
                if line.startswith('LibID:'):
                    current_entry.lib_id = line.split(':', 1)[1].strip()
                
                elif line.startswith('MW:'):
                    current_entry.mw = float(line.split(':', 1)[1].strip())
                
                elif line.startswith('PrecursorMZ:'):
                    current_entry.precursor_mz = float(line.split(':', 1)[1].strip())
                
                elif line.startswith('Status:'):
                    current_entry.status = line.split(':', 1)[1].strip()
                
                elif line.startswith('FullName:'):
                    current_entry.full_name = line.split(':', 1)[1].strip()
                
                elif line.startswith('Comment:'):
                    current_entry.comment = line.split(':', 1)[1].strip()
                
                elif line.startswith('NumPeaks:'):
                    current_entry.num_peaks = int(line.split(':', 1)[1].strip())
                    reading_peaks = True
                
                elif reading_peaks:
                    # Parse peak line: m/z intensity annotation
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            current_entry.peaks.append((mz, intensity))
                        except ValueError:
                            continue
    
    # Don't forget the last entry
    if current_entry and current_entry.peaks:
        entries.append(current_entry)
    
    return entries


def write_mgf(entries: List[SpectrumEntry], output_filepath: str):
    """
    Write spectrum entries to MGF format
    
    Args:
        entries: List of SpectrumEntry objects
        output_filepath: Path for output MGF file
    """
    with open(output_filepath, 'w') as f:
        for i, entry in enumerate(entries):
            # Write BEGIN IONS
            f.write("BEGIN IONS\n")
            
            # Write metadata
            f.write(f"PEPMASS={entry.precursor_mz}\n")
            f.write(f"CHARGE={entry.charge}+\n")  # MGF format uses + for positive charge
            f.write("MSLEVEL=2\n")  # Assuming MS2 spectra
            
            # Add peptide sequence if available
            peptide = entry.get_peptide_sequence()
            if peptide:
                f.write(f"SEQ={peptide}\n")
            
            # Add library ID as scan number
            f.write(f"SCANS={entry.lib_id}\n")
            
            # Parse and add relevant metadata from comment
            metadata = entry.parse_comment()
            if 'Prob' in metadata:
                f.write(f"SCORE={metadata['Prob']}\n")
            
            # Add molecular weight as custom field
            f.write(f"MW={entry.mw}\n")
            
            # Add full name as title
            if entry.full_name:
                f.write(f"TITLE={entry.full_name}\n")
            
            # Write peaks
            for mz, intensity in entry.peaks:
                f.write(f"{mz}\t{intensity}\n")
            
            # Write END IONS
            f.write("END IONS\n")
            
            # Add blank line between entries (except for last one)
            if i < len(entries) - 1:
                f.write("\n")


def convert_sptxt_to_mgf(input_file: str, output_file: Optional[str] = None):
    """
    Main conversion function
    
    Args:
        input_file: Path to input SPTXT file
        output_file: Path to output MGF file (optional, will auto-generate if not provided)
    """
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.with_suffix('.mgf'))
    
    print(f"Converting {input_file} to MGF format...")
    
    # Parse SPTXT file
    entries = parse_sptxt(input_file)
    print(f"Parsed {len(entries)} spectra from SPTXT file")
    
    # Write MGF file
    write_mgf(entries, output_file)
    print(f"Successfully wrote MGF file to {output_file}")
    
    # Print summary statistics
    total_peaks = sum(len(entry.peaks) for entry in entries)
    print(f"\nConversion Summary:")
    print(f"  - Total spectra: {len(entries)}")
    print(f"  - Total peaks: {total_peaks}")
    print(f"  - Average peaks per spectrum: {total_peaks/len(entries):.1f}")


def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python sptxt_to_mgf.py <input_sptxt_file> [output_mgf_file]")
        print("\nExample:")
        print("  python sptxt_to_mgf.py library.sptxt")
        print("  python sptxt_to_mgf.py library.sptxt output.mgf")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    try:
        convert_sptxt_to_mgf(input_file, output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
