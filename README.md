# HyperSpectral Search

HyperSpectral is a powerful mass spectrometry search tool that uses binary hypervectors and FAISS indexing to efficiently find similar spectra in large spectral datasets. Inspired by MASST (Mass Spectrometry Search Tool), HyperSpectral provides significantly faster searching while maintaining accuracy.

## Features

- **High-Performance Search Engine**: Using binary hypervectors and FAISS indexing for fast searches
- **Precursor Mass Filtering**: Enforce precursor mass tolerance for accurate matching
- **Hamming Distance Scoring**: Binary vector comparison for efficient similarity measurement
- **Versatile Output Formats**: Export results as TXT, CSV, or JSON
- **Visualization Tools**: Create mirror plots, histograms, and other visualizations of search results
- **Command-Line Interface**: Easy-to-use CLI for building indices and searching spectra

## Installation

### Requirements

- Python 3.7 or higher
- NumPy
- Pandas
- FAISS (faiss-cpu or faiss-gpu)
- Matplotlib (for visualization)
- Joblib
- PyYAML (for configuration files)
- CuPy (optional, for GPU acceleration)
- pyteomics (optional, for mzML/mzXML support)

### Install

```bash
git clone https://github.com/jose-bar/hyperspectral.git
cd hyperspectral
pip install -r requirements.txt
```

## Usage

HyperSpectral provides a command-line interface with three main commands: `build`, `search`, and `checkpoints`.

### Building an Index

```bash
python cli.py build --input /path/to/mgf/files --output-dir ./indices --index-type flat
```

#### Build Options

- `--input`: Path to MGF file or directory containing MGF files
- `--output-dir`: Directory to save the index (default: ./indices)
- `--prefix`: Prefix for index files (default: spectra_index)
- `--index-type`: Type of FAISS index to build (choices: flat, ivf, hnsw; default: flat)
- `--min-peaks`: Minimum number of peaks per spectrum
- `--min-mz`: Minimum m/z value to consider
- `--max-mz`: Maximum m/z value to consider
- `--fragment-tol`: Fragment ion tolerance in Da

### Searching Against an Index

Once you have built an index, you can search query spectra against it:

```bash
python cli.py search --index ./indices/spectra_index.bin --query query.mgf --output-dir ./results
```

#### Search Options

- `--index`: Path to index file (.bin)
- `--query`: Path to query MGF file
- `--output-dir`: Directory to save results (default: ./results)
- `--prefix`: Prefix for result files (default: search)
- `--top-k`: Number of results per query (default: 10)
- `--hamming-threshold`: Maximum Hamming distance for a match (default: 205)
- `--precursor-tol`: Precursor mass tolerance in Da (default: 0.05)
- `--output-format`: Output format (choices: txt, csv, json, all, none; default: txt)
- `--verbose`: Print detailed output

## Output Format

### Text Output (Default)

The text output format mimics the terminal display format:

```
# Search Results using Binary Hypervector Index

Total queries processed: 42
Total matches found: 156

Query 1: Scan Number: 8 (Scan 8, m/z 226.6540, charge 0)
  Rank  Scan       m/z        Charge  Distance   Source          Identifier
  1     8          226.6540   0       0          sample1.mgf     Scan Number: 8
  2     2693       746.4230   0       123        sample2.mgf     Scan Number: 2693

Query 2: Scan Number: 18 (Scan 18, m/z 252.6290, charge 0)
  No matches found below threshold
```

### CSV Output

The CSV output contains detailed information about all queries and matches:

```csv
Query Number,Query ID,Query Scan,Query m/z,Query Charge,Match Rank,Match Scan,Match m/z,Match Charge,Hamming Distance,Match ID,Source File
1,Scan Number: 8,8,226.6540,0,1,8,226.6540,0,0,Scan Number: 8,sample1.mgf
1,Scan Number: 8,8,226.6540,0,2,2693,746.4230,0,123,Scan Number: 2693,sample2.mgf
```

### JSON Output

The JSON output contains the complete search results structure for programmatic processing.

## Configuration Parameters

HyperSpectral can be configured with various parameters that affect preprocessing, encoding, and searching:

### Preprocessing Parameters

- `min_peaks` (default: 5): Minimum number of peaks per spectrum
- `min_mz_range` (default: 250.0): Minimum m/z range per spectrum
- `min_mz` (default: 101.0): Minimum m/z value to consider
- `max_mz` (default: 1500.0): Maximum m/z value to consider
- `remove_precursor_tol` (default: 1.5): Tolerance for removing precursor peak
- `min_intensity` (default: 0.01): Minimum relative intensity for peaks
- `max_peaks_used` (default: 50): Maximum number of peaks to use per spectrum
- `scaling` (default: 'off'): Intensity scaling method ('off', 'root', 'log')

### Search Parameters

- `precursor_tol` (default: 0.05): Precursor mass tolerance in Da
- `hamming_threshold` (default: 205): Maximum Hamming distance for a match
- `min_matched_peaks` (default: 6): Minimum number of matched peaks required

### Hyperdimensional Computing Parameters

- `hd_dim` (default: 2048): Dimension of hypervectors
- `hd_Q` (default: 16): Quantization level for intensity
- `hd_id_flip_factor` (default: 2.0): Factor for ID hypervector generation
- `fragment_tol` (default: 0.05): Fragment ion tolerance in Da

## API Usage

For programmatic use, you can directly use the `SpectraSearchPipeline` class:

```python
from hyperspectral import SpectraSearchPipeline

# Create pipeline
pipeline = SpectraSearchPipeline(output_dir='./indices')

# Process dataset with streaming (for large datasets)
pipeline.preprocess_dataset_streaming('/path/to/large/dataset', 
                                     batch_size=5000, max_memory_gb=8.0)
pipeline.build_index_streaming(index_type='hnsw')
pipeline.save_index(prefix='spectra_index')

# Later, search against the index
pipeline = SpectraSearchPipeline(index_path='./indices/spectra_index.bin')
pipeline.load_index()
results = pipeline.process_query_mgf('query.mgf', k=10, 
                                   hamming_threshold=205, precursor_tol=0.05)

# Export results
from export_utils import export_results_to_txt
export_results_to_txt(results, 'results.txt')
```

## Troubleshooting

### Out of Memory Errors
- Use streaming mode with smaller batch sizes
- Reduce `--max-memory` parameter
- Enable checkpointing to process in stages

### Checkpoint Issues
- List checkpoints with `python cli.py checkpoints list`
- Clean old checkpoints to free disk space
- Check `./checkpoints` directory permissions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
