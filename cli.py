#!/usr/bin/env python3
"""
Command-line interface for HyperSpectral Search.

This module provides a command-line interface for building indices,
searching spectra, and exporting results.
"""

import os
import argparse
import logging
import time
from typing import Dict, List, Any

from hyperspectral import SpectraSearchPipeline
from export_utils import (
    export_results_to_csv, 
    export_results_to_txt, 
    export_results_to_json, 
    export_summary_to_csv
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_index(args):
    """Build a search index from MGF files."""
    start_time = time.time()
    
    # Create pipeline
    pipeline = SpectraSearchPipeline(output_dir=args.output_dir)
    
    # Override config parameters if provided
    if args.min_peaks:
        pipeline.config['min_peaks'] = args.min_peaks
    if args.min_mz:
        pipeline.config['min_mz'] = args.min_mz
    if args.max_mz:
        pipeline.config['max_mz'] = args.max_mz
    if args.fragment_tol:
        pipeline.config['fragment_tol'] = args.fragment_tol
    
    # Process dataset
    pipeline.preprocess_dataset(args.input)
    
    # Build index
    pipeline.build_index(index_type=args.index_type)
    
    # Save index
    pipeline.save_index(prefix=args.prefix)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Index building completed in {elapsed_time:.2f} seconds")

def search(args):
    """Search a query MGF against a prebuilt index."""
    start_time = time.time()
    
    # Create pipeline
    pipeline = SpectraSearchPipeline(index_path=args.index)
    
    # Load index
    pipeline.load_index()
    
    # Override config parameters if provided
    if args.precursor_tol:
        pipeline.config['precursor_tol'] = args.precursor_tol
    if args.hamming_threshold:
        pipeline.config['hamming_threshold'] = args.hamming_threshold
    
    # Process query
    results = pipeline.process_query_mgf(
        args.query, 
        k=args.top_k,
        hamming_threshold=args.hamming_threshold,
        precursor_tol=args.precursor_tol
    )
    
    # Export results
    if args.output_format == 'txt' or args.output_format == 'all':
        txt_file = os.path.join(args.output_dir, f"{args.prefix}_results.txt")
        export_results_to_txt(results, txt_file)
    
    if args.output_format == 'csv' or args.output_format == 'all':
        csv_file = os.path.join(args.output_dir, f"{args.prefix}_results.csv")
        export_results_to_csv(results, csv_file)
    
    if args.output_format == 'json' or args.output_format == 'all':
        json_file = os.path.join(args.output_dir, f"{args.prefix}_results.json")
        export_results_to_json(results, json_file)
    
    # Export summary
    summary_file = os.path.join(args.output_dir, f"{args.prefix}_summary.csv")
    export_summary_to_csv(results, summary_file)
    
    # Print summary to console
    total_queries = results['total_queries']
    total_matches = results['total_matches']
    queries_with_matches = sum(1 for q in results['queries'] if q['matches'])
    
    print("\n=== Search Results Summary ===")
    print(f"Total queries processed: {total_queries}")
    print(f"Queries with matches: {queries_with_matches}")
    print(f"Total matches found: {total_matches}")
    
    if args.output_format != 'none':
        print(f"Results saved to {args.output_dir}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Search completed in {elapsed_time:.2f} seconds")
    
    # Print first few results to console
    if args.verbose:
        print("\nSample Results:")
        max_to_show = min(3, len(results['queries']))
        
        for i in range(max_to_show):
            query = results['queries'][i]['query_info']
            print(f"\nQuery {i+1}: {query['identifier']} (Scan {query['scan']}, m/z {query['precursor_mz']:.4f}, charge {query['precursor_charge']})")
            
            matches = results['queries'][i]['matches']
            if not matches:
                print("  No matches found below threshold")
                continue
                
            print(f"  {'Rank':<5} {'Scan':<10} {'m/z':<10} {'Charge':<7} {'Distance':<10} {'Identifier'}")
            for j, match in enumerate(matches[:5]):  # Show only top 5 matches
                print(f"  {j+1:<5} {match['scan']:<10} {match['precursor_mz']:.4f} {match['precursor_charge']:<7} {match['hamming_distance']:<10} {match['identifier']}")
            
            if len(matches) > 5:
                print(f"  ... and {len(matches) - 5} more matches")

def main():
    parser = argparse.ArgumentParser(description='HyperSpectral Search Tools for Mass Spectrometry Data')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build index command
    build_parser = subparsers.add_parser('build', help='Build a search index from MGF files')
    build_parser.add_argument('--input', required=True, help='Input MGF file or directory')
    build_parser.add_argument('--output-dir', default='./indices', help='Directory to save index')
    build_parser.add_argument('--prefix', default='spectra_index', help='Prefix for index files')
    build_parser.add_argument('--index-type', choices=['flat', 'ivf', 'hnsw'], default='flat', help='Type of FAISS index')
    build_parser.add_argument('--min-peaks', type=int, help='Minimum number of peaks per spectrum')
    build_parser.add_argument('--min-mz', type=float, help='Minimum m/z value to consider')
    build_parser.add_argument('--max-mz', type=float, help='Maximum m/z value to consider')
    build_parser.add_argument('--fragment-tol', type=float, help='Fragment ion tolerance in Da')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search a query MGF against a prebuilt index')
    search_parser.add_argument('--index', required=True, help='Path to index file (.bin)')
    search_parser.add_argument('--query', required=True, help='Query MGF file')
    search_parser.add_argument('--output-dir', default='./results', help='Directory to save results')
    search_parser.add_argument('--prefix', default='search', help='Prefix for result files')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results per query')
    search_parser.add_argument('--hamming-threshold', type=int, default=205, help='Maximum Hamming distance for a match')
    search_parser.add_argument('--precursor-tol', type=float, default=0.05, help='Precursor mass tolerance in Da')
    search_parser.add_argument('--output-format', choices=['txt', 'csv', 'json', 'all', 'none'], default='txt', help='Output format')
    search_parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.command == 'build':
        build_index(args)
    elif args.command == 'search':
        search(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()