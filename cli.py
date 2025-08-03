#!/usr/bin/env python3
"""
Command-line interface for HyperSpectral Search.

This module provides a command-line interface for building indices,
searching spectra, and exporting results with checkpoint support.
"""

import os
import argparse
import logging
import time
import sys
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
    
    # Check if we're using streaming mode
    if args.streaming:
        logger.info("Using streaming mode for memory-efficient processing")
        
        # Import checkpoint manager if needed
        if args.enable_checkpoint or args.resume_checkpoint:
            from checkpoint_manager import CheckpointManager
            
        # Handle checkpoint resume
        if args.resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_checkpoint}")
            pipeline.preprocess_dataset_streaming_with_checkpoint(
                args.input, 
                batch_size=args.batch_size,
                max_memory_gb=args.max_memory,
                enable_checkpointing=True,
                resume_checkpoint_id=args.resume_checkpoint
            )
        else:
            # New streaming job
            if args.enable_checkpoint:
                logger.info("Checkpointing enabled for this job")
                pipeline.preprocess_dataset_streaming_with_checkpoint(
                    args.input, 
                    batch_size=args.batch_size,
                    max_memory_gb=args.max_memory,
                    enable_checkpointing=True,
                    resume_checkpoint_id=None
                )
            else:
                # Streaming without checkpointing
                pipeline.preprocess_dataset_streaming(
                    args.input, 
                    batch_size=args.batch_size,
                    max_memory_gb=args.max_memory
                )
        
        # Build index with streaming
        pipeline.build_index_streaming(index_type=args.index_type)
        
    else:
        # Traditional mode (loads all data at once)
        logger.info("Using traditional mode (loading all data into memory)")
        pipeline.preprocess_dataset(args.input)
        pipeline.build_index(index_type=args.index_type)
    
    # Save index
    pipeline.save_index(prefix=args.prefix)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Index building completed in {elapsed_time:.2f} seconds")
    
    # Clean up checkpoint if requested
    if args.streaming and args.enable_checkpoint and args.clean_checkpoint:
        logger.info("Cleaning up checkpoint files...")
        # The checkpoint manager should have saved the checkpoint ID
        

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
    
    # Print sample results if verbose
    if args.verbose:
        print_sample_results(results)

def list_checkpoints(args):
    """List all available checkpoints."""
    from checkpoint_manager import CheckpointManager
    
    manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
    checkpoints = manager.list_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    print("\nAvailable checkpoints:")
    print("-" * 80)
    for cp in checkpoints:
        print(f"ID: {cp['id']}")
        print(f"  Job: {cp['job_name']}")
        print(f"  Status: {cp['status']}")
        print(f"  Progress: {cp['progress']}")
        print(f"  Created: {cp['created']}")
        print(f"  Updated: {cp['updated']}")
        print("-" * 80)

def clean_checkpoint(args):
    """Clean up a specific checkpoint."""
    from checkpoint_manager import CheckpointManager
    
    manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
    
    try:
        manager.cleanup_checkpoint(args.checkpoint_id)
        print(f"Successfully cleaned up checkpoint: {args.checkpoint_id}")
    except Exception as e:
        print(f"Error cleaning up checkpoint: {e}")
        sys.exit(1)

def print_sample_results(results):
    """Print sample results to console."""
    print("\nSample Results:")
    max_to_show = min(3, len(results['queries']))
    
    for i in range(max_to_show):
        query = results['queries'][i]['query_info']
        print(f"\nQuery {i+1}: {query['identifier']} (Scan {query['scan']}, "
              f"m/z {query['precursor_mz']:.4f}, charge {query['precursor_charge']})")
        
        matches = results['queries'][i]['matches']
        if not matches:
            print("  No matches found below threshold")
            continue
            
        print(f"  {'Rank':<5} {'Scan':<10} {'m/z':<10} {'Charge':<7} {'Distance':<10} {'Identifier'}")
        for j, match in enumerate(matches[:5]):  # Show only top 5 matches
            print(f"  {j+1:<5} {match['scan']:<10} {match['precursor_mz']:.4f} "
                  f"{match['precursor_charge']:<7} {match['hamming_distance']:<10} "
                  f"{match['identifier']}")
        
        if len(matches) > 5:
            print(f"  ... and {len(matches) - 5} more matches")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='HyperSpectral Search Tools for Mass Spectrometry Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index with traditional mode
  %(prog)s build --input /path/to/mgf/files --output-dir ./indices
  
  # Build index with streaming mode and checkpointing
  %(prog)s build --input /path/to/large/dataset --output-dir ./indices \\
      --streaming --enable-checkpoint --batch-size 5000
  
  # Resume from checkpoint
  %(prog)s build --input /path/to/large/dataset --output-dir ./indices \\
      --streaming --resume-checkpoint job_20240115_143022
  
  # Search against index
  %(prog)s search --index ./indices/spectra_index.bin --query query.mgf \\
      --output-dir ./results --top-k 20
  
  # List checkpoints
  %(prog)s checkpoints list
  
  # Clean up checkpoint
  %(prog)s checkpoints clean job_20240115_143022
        """
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # ============ BUILD COMMAND ============
    build_parser = subparsers.add_parser('build', 
                                       help='Build a search index from MGF files',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Required arguments
    build_parser.add_argument('--input', required=True, 
                            help='Input MGF file or directory containing MGF files')
    
    # Output options
    build_parser.add_argument('--output-dir', default='./indices', 
                            help='Directory to save index')
    build_parser.add_argument('--prefix', default='spectra_index', 
                            help='Prefix for index files')
    
    # Index options
    build_parser.add_argument('--index-type', choices=['flat', 'ivf', 'hnsw'], 
                            default='flat', help='Type of FAISS index to build')
    
    # Preprocessing parameters
    build_parser.add_argument('--min-peaks', type=int, 
                            help='Minimum number of peaks per spectrum')
    build_parser.add_argument('--min-mz', type=float, 
                            help='Minimum m/z value to consider')
    build_parser.add_argument('--max-mz', type=float, 
                            help='Maximum m/z value to consider')
    build_parser.add_argument('--fragment-tol', type=float, 
                            help='Fragment ion tolerance in Da')
    
    # Streaming options
    build_parser.add_argument('--streaming', action='store_true', 
                            help='Use streaming mode for large datasets')
    build_parser.add_argument('--batch-size', type=int, default=1000,
                            help='Batch size for streaming mode')
    build_parser.add_argument('--max-memory', type=float, default=4.0,
                            help='Maximum memory usage in GB for streaming mode')
    
    # Checkpoint options
    build_parser.add_argument('--enable-checkpoint', action='store_true', 
                            help='Enable checkpointing for interruption recovery')
    build_parser.add_argument('--resume-checkpoint', type=str, 
                            help='Resume from specified checkpoint ID')
    build_parser.add_argument('--clean-checkpoint', action='store_true',
                            help='Clean up checkpoint after successful completion')
    
    build_parser.set_defaults(func=build_index)
    
    # ============ SEARCH COMMAND ============
    search_parser = subparsers.add_parser('search', 
                                        help='Search a query MGF against a prebuilt index',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Required arguments
    search_parser.add_argument('--index', required=True, 
                             help='Path to index file (.bin)')
    search_parser.add_argument('--query', required=True, 
                             help='Query MGF file')
    
    # Output options
    search_parser.add_argument('--output-dir', default='./results', 
                             help='Directory to save results')
    search_parser.add_argument('--prefix', default='search', 
                             help='Prefix for result files')
    search_parser.add_argument('--output-format', 
                             choices=['txt', 'csv', 'json', 'all', 'none'], 
                             default='txt', help='Output format for results')
    
    # Search parameters
    search_parser.add_argument('--top-k', type=int, default=10, 
                             help='Number of results per query')
    search_parser.add_argument('--hamming-threshold', type=int, default=205, 
                             help='Maximum Hamming distance for a match')
    search_parser.add_argument('--precursor-tol', type=float, default=0.05, 
                             help='Precursor mass tolerance in Da')
    
    # Other options
    search_parser.add_argument('--verbose', action='store_true', 
                             help='Print detailed output')
    
    search_parser.set_defaults(func=search)
    
    # ============ CHECKPOINTS COMMAND ============
    checkpoint_parser = subparsers.add_parser('checkpoints', 
                                            help='Manage checkpoints',
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add global checkpoint directory option
    checkpoint_parser.add_argument('--checkpoint-dir', default='./checkpoints',
                                 help='Directory containing checkpoints')
    
    # Create subparsers for checkpoint commands
    checkpoint_subparsers = checkpoint_parser.add_subparsers(dest='checkpoint_command',
                                                           help='Checkpoint command')
    checkpoint_subparsers.required = True
    
    # List command
    list_parser = checkpoint_subparsers.add_parser('list', 
                                                  help='List all checkpoints')
    list_parser.set_defaults(func=list_checkpoints)
    
    # Clean command
    clean_parser = checkpoint_subparsers.add_parser('clean', 
                                                   help='Clean up a checkpoint')
    clean_parser.add_argument('checkpoint_id', 
                            help='ID of checkpoint to clean')
    clean_parser.set_defaults(func=clean_checkpoint)
    
    # ============ PARSE ARGUMENTS ============
    args = parser.parse_args()
    
    # Create output directory if specified
    if hasattr(args, 'output_dir') and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute the appropriate function
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
