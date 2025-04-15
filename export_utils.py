"""
Utility functions for exporting search results to various formats.
"""

import os
import csv
import json
from typing import Dict, List, Any, Optional
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_results_to_csv(results: Dict[str, Any], output_file: str) -> None:
    """
    Export search results to a CSV file with a cleaner format.
    
    Args:
        results: Dictionary containing search results
        output_file: Path to output CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    rows = []
    for i, query_result in enumerate(results['queries']):
        query = query_result['query_info']
        query_info = [
            i+1,
            query['identifier'],
            query['scan'],
            f"{query['precursor_mz']:.4f}",
            query['precursor_charge']
        ]
        
        if not query_result['matches']:
            # Add a row with no matches
            rows.append(query_info + ['', '', '', '', '', ''])
        else:
            # Write a row for each match
            for j, match in enumerate(query_result['matches']):
                match_info = [
                    j+1,
                    match['scan'],
                    f"{match['precursor_mz']:.4f}",
                    match['precursor_charge'],
                    match['hamming_distance'],
                    match['identifier'],
                    match.get('source_file', '')
                ]
                
                rows.append(query_info + match_info)
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header
        writer.writerow(['Query Number', 'Query ID', 'Query Scan', 'Query m/z', 'Query Charge', 
                        'Match Rank', 'Match Scan', 'Match m/z', 'Match Charge', 'Hamming Distance', 
                        'Match ID', 'Source File'])
        
        # Write data rows
        for row in rows:
            writer.writerow(row)
    
    logger.info(f"Results exported to {output_file}")

def export_results_to_txt(results: Dict[str, Any], output_file: str) -> None:
    """
    Export search results to a text file that mimics terminal output format.
    
    Args:
        results: Dictionary containing search results
        output_file: Path to output text file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(f"# Search Results using Binary Hypervector Index\n\n")
        f.write(f"Total queries processed: {results['total_queries']}\n")
        f.write(f"Total matches found: {results['total_matches']}\n\n")
        
        for i, query_result in enumerate(results['queries']):
            query = query_result['query_info']
            f.write(f"Query {i+1}: {query['identifier']} (Scan {query['scan']}, m/z {query['precursor_mz']:.4f}, charge {query['precursor_charge']})\n")
            
            if not query_result['matches']:
                f.write("  No matches found below threshold\n\n")
                continue
                
            f.write(f"  {'Rank':<5} {'Scan':<10} {'m/z':<10} {'Charge':<7} {'Distance':<10} {'Source':<15} {'Identifier'}\n")
            for j, match in enumerate(query_result['matches']):
                source = match.get('source_file', '')
                if len(source) > 15:
                    source = source[:12] + '...'
                f.write(f"  {j+1:<5} {match['scan']:<10} {match['precursor_mz']:.4f} {match['precursor_charge']:<7} " 
                        f"{match['hamming_distance']:<10} {source:<15} {match['identifier']}\n")
            
            f.write("\n")
    
    logger.info(f"Results exported to {output_file} in readable format")

def export_results_to_json(results: Dict[str, Any], output_file: str) -> None:
    """
    Export search results to a JSON file.
    
    Args:
        results: Dictionary containing search results
        output_file: Path to output JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results exported to {output_file} in JSON format")

def create_summary_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a pandas DataFrame summarizing the search results.
    
    Args:
        results: Dictionary containing search results
        
    Returns:
        DataFrame with summary statistics
    """
    # Count matches per query
    match_counts = [len(q['matches']) for q in results['queries']]
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Total Queries': [results['total_queries']],
        'Queries with Matches': [sum(1 for c in match_counts if c > 0)],
        'Total Matches': [results['total_matches']],
        'Average Matches per Query': [results['total_matches'] / results['total_queries'] if results['total_queries'] > 0 else 0],
        'Min Hamming Distance': [min([match['hamming_distance'] for q in results['queries'] for match in q['matches']]) if results['total_matches'] > 0 else None],
        'Max Hamming Distance': [max([match['hamming_distance'] for q in results['queries'] for match in q['matches']]) if results['total_matches'] > 0 else None],
        'Avg Hamming Distance': [sum([match['hamming_distance'] for q in results['queries'] for match in q['matches']]) / results['total_matches'] if results['total_matches'] > 0 else None]
    })
    
    return summary

def export_summary_to_csv(results: Dict[str, Any], output_file: str) -> None:
    """
    Export summary statistics to a CSV file.
    
    Args:
        results: Dictionary containing search results
        output_file: Path to output CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Create summary DataFrame
    summary = create_summary_dataframe(results)
    
    # Write to CSV
    summary.to_csv(output_file, index=False)
    
    logger.info(f"Summary statistics exported to {output_file}")