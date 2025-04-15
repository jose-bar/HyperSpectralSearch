"""
Visualization utilities for mass spectrometry data.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
from typing import Dict, List, Tuple, Any, Optional
import logging
from mgf_utils import load_mgf_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mirror_plot(spectrum1: Dict[str, Any], 
                      spectrum2: Dict[str, Any], 
                      title: Optional[str] = None,
                      output_file: Optional[str] = None,
                      annotation: bool = True,
                      matched_peak_tol: float = 0.05) -> plt.Figure:
    """
    Create a mirror plot comparing two spectra.
    
    Args:
        spectrum1: First spectrum dictionary (top)
        spectrum2: Second spectrum dictionary (bottom)
        title: Title for the plot
        output_file: Path to save the plot, if provided
        annotation: Whether to annotate matching peaks
        matched_peak_tol: Tolerance for matching peaks between the two spectra
        
    Returns:
        Matplotlib figure object
    """
    # Extract peaks
    mz1, intensity1 = zip(*spectrum1['peaks']) if spectrum1['peaks'] else ([], [])
    mz2, intensity2 = zip(*spectrum2['peaks']) if spectrum2['peaks'] else ([], [])
    
    # Normalize intensities
    if intensity1:
        max_int1 = max(intensity1)
        intensity1 = [i / max_int1 for i in intensity1]
    
    if intensity2:
        max_int2 = max(intensity2)
        intensity2 = [i / max_int2 for i in intensity2]
        # Negate for mirror plot
        intensity2 = [-i for i in intensity2]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot spectra
    ax.stem(mz1, intensity1, markerfmt=' ', linefmt='b-', basefmt=' ', label=spectrum1.get('identifier', 'Spectrum 1'))
    ax.stem(mz2, intensity2, markerfmt=' ', linefmt='r-', basefmt=' ', label=spectrum2.get('identifier', 'Spectrum 2'))
    
    # Find and highlight matching peaks
    if annotation and mz1 and mz2:
        matched_pairs = []
        for i, m1 in enumerate(mz1):
            for j, m2 in enumerate(mz2):
                if abs(m1 - m2) <= matched_peak_tol:
                    matched_pairs.append((i, j))
                    break
        
        for i, j in matched_pairs:
            # Draw dashed line connecting matched peaks
            ax.plot([mz1[i], mz2[j]], [intensity1[i], intensity2[j]], 'k--', alpha=0.3)
            
            # Annotate matched peaks with their m/z values
            ax.annotate(f"{mz1[i]:.2f}", xy=(mz1[i], intensity1[i]), 
                      xytext=(0, 5), textcoords='offset points',
                      ha='center', va='bottom', fontsize=8, rotation=45)
            
            ax.annotate(f"{mz2[j]:.2f}", xy=(mz2[j], intensity2[j]), 
                      xytext=(0, -5), textcoords='offset points',
                      ha='center', va='top', fontsize=8, rotation=45)
    
    # Set up plot
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    if not title:
        s1_id = spectrum1.get('identifier', '').split(' ')[0]
        s2_id = spectrum2.get('identifier', '').split(' ')[0]
        title = f"Mirror Plot: {s1_id} vs {s2_id}"
    
    ax.set_title(title)
    ax.set_xlabel('m/z')
    ax.set_ylabel('Relative Intensity')
    
    # Add precursor m/z and charge information
    precursor_info = (
        f"Top: m/z {spectrum1.get('precursor_mz', 0):.4f}, charge {spectrum1.get('precursor_charge', 0)}\n"
        f"Bottom: m/z {spectrum2.get('precursor_mz', 0):.4f}, charge {spectrum2.get('precursor_charge', 0)}"
    )
    ax.text(0.01, 0.01, precursor_info, transform=ax.transAxes, fontsize=8,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add metadata about match
    if 'hamming_distance' in spectrum2:
        match_info = f"Hamming Distance: {spectrum2['hamming_distance']}"
        ax.text(0.99, 0.01, match_info, transform=ax.transAxes, fontsize=10,
               horizontalalignment='right', verticalalignment='bottom', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Mirror plot saved to {output_file}")
    
    return fig

def create_batch_mirror_plots(query_spectra: List[Dict[str, Any]], 
                             match_results: Dict[str, Any],
                             output_dir: str,
                             max_plots: int = 10) -> None:
    """
    Create mirror plots for the top matches of each query spectrum.
    
    Args:
        query_spectra: List of query spectra
        match_results: Match results dictionary
        output_dir: Directory to save plots
        max_plots: Maximum number of plots to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    query_map = {spec['scan']: spec for spec in query_spectra}
    
    plots_created = 0
    for query_result in match_results['queries']:
        if plots_created >= max_plots:
            break
            
        query_info = query_result['query_info']
        query_spectrum = query_map.get(query_info['scan'])
        
        if not query_spectrum or not query_result['matches']:
            continue
        
        # Create mirror plot for the top match
        top_match = query_result['matches'][0]
        
        # Create simplified version of the match for the plot
        match_spectrum = {
            'peaks': top_match.get('peaks', []),
            'precursor_mz': top_match['precursor_mz'],
            'precursor_charge': top_match['precursor_charge'],
            'identifier': top_match['identifier'],
            'hamming_distance': top_match['hamming_distance']
        }
        
        query_id = query_info['identifier'].replace(' ', '_').replace('/', '_')
        match_id = top_match['identifier'].replace(' ', '_').replace('/', '_')
        output_file = os.path.join(output_dir, f"mirror_{query_id}_vs_{match_id}.png")
        
        create_mirror_plot(query_spectrum, match_spectrum, output_file=output_file)
        plots_created += 1
    
    logger.info(f"Created {plots_created} mirror plots in {output_dir}")

def plot_distance_histogram(results: Dict[str, Any], 
                           output_file: Optional[str] = None,
                           bins: int = 30) -> plt.Figure:
    """
    Plot a histogram of Hamming distances from search results.
    
    Args:
        results: Search results dictionary
        output_file: Path to save the plot, if provided
        bins: Number of bins for the histogram
        
    Returns:
        Matplotlib figure object
    """
    distances = [match['hamming_distance'] for query in results['queries'] for match in query['matches']]
    
    if not distances:
        logger.warning("No distance data available for histogram")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(distances, bins=bins, alpha=0.7, color='steelblue')
    
    ax.set_title('Distribution of Hamming Distances')
    ax.set_xlabel('Hamming Distance')
    ax.set_ylabel('Frequency')
    
    # Add vertical line at the threshold if specified in results
    if 'hamming_threshold' in results:
        threshold = results['hamming_threshold']
        ax.axvline(x=threshold, color='red', linestyle='--', 
                 label=f'Threshold: {threshold}')
        ax.legend()
    
    # Add statistics
    stats_text = (
        f"Total Matches: {len(distances)}\n"
        f"Min: {min(distances)}\n"
        f"Max: {max(distances)}\n"
        f"Mean: {np.mean(distances):.2f}\n"
        f"Median: {np.median(distances):.2f}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Histogram saved to {output_file}")
    
    return fig

def plot_precursor_mz_distribution(results: Dict[str, Any], 
                                 output_file: Optional[str] = None,
                                 bins: int = 30) -> plt.Figure:
    """
    Plot a histogram of precursor m/z values from search results.
    
    Args:
        results: Search results dictionary
        output_file: Path to save the plot, if provided
        bins: Number of bins for the histogram
        
    Returns:
        Matplotlib figure object
    """
    # Extract query and match m/z values
    query_mzs = [query['query_info']['precursor_mz'] for query in results['queries']]
    match_mzs = [match['precursor_mz'] for query in results['queries'] for match in query['matches']]
    
    if not query_mzs or not match_mzs:
        logger.warning("No m/z data available for histogram")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot both histograms
    ax.hist(query_mzs, bins=bins, alpha=0.5, color='blue', label='Queries')
    ax.hist(match_mzs, bins=bins, alpha=0.5, color='red', label='Matches')
    
    ax.set_title('Distribution of Precursor m/z Values')
    ax.set_xlabel('Precursor m/z')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Histogram saved to {output_file}")
    
    return fig

def create_match_heatmap(results: Dict[str, Any], 
                        output_file: Optional[str] = None,
                        max_queries: int = 20,
                        max_matches: int = 20) -> plt.Figure:
    """
    Create a heatmap of Hamming distances between queries and their matches.
    
    Args:
        results: Search results dictionary
        output_file: Path to save the plot, if provided
        max_queries: Maximum number of queries to show
        max_matches: Maximum number of matches per query to show
        
    Returns:
        Matplotlib figure object
    """
    # Prepare data for heatmap
    queries_with_matches = [q for q in results['queries'] if q['matches']]
    if not queries_with_matches:
        logger.warning("No matches available for heatmap")
        return None
    
    # Limit number of queries
    queries_with_matches = queries_with_matches[:max_queries]
    
    # Create data matrix
    data = []
    ylabels = []
    xlabels = []
    
    for i, query in enumerate(queries_with_matches):
        row = []
        if i == 0:
            # Create x labels for the first row
            xlabels = [f"Match {j+1}" for j in range(min(max_matches, len(query['matches'])))]
        
        # Create y label
        query_info = query['query_info']
        ylabels.append(f"Q{i+1}: {query_info['scan']}")
        
        # Fill row with distances
        for j, match in enumerate(query['matches'][:max_matches]):
            row.append(match['hamming_distance'])
        
        # Pad row if needed
        while len(row) < max_matches:
            row.append(np.nan)
        
        data.append(row)
    
    # Convert to numpy array
    data = np.array(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(8, max_matches * 0.5), max(6, len(data) * 0.4)))
    
    # Create colormap that goes from green (low distance) to red (high distance)
    cmap = LinearSegmentedColormap.from_list('GreenToRed', ['green', 'yellow', 'red'])
    
    # Create heatmap
    im = ax.imshow(data, cmap=cmap)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Hamming Distance', rotation=-90, va="bottom")
    
    # Add labels
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title
    ax.set_title('Hamming Distances Between Queries and Matches')
    
    # Loop over data dimensions and create text annotations
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            if j < len(queries_with_matches[i]['matches']):
                text = ax.text(j, i, f"{data[i, j]:.0f}",
                            ha="center", va="center", color="black" if data[i, j] < 150 else "white")
    
    fig.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to {output_file}")
    
    return fig