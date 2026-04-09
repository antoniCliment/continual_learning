#!/usr/bin/env python3
"""
Script to find all metrics_summary.csv files in subdirectories and create plots for each.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def create_metrics_plot(csv_path):
    """
    Create a plot from a metrics summary CSV file and save it in the same directory.
    Supports both binary and multiple choice metrics.
    """
    csv_path = Path(csv_path)
    csv_name = csv_path.name
    
    # Determine type and output name
    is_mc = 'metrics_mc' in csv_name
    dataset_type = "Train" if "train" in csv_name else "Val" if "val" in csv_name else ""
    metric_mode = "Multiple Choice" if is_mc else "Binary"
    
    out_name = f'metrics_{"mc" if is_mc else "binary"}_{dataset_type.lower()}_plot.png'
    output_path = csv_path.parent / out_name
    
    if output_path.exists():
        # print(f"⊙ Plot already exists: {output_path}")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"✗ Failed to read {csv_path}: {e}")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_suffix = f" ({dataset_type})" if dataset_type else ""
    fig.suptitle(f'{metric_mode} Metrics Summary{title_suffix} - {csv_path.parent.name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy over steps
    axes[0, 0].plot(df['step'], df['accuracy'], marker='o', linewidth=2, markersize=6, color='#2E86AB')
    axes[0, 0].set_xlabel('Step', fontsize=11)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 0].set_title('Accuracy over Training Steps', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    if is_mc:
        # MC Specific Plots
        # Plot 2: Correct, Incorrect, Unknown over steps
        axes[0, 1].plot(df['step'], df['correct'], marker='o', label='Correct', linewidth=2, markersize=5, color='#06A77D')
        axes[0, 1].plot(df['step'], df['incorrect'], marker='s', label='Incorrect', linewidth=2, markersize=5, color='#D62828')
        if 'unknown' in df.columns:
            axes[0, 1].plot(df['step'], df['unknown'], marker='^', label='Unknown', linewidth=2, markersize=5, color='#888888')
        axes[0, 1].set_xlabel('Step', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].set_title('Performance Components', fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='best', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Stacked bar chart
        width = 0.6
        x_pos = range(len(df['step']))
        axes[1, 0].bar(x_pos, df['correct'], width, label='Correct', color='#06A77D')
        axes[1, 0].bar(x_pos, df['incorrect'], width, bottom=df['correct'], label='Incorrect', color='#D62828')
        if 'unknown' in df.columns:
            axes[1, 0].bar(x_pos, df['unknown'], width, bottom=df['correct']+df['incorrect'], label='Unknown', color='#888888')
            
        axes[1, 0].set_xlabel('Step', fontsize=11)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_title('Stacked Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(df['step'], rotation=45)
        axes[1, 0].legend(loc='upper left', fontsize=9)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Multiple Answer Plot (MC Confusion Matrix)
        # Try to plot a heatmap of the confusion matrix for the LATEST step
        cm_cols = [c for c in df.columns if c.startswith('cm_')]
        if cm_cols:
            latest_row = df.iloc[-1]
            labels = ['a', 'b', 'c']
            cm_matrix = []
            for t in labels:
                row_vals = [latest_row.get(f"cm_{t}_{p}", 0) for p in labels]
                cm_matrix.append(row_vals)
            
            # Use imshow for heatmap
            im = axes[1, 1].imshow(cm_matrix, cmap='Blues')
            
            # Add labels and title
            axes[1, 1].set_xticks(range(len(labels)))
            axes[1, 1].set_yticks(range(len(labels)))
            axes[1, 1].set_xticklabels(labels)
            axes[1, 1].set_yticklabels(labels)
            axes[1, 1].set_xlabel('Predicted', fontsize=10)
            axes[1, 1].set_ylabel('Target', fontsize=10)
            axes[1, 1].set_title(f'MC Confusion Matrix (Step {latest_row["step"]})', fontsize=11, fontweight='bold')
            
            # Add text annotations in the heatmap
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axes[1, 1].text(j, i, int(cm_matrix[i][j]), ha='center', va='center', 
                                   color='white' if cm_matrix[i][j] > (df['total'].iloc[-1]/6) else 'black')
            
            # Add a colorbar
            fig.colorbar(im, ax=axes[1, 1], shrink=0.8)
        elif 'unknown' in df.columns:
            unknown_ratio = (df['unknown'] / df['total']) * 100
            axes[1, 1].plot(df['step'], unknown_ratio, marker='x', color='#888888', linewidth=2)
            axes[1, 1].set_xlabel('Step', fontsize=11)
            axes[1, 1].set_ylabel('Unknown Ratio (%)', fontsize=11)
            axes[1, 1].set_title('Unknown Prediction Rate', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
    else:
        # Binary Specific Plots (Existing logic adapted)
        # Plot 2: TP, FP, FN, TN over steps
        axes[0, 1].plot(df['step'], df['TP'], marker='o', label='True Positive', linewidth=2, markersize=5)
        axes[0, 1].plot(df['step'], df['FP'], marker='s', label='False Positive', linewidth=2, markersize=5)
        axes[0, 1].plot(df['step'], df['FN'], marker='^', label='False Negative', linewidth=2, markersize=5)
        axes[0, 1].plot(df['step'], df['TN'], marker='d', label='True Negative', linewidth=2, markersize=5)
        axes[0, 1].set_xlabel('Step', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].set_title('Confusion Matrix Components', fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='best', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Stacked bar chart
        width = 0.6
        x_pos = range(len(df['step']))
        axes[1, 0].bar(x_pos, df['TP'], width, label='True Positive', color='#06A77D')
        axes[1, 0].bar(x_pos, df['FP'], width, bottom=df['TP'], label='False Positive', color='#F77F00')
        axes[1, 0].bar(x_pos, df['FN'], width, bottom=df['TP']+df['FP'], label='False Negative', color='#D62828')
        axes[1, 0].bar(x_pos, df['TN'], width, bottom=df['TP']+df['FP']+df['FN'], label='True Negative', color='#003049')
        if 'UNKNOWN' in df.columns:
            axes[1, 0].bar(x_pos, df['UNKNOWN'], width, bottom=df['TP']+df['FP']+df['FN']+df['TN'], 
                           label='Unknown', color='#CCCCCC')
        axes[1, 0].set_xlabel('Step', fontsize=11)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_title('Stacked Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(df['step'], rotation=45)
        axes[1, 0].legend(loc='upper left', fontsize=9)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Precision/Recall or Unknowns
        if 'UNKNOWN' in df.columns and df['UNKNOWN'].any():
            axes[1, 1].plot(df['step'], df['UNKNOWN'], marker='o', linewidth=2, markersize=6, color='#A4133C')
            axes[1, 1].set_xlabel('Step', fontsize=11)
            axes[1, 1].set_ylabel('Unknown Count', fontsize=11)
            axes[1, 1].set_title('Unknown Predictions over Steps', fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Avoid division by zero
            precision = df['TP'] / (df['TP'] + df['FP'].replace(0, 1e-9))
            recall = df['TP'] / (df['TP'] + df['FN'].replace(0, 1e-9))
            axes[1, 1].plot(df['step'], precision, marker='o', label='Precision', linewidth=2, markersize=6)
            axes[1, 1].plot(df['step'], recall, marker='s', label='Recall', linewidth=2, markersize=6)
            axes[1, 1].set_xlabel('Step', fontsize=11)
            axes[1, 1].set_ylabel('Score', fontsize=11)
            axes[1, 1].set_title('Precision and Recall', fontsize=12, fontweight='bold')
            axes[1, 1].legend(loc='best', fontsize=9)
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created plot: {output_path}")


def main():
    """Find all metric summary files and create plots."""
    current_dir = Path.cwd()
    
    # Patterns to match existing and new naming conventions
    patterns = [
        # '**/metrics_summary.csv',
        # '**/metrics_train_summary.csv',
        # '**/metrics_val_summary.csv',
        '**/metrics_binary_*_summary.csv',
        '**/metrics_mc_*_summary.csv'
    ]
    
    csv_files = []
    for pattern in patterns:
        csv_files.extend(list(current_dir.glob(pattern)))
    
    # Filter and remove duplicates (Path objects handle this if we use set)
    csv_files = sorted(list(set(csv_files)))
    
    if not csv_files:
        print("No metrics summary files found.")
        return
    
    print(f"Found {len(csv_files)} metrics summary file(s)\n")
    
    for csv_file in csv_files:
        try:
            print(f"Processing: {csv_file.relative_to(current_dir)}")
            create_metrics_plot(csv_file)
        except Exception as e:
            print(f"✗ Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("-" * 80)
    print(f"Done! Processed {len(csv_files)} file(s).")


if __name__ == "__main__":
    main()
