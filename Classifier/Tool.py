import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from feature_extraction import calc_recurrence_plot, calc_rqa_measures, pyrqa
from preprocessing import sliding_window_view

def load_data(filepath, column_name, num_samples):
    return pd.read_csv(filepath)[column_name][:num_samples].values

def calculate_windowed_rqa_measures(data, window_size, delay, m, T, epsilon):
    all_measures = []
    for series in data:
        windows = sliding_window_view(series, window_size, delay)
        for window in windows:
            rp = calc_recurrence_plot(window, m, T, epsilon)
            measures = calc_rqa_measures(rp)
            all_measures.append(measures)
    return np.array(all_measures)
def plot_distributions(rqa_df):
    measure_names = rqa_df.columns
    fig, axes = plt.subplots(4, 2, figsize=(20, 30))
    fig.suptitle('Distribution of RQA Measures Across All Windows', fontsize=16)
    
    for i, measure in enumerate(measure_names):
        ax = axes[i // 2, i % 2]
        sns.histplot(rqa_df[measure], kde=True, ax=ax)
        ax.set_title(f'{measure} Distribution')
        ax.set_xlabel(measure)
        
        # Add statistics
        mean = rqa_df[measure].mean()
        median = rqa_df[measure].median()
        skewness = skew(rqa_df[measure])
        kurt = kurtosis(rqa_df[measure])
        
        stats_text = f'Mean: {mean:.2f}\nMedian: {median:.2f}\nSkew: {skewness:.2f}\nKurtosis: {kurt:.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.show()

def plot_boxplots(rqa_df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=rqa_df)
    plt.title('Boxplots of RQA Measures Across All Windows')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    m, T, epsilon = 10, 2, 0.1
    num_samples = 25000
    window_size = 1000
    delay = 100

    # Load data
    data_paths = [
        'Classifier/data/normal_3hp_1730rpm.csv',
        'Classifier/data/.007_inner_race.csv',
        'Classifier/data/.007_ball.csv',
        'Classifier/data/.007_centerd_6.csv'
    ]
    column_names = ['X100_DE_time', 'X121_DE_time', 'X108_DE_time', 'X133_DE_time']
    fault_types = ['Healthy', 'Inner Race Fault', 'Ball Fault', 'Outer Race Fault']
    all_data = [load_data(path, col, num_samples) for path, col in zip(data_paths, column_names)]
    
    # Calculate RQA measures for all windows
    all_measures = []
    for i, data in enumerate(all_data):
        measures = calculate_windowed_rqa_measures([data], window_size, delay, m, T, epsilon)
        all_measures.append(measures)
        print(f"Calculated measures for {fault_types[i]}: {measures.shape}")
    
    all_measures = np.vstack(all_measures)
    
    # Create DataFrame
    measure_names = ['RR', 'DET', 'L', 'TT', 'Lmax', 'DIV', 'ENTR', 'LAM']
    rqa_df = pd.DataFrame(all_measures, columns=measure_names)
    
    # Plot distributions
    plot_distributions(rqa_df)
    
    # Plot boxplots
    plot_boxplots(rqa_df)
    
    # Print summary statistics
    print(rqa_df.describe())
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print(rqa_df.corr())

    # Plot distributions for each fault type
    for i, fault_type in enumerate(fault_types):
        start = i * (all_measures.shape[0] // len(fault_types))
        end = (i + 1) * (all_measures.shape[0] // len(fault_types))
        fault_df = pd.DataFrame(all_measures[start:end], columns=measure_names)
        
        plt.figure(figsize=(20, 10))
        fault_df.hist(bins=50)
        plt.suptitle(f'Distribution of RQA Measures for {fault_type}')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()