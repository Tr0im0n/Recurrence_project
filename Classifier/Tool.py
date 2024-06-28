import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from preprocessing import load_data, prepare_datasets_multi_class, sliding_window_view
from feature_extraction import pyrqa
from classifier import train_multiclass_classifier
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

def main():
    # Constants (same as before)
    m = 3
    T = 2
    epsilon = 0.5
    l = 1000
    delay = 100
    num_samples = 50000
    train_samples = 30000

    # Load data (same as before)
    healthy_data_path = 'Classifier/data/normal_3hp_1730rpm.csv'
    inner_race_fault_007_path = 'Classifier/data/.007_inner_race.csv'
    ball_fault_007_path = 'Classifier/data/.007_ball.csv'
    outer_race_fault_007_path = 'Classifier/data/.007_centerd_6.csv'

    healthy = load_data(healthy_data_path, 'X100_DE_time', num_samples)
    inner_race_fault_007 = load_data(inner_race_fault_007_path, 'X121_DE_time', num_samples)
    ball_fault_007 = load_data(ball_fault_007_path, 'X108_DE_time', num_samples)
    outer_race_fault_007 = load_data(outer_race_fault_007_path, 'X133_DE_time', num_samples)

    data = [healthy, ball_fault_007, inner_race_fault_007, outer_race_fault_007]
    fault_names = ['Healthy', 'Ball fault', 'Inner race fault', 'Outer race fault']

    # Feature extraction and dataset preparation (same as before)
    feature_func2 = lambda data: pyrqa(data, m, T, epsilon)
    X_train, y_train = prepare_datasets_multi_class(data, l, delay, feature_func2, train_samples)

    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train classifier
    classifier = train_multiclass_classifier(X_train_scaled, y_train)

    # Extract recurrence rate (RR) and determinism (DET) from scaled features
    rr = X_train_scaled[:, 0]
    det = X_train_scaled[:, 1]

    # Create a mesh grid for the decision boundary (increase resolution for smoothness)
    x_min, x_max = rr.min() - 0.5, rr.max() + 0.5
    y_min, y_max = det.min() - 0.5, det.max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    # Prepare the full feature space for prediction
    mean_other_features = X_train_scaled[:, 2:].mean(axis=0)
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    full_mesh_points = np.column_stack([mesh_points, np.tile(mean_other_features, (mesh_points.shape[0], 1))])

    # Predict using the SVM
    Z = classifier.predict(full_mesh_points)
    Z = Z.reshape(xx.shape)

    # Set up custom colors and markers
    light_grey = '#D3D3D3'
    dark_turquoise = '#00CED1'
    custom_cmap = ListedColormap([light_grey, dark_turquoise, '#FF69B4', dark_turquoise])
    markers = ['o', 's', '^', 'p']  # circle, square, triangle, pentagon
    colors = ['#000000', '#0000FF', '#FF0000', '#008000']  # black, blue, red, green

    # Plot the decision boundary and training points
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=custom_cmap)

    # Plot the training points with different shapes and filled markers
    for i, (marker, color) in enumerate(zip(markers, colors)):
        mask = y_train == i
        plt.scatter(rr[mask], det[mask], c=color, marker=marker, s=60, edgecolor='black')

    # Plot the support vectors (projected onto RR and DET)
    support_vectors_proj = classifier.support_vectors_[:, :2]
    plt.scatter(support_vectors_proj[:, 0], support_vectors_proj[:, 1],
                s=100, facecolors='none', edgecolors='k', linewidths=1.5, alpha=0.5)

    # Remove axis labels and titles
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')

    # Add legend with custom labels, colors, and shapes
    legend_elements = [Line2D([0], [0], marker=markers[i], color='w', label=fault_names[i], 
                              markerfacecolor=colors[i], markersize=10, markeredgecolor='black') 
                       for i in range(len(fault_names))]
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none', 
                                  markeredgecolor='k', markersize=10, label='Support Vectors', markeredgewidth=1.5))
    plt.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig('svm_decision_boundary_custom_filled.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()










""" import numpy as np
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
    main() """