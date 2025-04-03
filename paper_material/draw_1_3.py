import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl


def plot_scholar_copilot_ratings(save_path='scholar_copilot_ratings.pdf', dpi=300):
    # Set LaTeX-like fonts
    plt.style.use('seaborn-v0_8-paper')

    # Configure fonts to match LaTeX
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times', 'Palatino', 'New Century Schoolbook', 'Bookman']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 16

    # Uncomment for true LaTeX rendering (requires LaTeX installation)
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Data
    metrics = {
        'Citation Quality': ['Citation Relevance', 'Citation Accuracy', 'Citation Timeliness'],
        'User Experience': ['Ease of Use', 'Response Time', 'Interface Clarity', 'Interaction Fluidity'],
        'Content Quality': ['Academic Rigor', 'Factual Accuracy', 'Writing Style', 'Logical Flow',
                            'Completeness', 'Topic Relevance', 'Innovation', 'Text Repetition']
    }

    ratings = {
        'Citation Relevance': 4.20,
        'Citation Accuracy': 4.60,
        'Citation Timeliness': 4.10,
        'Ease of Use': 3.90,
        'Response Time': 3.30,
        'Interface Clarity': 4.50,
        'Interaction Fluidity': 3.90,
        'Academic Rigor': 3.80,
        'Factual Accuracy': 4.30,
        'Writing Style': 4.50,
        'Logical Flow': 3.80,
        'Completeness': 3.60,
        'Topic Relevance': 3.90,
        'Innovation': 2.50,
        'Text Repetition': 3.10
    }

    # Get color palette
    colors = sns.color_palette('colorblind')
    category_colors = {
        'Citation Quality': colors[0],
        'User Experience': colors[1],
        'Content Quality': colors[2]
    }

    # Create figure with higher resolution
    plt.figure(figsize=(15, 6))

    # Plot bars
    x = np.arange(len(ratings))
    bars = plt.bar(x, list(ratings.values()), width=0.7)

    # Color the bars by category
    current_idx = 0
    for category, metrics_list in metrics.items():
        for _ in metrics_list:
            bars[current_idx].set_color(category_colors[category])
            bars[current_idx].set_alpha(0.8)
            current_idx += 1

    # Customize the plot
    plt.ylabel('Average Rating (1-5)', fontsize=16)
    plt.ylim(2, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Rotate x-axis labels
    plt.xticks(x, list(ratings.keys()), rotation=30, ha='right', fontsize=14)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=16, family='serif')

    # Add legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.8)
                       for color in category_colors.values()]
    plt.legend(legend_elements, category_colors.keys(),
               loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=3, fontsize=16, frameon=False)

    # Adjust layout
    plt.tight_layout()

    # Save figure with high resolution
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


# Call the function
plot_scholar_copilot_ratings()