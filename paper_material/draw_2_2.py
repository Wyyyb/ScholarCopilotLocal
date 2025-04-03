import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_heatmap_comparison(save_path='comparative_heatmap.pdf'):
    # Configure fonts to match LaTeX
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times', 'Palatino']
    plt.rcParams['font.size'] = 13

    # Raw data (counts per category)
    data = {
        'Citation Quality': {
            'Much worse': 0,
            'Worse': 0,
            'Similar': 0,
            'Better': 2,
            'Much better': 8
        },
        'Writing Quality': {
            'Much worse': 0,
            'Worse': 1,
            'Similar': 4,
            'Better': 5,
            'Much better': 0
        },
        'Ease of Use': {
            'Much worse': 0,
            'Worse': 1,
            'Similar': 3,
            'Better': 3,
            'Much better': 3
        },
        'Time Efficiency': {
            'Much worse': 1,
            'Worse': 1,
            'Similar': 1,
            'Better': 4,
            'Much better': 3
        },
        'Overall Usefulness': {
            'Much worse': 0,
            'Worse': 0,
            'Similar': 3,
            'Better': 4,
            'Much better': 3
        }
    }

    # Convert to percentages
    categories = ['Much worse', 'Worse', 'Similar', 'Better', 'Much better']
    dimensions = list(data.keys())

    # Create percentage matrix
    percentage_matrix = np.zeros((len(dimensions), len(categories)))

    for i, dimension in enumerate(dimensions):
        total = sum(data[dimension].values())
        for j, category in enumerate(categories):
            percentage_matrix[i, j] = (data[dimension][category] / total) * 100

    # Create figure
    plt.figure(figsize=(12, 6))

    # Create heatmap
    ax = sns.heatmap(percentage_matrix, annot=True, fmt='.0f', cmap='YlGnBu',
                     xticklabels=categories, yticklabels=dimensions,
                     cbar_kws={'label': 'Percentage of Participants'})

    # Add percentage sign to annotations
    for text in ax.texts:
        text.set_text(f"{text.get_text()}%")

    plt.title('Comparative Analysis of ScholarCopilot vs. ChatGPT', fontsize=16, pad=20)
    plt.tight_layout()
    # plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # Alternatively, use subplots_adjust to create more space on the right
    # plt.subplots_adjust(right=0.85)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


# Call the function
plot_heatmap_comparison()