import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_comparative_analysis(save_path='scholar_copilot_comparative.pdf'):
    # Configure fonts to match LaTeX
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times', 'Palatino', 'New Century Schoolbook', 'Bookman']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

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
    for category in data:
        total = sum(data[category].values())
        for rating in data[category]:
            data[category][rating] = (data[category][rating] / total) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # X-axis categories
    categories = ['Much worse', 'Worse', 'Similar', 'Better', 'Much better']
    x = np.arange(len(categories))

    # Set the width for the bars and spacing
    total_width = 0.8
    num_dims = len(data)
    width = total_width / num_dims

    # Custom colors that match the image
    colors = ['#3498db', '#f39c12', '#2ecc71', '#e67e22', '#e91e63']

    # Plot each dimension side by side (not overlapping)
    for i, (dimension, values) in enumerate(data.items()):
        percentages = [values[cat] for cat in categories]
        offset = (i - (num_dims - 1) / 2) * width
        bars = ax.bar(x + offset, percentages, width, label=dimension, color=colors[i], alpha=0.9)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only label bars with values > 0
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{int(round(height))}%',
                        ha='center', va='bottom', fontsize=10, family='serif')

    # Customize the plot
    ax.set_ylabel('Percentage of Participants', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#dddddd')
    ax.set_axisbelow(True)  # Place gridlines behind bars

    # Remove frame on right and top
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Add legend without frame at the top
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=len(data), fontsize=12, frameon=False)

    # Adjust layout with more space at top for legend
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


# Call the function
plot_comparative_analysis()