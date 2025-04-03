import matplotlib.pyplot as plt
import numpy as np


def plot_scholar_copilot_ratings():
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
        'Ease of Use': 3.60,
        'Response Time': 3.10,
        'Interface Clarity': 4.50,
        'Interaction Fluidity': 3.70,
        'Academic Rigor': 3.80,
        'Factual Accuracy': 4.30,
        'Writing Style': 4.50,
        'Logical Flow': 3.80,
        'Completeness': 3.60,
        'Topic Relevance': 3.90,
        'Innovation': 2.50,
        'Text Repetition': 3.10
    }

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot bars with different colors for each category
    x = np.arange(len(ratings))
    bars = plt.bar(x, list(ratings.values()))

    # Color the bars by category
    colors = {'Citation Quality': 'red', 'User Experience': 'blue', 'Content Quality': 'green'}
    current_idx = 0
    for category, metrics_list in metrics.items():
        for _ in metrics_list:
            bars[current_idx].set_color(colors[category])
            bars[current_idx].set_alpha(0.6)
            current_idx += 1

    # Customize the plot
    plt.ylabel('Average Rating (1-5)')
    plt.ylim(0, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Rotate x-axis labels
    plt.xticks(x, list(ratings.keys()), rotation=45, ha='right')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom')

    # Add legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.6)
                       for color in colors.values()]
    plt.legend(legend_elements, colors.keys(),
               loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    # Adjust layout
    plt.title('Under review as a conference paper at COLM 2025')
    plt.tight_layout()

    # Show plot
    plt.show()


# Call the function
plot_scholar_copilot_ratings()