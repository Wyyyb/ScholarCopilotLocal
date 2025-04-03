import matplotlib.pyplot as plt
import numpy as np


def plot_radar_comparison(save_path='scholar_copilot_radar.pdf'):
    # Configure fonts to match LaTeX
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times', 'Palatino']
    plt.rcParams['font.size'] = 12

    # Categories and their corresponding weights
    categories = ['Much worse', 'Worse', 'Similar', 'Better', 'Much better']
    # Convert to numerical scale (1-5)
    weights = {
        'Much worse': 1,
        'Worse': 2,
        'Similar': 3,
        'Better': 4,
        'Much better': 5
    }

    # Data with counts
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

    # Calculate weighted average for each dimension
    dimensions = list(data.keys())
    scores = []

    for dimension in dimensions:
        total = sum(data[dimension].values())
        weighted_sum = sum(data[dimension][cat] * weights[cat] for cat in categories)
        scores.append(weighted_sum / total)

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    scores += scores[:1]  # Close the loop
    angles += angles[:1]  # Close the loop
    dimensions += dimensions[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw polygon and lines
    ax.plot(angles, scores, 'o-', linewidth=2, label='ScholarCopilot vs ChatGPT')
    ax.fill(angles, scores, alpha=0.25)

    # Set labels
    ax.set_thetagrids(np.degrees(angles[:-1]), dimensions[:-1], fontsize=12)

    # Set radial limits
    ax.set_ylim(1, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['Much worse', 'Worse', 'Similar', 'Better', 'Much better'])

    # Add title
    plt.title('Comparative Analysis of ScholarCopilot vs. ChatGPT', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


# Call the function
plot_radar_comparison()