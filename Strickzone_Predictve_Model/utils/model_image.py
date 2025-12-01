import matplotlib.pyplot as plt
import numpy as np

def draw_mlp_pretty(sizes, layer_labels=None, max_nodes_to_draw=8, figsize=(12, 6)):
    """
    sizes: list of neuron counts per layer, e.g. [4, 32, 64, 1]
    layer_labels: optional list of labels per layer
    max_nodes_to_draw: max number of circles to draw per layer (for big layers)
    """
    n_layers = len(sizes)
    h_spacing = 2.5

    # Figure out vertical limits based on how many nodes we *draw*, not actual size
    drawn_counts = [min(s, max_nodes_to_draw) for s in sizes]
    max_drawn = max(drawn_counts)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Precompute x positions
    x_positions = [i * h_spacing for i in range(n_layers)]

    for i, (layer_size, drawn_size) in enumerate(zip(sizes, drawn_counts)):
        x = x_positions[i]

        # y positions for the drawn nodes (evenly spaced)
        if drawn_size == 1:
            y_positions = np.array([0.0])
        else:
            y_positions = np.linspace(-(drawn_size - 1) / 2, (drawn_size - 1) / 2, drawn_size)

        # Draw neurons
        for y in y_positions:
            circ = plt.Circle((x, y), 0.2, color="#4da6ff")
            ax.add_patch(circ)

        # If we're collapsing (drawing fewer than actual), show "..." in the middle
        if layer_size > drawn_size:
            mid_y = 0.0
            ax.text(x, mid_y, "⋮", ha="center", va="center", fontsize=18, weight="bold")

        # Label layer name above
        if layer_labels:
            ax.text(x, (max_drawn / 2) + 0.8,
                    layer_labels[i],
                    ha="center", va="bottom",
                    fontsize=14, weight="bold")

        # Label neuron count below
        ax.text(x, -(max_drawn / 2) - 0.8,
                f"{layer_size} neurons",
                ha="center", va="top",
                fontsize=11)

    # Draw connections between layers (only between drawn nodes)
    for i in range(n_layers - 1):
        x1 = x_positions[i]
        x2 = x_positions[i + 1]

        drawn_size_1 = drawn_counts[i]
        drawn_size_2 = drawn_counts[i + 1]

        if drawn_size_1 == 1:
            y1_positions = np.array([0.0])
        else:
            y1_positions = np.linspace(-(drawn_size_1 - 1) / 2,
                                       (drawn_size_1 - 1) / 2,
                                       drawn_size_1)
        if drawn_size_2 == 1:
            y2_positions = np.array([0.0])
        else:
            y2_positions = np.linspace(-(drawn_size_2 - 1) / 2,
                                       (drawn_size_2 - 1) / 2,
                                       drawn_size_2)

        for y1 in y1_positions:
            for y2 in y2_positions:
                ax.plot([x1, x2], [y1, y2], color="gray", linewidth=0.6, alpha=0.7)

    # Nice bounds & aspect
    x_min = -1
    x_max = x_positions[-1] + 1
    y_min = -(max_drawn / 2) - 2
    y_max = (max_drawn / 2) + 2

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


# Example: your architecture
draw_mlp_pretty(
    sizes=[5, 32, 64, 1],
    max_nodes_to_draw=6,   # try 5–8 for slides
    figsize=(12, 6)
)
