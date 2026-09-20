import matplotlib.pyplot as plt


def save_plot(filename):

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()