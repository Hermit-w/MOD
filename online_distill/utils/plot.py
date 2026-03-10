from matplotlib import pyplot as plt
import numpy as np
import yaml

AVAI_COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
AVAI_LINESTYLES = ['-', '--', '-.', ':']
AVAI_MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '+']


def plot_curve(
    curves: list[dict],
    save_path: str,
    title: str = "Curve Plot",
    xlabel: str = "Steps",
    ylabel: str = "Values"
):
    """
    Plot multiple curves with uniform length sequences (aligned on the x-axis).

    Args:
        curves: A list of dictionaries, where each dictionary represents a curve and contains:
            - 'data': The data of the curve (list or numpy array, required)
            - 'label': The label of the curve (str, required)
            - 'color': Line color (str, e.g., 'r', 'g', 'b', optional)
            - 'linestyle': Line style (str, e.g., '-', '--', '-.', ':', optional)
            - 'marker': Data point marker (str, e.g., 'o', 's', '^', optional)
        save_path: The file path to save the plot (str)
        title: The title of the plot (str)
        xlabel: The x-axis label (str)
        ylabel: The y-axis label (str)
    """
    plt.figure(figsize=(10, 6))

    for curve in curves:
        data = curve.get('data')
        if data is None:
            print(f"Warning: Missing 'data' key in curve dictionary, skipping: {curve}")
            continue

        # Assuming all curves have the same length, or generating x-axis coordinates automatically
        x = np.arange(len(data))

        plt.plot(
            x,
            data,
            label=curve.get('label', ''),
            color=curve.get('color', None),
            linestyle=curve.get('linestyle', '-'),
            marker=curve.get('marker', None)
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def parse_yaml(file_path: str) -> list[dict]:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    res_data = []
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], list):
                for idx, sub_value in enumerate(value):
                    line_style = AVAI_LINESTYLES[(len(res_data)) % len(AVAI_LINESTYLES)]
                    color = AVAI_COLORS[(len(res_data)) % len(AVAI_COLORS)]
                    marker = AVAI_MARKERS[(len(res_data)) % len(AVAI_MARKERS)]
                    res_data.append({'label': f"{key}_{idx}", 'data': sub_value, 'linestyle': line_style, 'color': color, 'marker': marker})
            else:
                line_style = AVAI_LINESTYLES[len(res_data) % len(AVAI_LINESTYLES)]
                color = AVAI_COLORS[len(res_data) % len(AVAI_COLORS)]
                marker = AVAI_MARKERS[len(res_data) % len(AVAI_MARKERS)]
                res_data.append({'label': key, 'data': value, 'linestyle': line_style, 'color': color, 'marker': marker})
        else:
            raise ValueError(f"Expected a list for key '{key}', but got: {value}")
    return res_data


if __name__ == "__main__":
    path_to_yaml = "/export/home/lanliwei.1/code/MOD/saving_results/offline_single_drafter/2026_03_09_17_50_50/spec_metrics.yaml"
    data = parse_yaml(path_to_yaml)
    plot_curve(data, "/export/home/lanliwei.1/code/MOD/saving_results/offline_single_drafter/2026_03_09_17_50_50/spec_metrics.png")
