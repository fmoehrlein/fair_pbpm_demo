import os
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, PowerNorm


def _set_theme():
    sns.set_theme(
        rc={
            "figure.autolayout": False,
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            # 'text.usetex': True
        }
    )
    plt.rcParams.update(
        {
            "figure.autolayout": False,
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            # 'text.usetex': True
        }
    )
    sns.set_style(
        rc={
            "font.family": ["Computer Modern", "CMU Serif", "cmu", "serif"],
            "font.serif": ["Computer Modern", "CMU Serif", "cmu"],
            # 'text.usetex': True
        }
    )
    sns.set(font="CMU Serif", font_scale=1.25)
    plt.rcParams["font.family"] = "CMU Serif"


def plot_density(data, title):
    sns.kdeplot(data, fill=True, color="blue", alpha=0.5)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()


def plot_scatter(
        unfairness_measurements: typing.List[str],
        unfairness_label: str,
        title: str,
        folder_name: str,
        *,
        with_legend: bool = True,
        density_levels: int = 2,
        mean_only: bool = False,
        fig_size: typing.Tuple[float, float] = None,
        min_accuracy: float = None,
        max_accuracy: float = None
):
    if fig_size is None:
        fig_size = (6.4, 4.8)
    fig, ((a1, legend_ax), (acc_kde_ax, a2), (scatter_ax, fair_kde_ax)) = plt.subplots(
        nrows=3, ncols=2,
        figsize=fig_size,
        gridspec_kw={
            "hspace": 0,
            "wspace": 0,
            "height_ratios": [1.2, 1, 6],
            "width_ratios": [10, 1]
        }
    )

    # combine top row
    a1.remove()
    legend_ax.remove()

    legend_ax = fig.add_subplot(3, 2, (1, 2))

    data = pd.read_pickle(pathlib.Path("processed_data") / folder_name / "results_df.pkl")
    rows = []

    for i, model_type in enumerate(["base", "modified", "enriched"]):
        accuracy_vals = data[f"{model_type}_accuracy"]
        unfairness_vals = data[[f"{m}_{model_type}" for m in unfairness_measurements]].mean(axis=1)
        for accuracy, unfairness in zip(accuracy_vals, unfairness_vals):
            rows.append({
                "accuracy": accuracy,
                "unfairness": unfairness,
                "model": model_type
            })

    df = pd.DataFrame(rows)

    styles = {"base": ("#1f77b4", "o"), "modified": ("#2ca02c", "X"), "enriched": ("#ff7f0e", "^")}
    for model_type, (color, marker) in styles.items():
        df_slice = df[df["model"] == model_type]

        xs = df_slice["accuracy"]
        ys = df_slice["unfairness"]

        if density_levels > 1:
            sns.kdeplot(data=df_slice, x="accuracy", y="unfairness", legend=False, ax=scatter_ax, levels=density_levels,
                        fill=True, color=color, alpha=.2)
            sns.kdeplot(data=df_slice, x="accuracy", y="unfairness", legend=False, ax=scatter_ax, levels=density_levels,
                        fill=False, color=color, alpha=.4, linewidths=1)

        sns.kdeplot(y=ys, color=color, ax=fair_kde_ax, fill=True)
        sns.kdeplot(x=xs, color=color, ax=acc_kde_ax, fill=True)

        if mean_only:
            xs = np.mean(xs)
            ys = np.mean(ys)

        scatter_ax.scatter(xs, ys, color=color, marker=marker, label=model_type, alpha=0.6)

    scatter_ax.set_ylabel(unfairness_label)
    scatter_ax.set_xlabel("Accuracy")
    scatter_ax.set_ylim(-0.05, 1.05)

    # create gradient
    (min_x, max_x), (min_y, max_y) = scatter_ax.get_xlim(), scatter_ax.get_ylim()
    if min_accuracy is not None:
        min_x = min_accuracy
    if max_accuracy is not None:
        max_x = max_accuracy
    scatter_ax.set_xlim(min_x, max_x)
    gradient_colors = ["#D5E8D4", "#F8CECC"]  # low values: green, high values: red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom_cmap", gradient_colors, N=5)

    gx, gy = np.meshgrid(
        np.flip(np.linspace(0, 1, n_bins)),
        np.flip(np.linspace(min_y, max_y, n_bins))
    )
    im = np.sqrt(gx ** 2 + gy ** 2)
    vmin = 0
    vmax = 1

    scatter_ax.imshow(
        im, cmap=cmap, aspect='auto',
        extent=[min_x, max_x, min_y, max_y],
        # vmin=vmin, vmax=vmax,
        norm=PowerNorm(gamma=2, vmin=vmin, vmax=vmax),
        zorder=-1, alpha=1
    )

    fair_kde_ax.set_xlim(fair_kde_ax.get_xlim()[0] - 1, fair_kde_ax.get_xlim()[1] + 5)
    fair_kde_ax.set_ylim(scatter_ax.get_ylim())
    fair_kde_ax.set_yticks([])
    fair_kde_ax.set_xticks([])
    fair_kde_ax.set_xlabel("")
    fair_kde_ax.set_ylabel("")
    fair_kde_ax.grid(visible=False)
    fair_kde_ax.set_facecolor("white")

    acc_kde_ax.set_xlim(scatter_ax.get_xlim())
    acc_kde_ax.set_ylim(acc_kde_ax.get_ylim()[0] - 1, acc_kde_ax.get_ylim()[1] + 5)
    acc_kde_ax.set_yticks([])
    acc_kde_ax.set_xticks([])
    acc_kde_ax.set_xlabel("")
    acc_kde_ax.set_ylabel("")
    acc_kde_ax.grid(visible=False)
    acc_kde_ax.set_facecolor("white")

    a2.grid(visible=False)
    a2.set_facecolor("white")
    a2.set_xticks([])
    a2.set_yticks([])
    a2.set_xlabel("")
    a2.set_ylabel("")

    legend_ax.grid(visible=False)
    legend_ax.set_facecolor("white")
    legend_ax.set_yticks([])
    legend_ax.set_xticks([])
    if with_legend:
        legend_handles, legend_labels = scatter_ax.get_legend_handles_labels()
        legend_ax.legend(legend_handles, legend_labels, ncols=3, labelspacing=0, borderpad=0.2)

    fig.suptitle(title)

    plt.tight_layout()
    (pathlib.Path("img") / folder_name).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join("img", folder_name, f"{folder_name}-accuracy-vs-fairness.png"))
    plt.savefig(os.path.join("img", folder_name, f"{folder_name}-accuracy-vs-fairness.pdf"))
    plt.close()


def plot_distribution(base_values, enriched_values, modified_values, folder_name, title, measurement, fig_size=None,
                      tick_format: str = None):
    if tick_format is None:
        tick_format = ".3f"
    if fig_size is None:
        fig_size = (6.4, 4.8)
    img_folder = os.path.join("img", folder_name)
    os.makedirs(img_folder, exist_ok=True)

    data = {
        'Base': np.array(base_values),
        'Modified': np.array(modified_values),
        'Enriched': np.array(enriched_values),
    }

    # Calculate mean and standard deviation
    stats = {name: (np.mean(values), np.std(values)) for name, values in data.items()}
    print(f"{title} Statistics:")
    for name, (mean, std) in stats.items():
        print(f"{name}: Mean = {mean:.3f}, Std Dev = {std:.3f}")

    # Create the plot
    plt.figure(figsize=fig_size)
    fig, (legend_ax, kde_ax, scatter_ax) = plt.subplots(nrows=3, ncols=1,
                                                        gridspec_kw={"height_ratios": [1.25, 7, 1], "hspace": 0},
                                                        sharex=True)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Define custom colors
    for i, (name, values) in enumerate(data.items()):
        sns.kdeplot(values, label=f'{name}',  # (μ={stats[name][0]:.1%}, σ={stats[name][1]:.1%})
                    color=colors[i], fill=True, alpha=0.6, linewidth=2, ax=kde_ax)
        scatter_ax.scatter(values, np.random.uniform(low=-1, high=1, size=len(values)), color=colors[i], alpha=0.6)

    # Add titles and labels
    fig.suptitle(title, fontweight='bold')

    scatter_ax.set_xlabel(measurement)
    x_ticks = np.linspace(kde_ax.get_xlim()[0], kde_ax.get_xlim()[1], 5)
    scatter_ax.set_xticks(x_ticks, [f"{x:{tick_format}}" for x in x_ticks])
    scatter_ax.set_ylabel("")
    scatter_ax.set_yticks([])
    scatter_ax.set_ylim([-3, 3])
    scatter_ax.grid(visible=True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)

    kde_ax.set_ylabel('Density')
    # kde_ax.set_ylim(0, 1.25 * kde_ax.get_ylim()[1])
    kde_ax.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    legend_ax.grid(visible=False)
    legend_ax.set_facecolor("white")
    legend_ax.set_yticks([])
    legend_handles, legend_labels = kde_ax.get_legend_handles_labels()
    legend_ax.legend(legend_handles, legend_labels, ncols=3)

    # Save the plot
    measurement_slug = measurement.replace(" ", "-").replace("(", "_").replace(")", "")
    plt.tight_layout()
    plt.savefig(os.path.join("img", folder_name, f"{folder_name}-{measurement_slug}.png"))
    plt.savefig(os.path.join("img", folder_name, f"{folder_name}-{measurement_slug}.pdf"))
    plt.close()


def plot_attributes(df: pd.DataFrame, rules: list, folder_name: str):
    img_folder = os.path.join("img", folder_name)
    os.makedirs(img_folder, exist_ok=True)
    # Group by case_id to ensure each case is only counted once
    grouped = df.groupby('case_id')

    # Collect unique attributes and their rules
    attribute_rules = {}
    for rule in rules:
        attribute = rule['attribute']
        if attribute not in attribute_rules:
            attribute_rules[attribute] = []
        attribute_rules[attribute].append(rule)

    for attribute, rules in attribute_rules.items():
        # Combine data for the attribute
        attribute_values = grouped[attribute].first().dropna()

        # Start plotting
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Handle discrete attributes
        if any(rule['distribution']['type'] == 'discrete' for rule in rules):
            # Discrete values and their labels
            discrete_values = []
            for rule in rules:
                if rule['distribution']['type'] == 'discrete':
                    values, _ = zip(*rule['distribution']['values'])
                    discrete_values.extend(values)
            discrete_values = list(set(discrete_values))  # Remove duplicates

            # Calculate percentages
            counts = attribute_values.value_counts(normalize=True).reindex(discrete_values, fill_value=0)
            counts *= 100  # Convert to percentages

            # Bar plot for discrete attributes
            sns.barplot(x=counts.index, y=counts.values, palette="viridis", saturation=0.9)
            plt.title(f"Distribution of {attribute} (Discrete, per Case)", fontsize=16, fontweight="bold")
            plt.xlabel("Value", fontsize=14)
            plt.ylabel("Percentage (%)", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            for i, v in enumerate(counts.values):
                plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10, fontweight="bold")

        # Handle continuous attributes
        elif any(rule['distribution']['type'] == 'normal' for rule in rules):
            # Calculate number of bins (using Sturges' rule)
            bins = int(np.ceil(np.log2(len(attribute_values))) + 1)

            # Histogram for continuous attributes
            sns.histplot(attribute_values, bins=bins, kde=True, color='mediumvioletred', alpha=0.7)
            plt.title(f"Distribution of {attribute} (Continuous, per Case)", fontsize=16, fontweight="bold")
            plt.xlabel("Value", fontsize=14)
            plt.ylabel("Percentage (%)", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # Display percentages on the histogram bars
            n, bin_edges = np.histogram(attribute_values, bins=bins)
            percentages = (n / n.sum()) * 100
            for i in range(len(n)):
                plt.text(bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2, percentages[i] + 0.5,
                         f"{percentages[i]:.1f}%", ha='center', fontsize=10, fontweight="bold")

        else:
            print(f"Unsupported distribution type for attribute '{attribute}'. Skipping.")
            continue

        # Add final touches
        plt.tight_layout()
        plt.savefig(os.path.join('img', folder_name, f"{attribute}.png"))
        plt.savefig(os.path.join('img', folder_name, f"{attribute}.pdf"))
        plt.close()


def plot_ablation(x_values, base_metrics, enriched_metrics, modified_metrics, title, x_label, y_label, folder_name,
                  override_x_ticks=True):
    # Define custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Prepare data for statistical summaries
    data = {
        'Base': np.array(base_metrics),
        'Enriched': np.array(enriched_metrics),
        'Modified': np.array(modified_metrics)
    }

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each line with customized styles
    for i, (name, values) in enumerate(data.items()):
        plt.plot(x_values, values, label=f'{name}',
                 color=colors[i], marker=['o', 's', '^'][i], linestyle=['-', '--', '-.'][i],
                 linewidth=2, markersize=8, alpha=0.9)

    # Add titles, labels, and grid
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    if override_x_ticks:
        plt.xticks(ticks=x_values, labels=x_values, fontsize=12)
    else:
        plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add legend with title
    plt.legend(fontsize=12, title='Legend', title_fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Adjust layout and show the plot
    plt.tight_layout()
    img_folder = os.path.join("img", folder_name)
    os.makedirs(img_folder, exist_ok=True)
    plt.savefig(os.path.join('img', folder_name, f"{title}.png"))
    plt.savefig(os.path.join('img', folder_name, f"{title}.pdf"))
    plt.close()


def plot_ablation_alternative(experiments: typing.Dict[str, str], title: str,
                              fig_size: typing.Tuple[float, float] = None):
    if fig_size is None:
        fig_size = (6.4, 4.8)

    num_rows = 4
    num_cols = len(experiments)

    fig, (legend_axs, title_axs, acc_axs, dp_axs) = plt.subplots(
        num_rows, num_cols,
        figsize=fig_size,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.5,
            "height_ratios": [1.5, 1.5, 8, 4],
        },
        #sharex="col"
    )

    for dp_ax in dp_axs[1:]:
        dp_ax.sharey(dp_axs[0])

    sup_title = fig.suptitle("Ablation Study Results")

    # combine first row into single plot for legend
    for ax in legend_axs:
        ax.remove()
    legend_ax = fig.add_subplot(num_rows, num_cols, (1, num_cols))

    # make legend axis and margin axes white
    legend_ax.grid(visible=False)
    legend_ax.set_facecolor("white")
    legend_ax.set_yticks([])
    legend_ax.set_xticks([])

    for ax in title_axs:
        ax.grid(visible=False)
        ax.set_facecolor("white")
        ax.set_yticks([])
        ax.set_xticks([])

    axis_titles = []

    for i, (experiment, param_name) in enumerate(experiments.items()):
        acc_ax = acc_axs[i]
        dp_ax = dp_axs[i]
        title_ax = title_axs[i]

        data = pd.read_pickle(pathlib.Path("processed_data") / experiment / "results_df.pkl")
        rows = []
        for model_type in ["base", "modified", "enriched"]:
            accuracy_vals = data[f"{model_type}_accuracy"]
            unfairness_vals = data[f"dp_{model_type}"]
            param_vals = data["param"]
            for accuracy, unfairness, param in zip(accuracy_vals, unfairness_vals, param_vals):
                rows.append({
                    "accuracy": accuracy,
                    "unfairness": unfairness,
                    "param": param,
                    "model": model_type
                })
        df = pd.DataFrame(rows)

        styles = {"base": ("#1f77b4", "o"), "modified": ("#2ca02c", "X"), "enriched": ("#ff7f0e", "^")}
        for model_type, (color, marker) in styles.items():
            df_slice = df[df["model"] == model_type]
            sns.lineplot(x=df_slice["param"], y=df_slice["accuracy"], label=model_type, color=color, marker=marker,
                         ax=acc_ax, legend=False)
            sns.lineplot(x=df_slice["param"], y=df_slice["unfairness"], label=model_type, color=color, marker=marker,
                         ax=dp_ax, legend=False)

        if i == 0:
            dp_ax.set_ylabel("ΔDP")
            acc_ax.set_ylabel("Accuracy")
        else:
            dp_ax.set_ylabel("")
            acc_ax.set_ylabel("")

        dp_ax.set_xlabel(param_name)
        # axis titles are not considered by bbox_inches="tight"... work around that
        title_ax.text(0.5, 0.5, f"{chr(i + 97)})", transform=title_ax.transAxes)

    legend_handles, legend_labels = acc_ax.get_legend_handles_labels()
    legend_ax.legend(legend_handles, legend_labels, ncols=3, labelspacing=0, borderpad=0.2)

    # plt.tight_layout()
    img_folder = os.path.join("img", "ablation")
    os.makedirs(img_folder, exist_ok=True)
    plt.savefig(os.path.join(img_folder, f"{title}.png"), bbox_inches="tight")
    plt.savefig(os.path.join(img_folder, f"{title}.pdf"), bbox_inches="tight")
    plt.close()


def plot_all():
    def main():
        plot_ablation_alternative(
        experiments={"ablation_attributes": "Num. attributes", "ablation_bias": "Bias strength",
                         "ablation_decisions": "Num. decisions"}, title="all_ablations", fig_size=(11.0, 5.0))
        experiments = {
            "cs": "Cancer Screening",
            "hb_+age_+gender": "Hospital Billing - Age Bias, Gender Bias",
            "hb_+age_-gender": "Hospital Billing - Age Bias",
            "hb_-age_+gender": "Hospital Billing - Gender Bias",
            "hb_-age_-gender": "Hospital Billing - No Bias",
            "bpi_2012": "BPI Challenge 2012"
        }
        processed_data_dir = pathlib.Path("processed_data")
        for experiment, title in experiments.items():
            if experiment.startswith("hb_"):
                plot_scatter(
                    folder_name=experiment,
                    unfairness_measurements=["dp_('gender = non conforming', 'CODE OK')", "dp_('age', 'CODE OK')"],
                    unfairness_label="ΔDP (gender & age)",
                    title=experiments[experiment],
                    density_levels=0,
                    fig_size=(4.8, 4.8),
                    with_legend=experiment in ["hb_+age_+gender", "hb_-age_+gender"],
                    min_accuracy=0.909,
                    max_accuracy=0.935
                )
            if experiment == "bpi_2012":
                plot_scatter(
                    folder_name=experiment,
                    unfairness_measurements=["dp_('gender = male', 'A_PREACCEPTED')"],
                    unfairness_label="ΔDP (gender)",
                    title=experiments[experiment],
                    density_levels=0,
                    fig_size=(4.8, 4.8),
                    with_legend=True,
                )

            if experiment == "cs":
                plot_scatter(
                    folder_name=experiment,
                    unfairness_measurements=["dp_('gender = male', 'collect history')"],
                    unfairness_label="ΔDP (gender)",
                    title=experiments[experiment],
                    density_levels=0,
                    fig_size=(4.8, 4.8),
                    with_legend=False
                )

            data = pd.read_pickle(processed_data_dir / experiment / "results_df.pkl")
            plot_distribution(
                base_values=data["base_accuracy"],
                enriched_values=data["enriched_accuracy"],
                modified_values=data["modified_accuracy"],
                folder_name=experiment,
                measurement="Accuracy",
                title=title,
                tick_format=".0%"
            )

            if experiment.startswith("hb_"):
                plot_distribution(
                    base_values=data["dp_('gender = non conforming', 'CODE OK')_base"],
                    enriched_values=data["dp_('gender = non conforming', 'CODE OK')_enriched"],
                    modified_values=data["dp_('gender = non conforming', 'CODE OK')_modified"],
                    folder_name=experiment,
                    measurement="Demographic Parity (gender)",
                    title=title,
                    tick_format=".1"
                )
                plot_distribution(
                    base_values=data["dp_('age', 'CODE OK')_base"],
                    enriched_values=data["dp_('age', 'CODE OK')_enriched"],
                    modified_values=data["dp_('age', 'CODE OK')_modified"],
                    folder_name=experiment,
                    measurement="Demographic Parity (age)",
                    title=title,
                    tick_format=".1"
                )

            if experiment == "bpi_2012":
                plot_distribution(
                    base_values=data["dp_('gender = male', 'A_PREACCEPTED')_base"],
                    enriched_values=data["dp_('gender = male', 'A_PREACCEPTED')_enriched"],
                    modified_values=data["dp_('gender = male', 'A_PREACCEPTED')_modified"],
                    folder_name=experiment,
                    measurement="Demographic Parity (gender)",
                    title=title,
                    tick_format=".1"
                )

            if experiment == "cs":
                plot_distribution(
                    base_values=data["dp_('gender = male', 'collect history')_base"],
                    enriched_values=data["dp_('gender = male', 'collect history')_enriched"],
                    modified_values=data["dp_('gender = male', 'collect history')_modified"],
                    folder_name=experiment,
                    measurement="Demographic Parity (gender)",
                    title=title,
                    tick_format=".1"
                )


    _set_theme()
    main()
