import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib import ticker

from src.dataset import RawEvalArkGuesserDataset
from src.model import ArkGuesserModelV0


def regroup(stats: dict, k: int):
    """
    Regroup stats by merging every k consecutive keys.

    :param stats: dict with int keys (e.g., round numbers), values are dict[str, Any]
    :param k: number of keys per group
    :return: dict with group labels as keys, merged stats as values
    """
    if not stats:
        return {}

    sorted_keys = sorted(stats.keys())
    grouped_stats = {}

    for i in range(0, len(sorted_keys), k):
        group_keys = sorted_keys[i : i + k]
        group_label = f"{group_keys[0]}-{group_keys[-1]}"

        merged = {}
        for j in group_keys:
            for field_key, field_value in stats[j].items():
                if field_key not in merged:
                    if isinstance(field_value, list):
                        merged[field_key] = []
                    else:
                        merged[field_key] = 0
                if isinstance(field_value, list):
                    merged[field_key].extend(field_value)
                else:
                    merged[field_key] += field_value

        grouped_stats[group_label] = merged

    return grouped_stats


def visualize(model_stats: dict, human_stats: dict, k: int = 2):
    """
    Visualize model and human performance by game round, grouped every k rounds.

    :param model_stats: dict[game_round, {'correct': int, 'total': int, 'p_values': list}]
    :param human_stats: dict[game_round, {'correct': int, 'wrong': int, 'neutral': int, 'total': int}]
    :param k: number of rounds per group
    """
    # Regroup stats
    grouped_model_stats = regroup(model_stats, k)
    grouped_human_stats = regroup(human_stats, k)

    group_labels = list(grouped_model_stats.keys())
    if not group_labels:
        return

    # Calculate ratios for each group
    all_group_model_accs = []
    all_group_human_correct_p = []
    all_group_human_neutral_p = []
    all_group_human_wrong_p = []
    all_group_model_correct_p = []
    all_group_model_neutral_p = []
    all_group_model_wrong_p = []

    for group_label in group_labels:
        group_model = grouped_model_stats.get(group_label, {})
        group_human = grouped_human_stats.get(group_label, {})

        # Calculate ratios
        model_acc = group_model["correct"] / group_model["total"] if group_model["total"] > 0 else 0
        human_correct_ratio = group_human["correct"] / group_human["total"] if group_human["total"] > 0 else 0
        human_neutral_ratio = group_human["neutral"] / group_human["total"] if group_human["total"] > 0 else 0
        human_wrong_ratio = group_human["wrong"] / group_human["total"] if group_human["total"] > 0 else 0

        all_group_model_accs.append(model_acc)
        all_group_human_correct_p.append(human_correct_ratio)
        all_group_human_neutral_p.append(human_neutral_ratio)
        all_group_human_wrong_p.append(human_wrong_ratio)

        # Find P threshold for similar neutral ratio
        all_p_values = [p for p, _ in group_model["samples"]]
        if all_p_values:
            target_neutral_ratio = human_neutral_ratio
            sorted_p = sorted(all_p_values)
            n = len(sorted_p)
            best_threshold = None
            best_diff = float("inf")
            for j in range(n + 1):
                threshold = sorted_p[j] if j < n else 1.0
                neutral_ratio_model = j / n
                diff = abs(neutral_ratio_model - target_neutral_ratio)
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = threshold
                else:
                    break
            print(
                f"Group {group_label}: Human neutral ratio {human_neutral_ratio:.2%}, Model P threshold {best_threshold:.6f}"
            )

            # Calculate model neutral ratio using the threshold
            neutral_count = sum(1 for p in all_p_values if p < best_threshold)
            model_neutral_ratio = neutral_count / len(all_p_values) if all_p_values else 0

            # Calculate confident correct: correct samples with p >= threshold
            correct_samples_p = [p for p, correct in group_model["samples"] if correct]
            neutral_in_correct = sum(1 for p in correct_samples_p if p < best_threshold)
            confident_correct_count = len(correct_samples_p) - neutral_in_correct
            confident_correct_ratio = confident_correct_count / group_model["total"] if group_model["total"] > 0 else 0

            # Calculate model wrong: wrong samples with p >= threshold (to avoid double counting with uncertain)
            wrong_samples_p = [p for p, correct in group_model["samples"] if not correct]
            neutral_in_wrong = sum(1 for p in wrong_samples_p if p < best_threshold)
            confident_wrong_count = len(wrong_samples_p) - neutral_in_wrong
            confident_wrong_ratio = confident_wrong_count / group_model["total"] if group_model["total"] > 0 else 0
        else:
            model_neutral_ratio = 0
            confident_correct_ratio = model_acc  # fallback
            confident_wrong_ratio = 0  # fallback

        all_group_model_neutral_p.append(model_neutral_ratio)
        all_group_model_correct_p.append(confident_correct_ratio)
        all_group_model_wrong_p.append(confident_wrong_ratio)

    # Plot
    x = range(len(group_labels))
    width = 0.25  # Adjusted width for better spacing

    fig, ax = plt.subplots(figsize=(12, 6))

    # Positions for bars to avoid overlap
    human_pos = [i - 0.15 for i in x]
    model_confident_pos = [i + 0.15 for i in x]

    # Human bars: stacked
    ax.bar(human_pos, all_group_human_correct_p, width, label="Human Correct", color="blue", alpha=0.7)
    ax.bar(
        human_pos,
        all_group_human_neutral_p,
        width,
        bottom=all_group_human_correct_p,
        label="Human Wait-and-see",
        color="lightblue",
        alpha=0.6,
    )
    ax.bar(
        human_pos,
        all_group_human_wrong_p,
        width,
        bottom=[c + n for c, n in zip(all_group_human_correct_p, all_group_human_neutral_p)],
        label="Human Wrong",
        color="blue",
        alpha=0.6,
        hatch="//",
        edgecolor="red",
    )

    # Model confident correct bars: stacked
    ax.bar(
        model_confident_pos,
        all_group_model_correct_p,
        width,
        label="Model Correct",
        color="green",
        alpha=0.6,
    )
    ax.bar(
        model_confident_pos,
        all_group_model_neutral_p,
        width,
        bottom=all_group_model_correct_p,
        label="Model Wait-and-see",
        color="lightgreen",
        alpha=0.6,
    )
    ax.bar(
        model_confident_pos,
        all_group_model_wrong_p,
        width,
        bottom=[c + n for c, n in zip(all_group_model_correct_p, all_group_model_neutral_p)],
        label="Model Wrong",
        color="green",
        alpha=0.7,
        hatch="\\\\",
        edgecolor="red",
    )

    # Model bars: not stacked
    # Removed Our Model bar

    # Add data labels
    for i in range(len(x)):
        # Human Correct
        if all_group_human_correct_p[i] > 0:
            ax.text(
                human_pos[i],
                all_group_human_correct_p[i] / 2,
                f"{all_group_human_correct_p[i]:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        # Human Neutral
        if all_group_human_neutral_p[i] > 0:
            y_pos = all_group_human_correct_p[i] + all_group_human_neutral_p[i] / 2
            ax.text(
                human_pos[i],
                y_pos,
                f"{all_group_human_neutral_p[i]:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
        # Human Wrong
        if all_group_human_wrong_p[i] > 0:
            y_pos = all_group_human_correct_p[i] + all_group_human_neutral_p[i] + all_group_human_wrong_p[i] / 2
            ax.text(
                human_pos[i],
                y_pos,
                f"{all_group_human_wrong_p[i]:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

        # Model Confident Correct
        if all_group_model_correct_p[i] > 0:
            ax.text(
                model_confident_pos[i],
                all_group_model_correct_p[i] / 2,
                f"{all_group_model_correct_p[i]:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        # Model Uncertain
        if all_group_model_neutral_p[i] > 0:
            y_pos = all_group_model_correct_p[i] + all_group_model_neutral_p[i] / 2
            ax.text(
                model_confident_pos[i],
                y_pos,
                f"{all_group_model_neutral_p[i]:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
        # Model Wrong
        if all_group_model_wrong_p[i] > 0:
            y_pos = all_group_model_correct_p[i] + all_group_model_neutral_p[i] + all_group_model_wrong_p[i] / 2
            ax.text(
                model_confident_pos[i],
                y_pos,
                f"{all_group_model_wrong_p[i]:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

        # Our Model
        # Removed data label for Our Model

    ax.set_xlabel("Game Round Groups")
    ax.set_ylabel("Ratio")
    ax.set_title(f"Model vs Human Performance by Game Round (allowing wait-and-see)")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.set_ylim(0, 1)

    plt.tight_layout()
    import os

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/performance_comparison.png")
    plt.close()
    print("Visualization saved to outputs/performance_comparison.png")


def main(dataset_path: str, model_path: str):
    # Config
    device = torch.device("cpu")

    # Dataset
    dataset = RawEvalArkGuesserDataset(dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=RawEvalArkGuesserDataset.collate_fn,
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Load model
    model = ArkGuesserModelV0(dataset.num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")

    # Evaluate
    correct = 0
    total = 0
    import time
    from collections import defaultdict

    model_stats = defaultdict(lambda: {"correct": 0, "total": 0, "samples": []})
    human_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "neutral": 0, "total": 0})

    start_time = time.perf_counter()

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            p_value = probs.max(dim=-1).values.item()
            preds = logits.argmax(dim=-1)
            pred_correct = (preds == y).item()
            correct += pred_correct
            total += 1

            # Collect model stats by game round
            game_round = dataset.get_game_ground(i)
            model_stats[game_round]["correct"] += pred_correct
            model_stats[game_round]["total"] += 1
            model_stats[game_round]["samples"].append((p_value, pred_correct))

            # Collect human stats by game round
            h_correct, h_wrong, h_neutral = dataset.get_human_performance(i)
            human_stats[game_round]["correct"] += h_correct
            human_stats[game_round]["wrong"] += h_wrong
            human_stats[game_round]["neutral"] += h_neutral
            human_stats[game_round]["total"] += h_correct + h_wrong + h_neutral

    end_time = time.perf_counter()

    acc = correct / total
    elapsed_time = end_time - start_time
    throughput = total / elapsed_time
    print(f"Evaluation accuracy: {acc:.2%}")
    print(f"Processed {total} items in {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} it/s")

    # Visualize
    visualize(model_stats, human_stats, k=2)
