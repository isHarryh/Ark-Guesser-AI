import json
import os
import statistics
import urllib.parse
from pathlib import Path
from typing import Any
from html import escape as html_escape
from dataclasses import dataclass, field

import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from dash.dash_table import FormatTemplate, Format


def _coerce_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {i: str(name) for i, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        id_to_name: dict[int, str] = {}
        name_to_id: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                idx = int(key)
                id_to_name[idx] = str(value)
                continue
            except (TypeError, ValueError):
                pass

            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            name_to_id[idx] = str(key)

        if id_to_name:
            return id_to_name
        if name_to_id:
            return name_to_id
    return {}


def _load_dataset(dataset_path: str) -> tuple[str, dict[int, str], list[dict[str, Any]]]:
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Dataset JSON root must be an object")

    version = str(raw.get("version", ""))
    names = _coerce_names(raw.get("names"))
    data = raw.get("data", [])
    if not isinstance(data, list):
        raise ValueError("Dataset JSON 'data' must be a list")
    return version, names, data


def _format_number(value: float) -> float:
    if abs(value - round(value)) > 1e-6:
        return value
    return float(int(round(value)))


def _avatar_markdown(name: str) -> str:
    if not name:
        return ""
    path = f"/assets/avatars_resized/{urllib.parse.quote(name)}.png"
    return f'<img src="{path}" height="64" alt="{html_escape(name)}" />'


@dataclass
class ClassStats:
    total_qty: int = 0
    appearances: int = 0
    wins: int = 0
    rounds: list[int] = field(default_factory=list)


@dataclass
class ReportData:
    version: str
    names: dict[int, str]
    entries: list[dict[str, Any]]
    total_qty_matches: list[int] = field(default_factory=list)
    win_quantities: list[int] = field(default_factory=list)
    lose_quantities: list[int] = field(default_factory=list)
    class_stats: dict[int, ClassStats] = field(default_factory=dict)
    rounds: list[int] = field(default_factory=list)
    human_correct: list[int] = field(default_factory=list)
    human_wrong: list[int] = field(default_factory=list)
    human_neutral: list[int] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.entries)

    @property
    def has_eval(self) -> bool:
        return bool(self.rounds)


class ReportBuilder:
    def __init__(self, dataset_path: str) -> None:
        version, names, entries = _load_dataset(dataset_path)
        self._dataset_path = dataset_path
        self._data = ReportData(version=version, names=names, entries=entries)

    @property
    def data(self) -> ReportData:
        return self._data

    def collect(self) -> None:
        for entry in self._data.entries:
            if not isinstance(entry, dict):
                continue
            groups = entry.get("groups", [])
            if not (isinstance(groups, list) and len(groups) == 2):
                continue

            match_sum = 0
            match_classes: set[int] = set()
            side_sums = [0, 0]
            winner = entry.get("winner")
            game_round = entry.get("game_round")
            has_round = isinstance(game_round, int)

            for side in range(2):
                group = groups[side]
                if not isinstance(group, dict):
                    continue
                for key, value in group.items():
                    try:
                        idx = int(key)
                        qty = int(value)
                    except (TypeError, ValueError):
                        continue
                    if qty <= 0:
                        continue

                    stats = self._data.class_stats.setdefault(idx, ClassStats())
                    stats.total_qty += qty
                    stats.appearances += 1
                    if isinstance(winner, int) and winner == side:
                        stats.wins += 1

                    side_sums[side] += qty
                    match_sum += qty
                    match_classes.add(idx)

            if match_classes:
                self._data.total_qty_matches.append(match_sum)
                if isinstance(winner, int) and winner in (0, 1) and len(self._data.win_quantities) < 2500:
                    self._data.win_quantities.append(side_sums[winner])
                    self._data.lose_quantities.append(side_sums[1 - winner])

            if has_round:
                for idx in match_classes:
                    self._data.class_stats[idx].rounds.append(game_round)

            if all(k in entry for k in ("game_round", "human_correct", "human_wrong", "human_neutral")):
                try:
                    self._data.rounds.append(int(entry["game_round"]))
                    self._data.human_correct.append(int(entry["human_correct"]))
                    self._data.human_wrong.append(int(entry["human_wrong"]))
                    self._data.human_neutral.append(int(entry["human_neutral"]))
                except (TypeError, ValueError):
                    pass

    def build_summary(self) -> list[tuple[str, str]]:
        return [
            ("Dataset Name", os.path.basename(self._dataset_path)),
            ("Dataset Version", self._data.version or "(unknown)"),
            ("Samples", str(self._data.sample_count)),
            ("Classes", str(len(self._data.class_stats))),
            ("Has Eval Fields", "Yes" if self._data.has_eval else "No"),
        ]

    def build_figures(self) -> list[go.Figure]:
        figs: list[go.Figure] = []

        if self._data.total_qty_matches:
            qty_fig = go.Figure(
                data=[go.Histogram(x=self._data.total_qty_matches, marker_color="#1f77b4", opacity=0.85)]
            )
            qty_fig.update_layout(
                title="Total Unit Quantity per Match", xaxis_title="Unit Quantity", yaxis_title="Samples"
            )
            figs.append(qty_fig)

        if self._data.win_quantities and self._data.lose_quantities:
            scatter_fig = go.Figure(
                data=[
                    go.Scatter(
                        x=self._data.lose_quantities,
                        y=self._data.win_quantities,
                        mode="markers",
                        marker=dict(size=6, color="#2ca02c", opacity=0.6),
                        name="Match",
                    )
                ]
            )
            if len(self._data.win_quantities) >= 2:
                x_mean = sum(self._data.lose_quantities) / len(self._data.lose_quantities)
                y_mean = sum(self._data.win_quantities) / len(self._data.win_quantities)
                var_x = sum((x - x_mean) ** 2 for x in self._data.lose_quantities)
                if var_x > 0:
                    cov_xy = sum(
                        (x - x_mean) * (y - y_mean)
                        for x, y in zip(self._data.lose_quantities, self._data.win_quantities)
                    )
                    slope = cov_xy / var_x
                    intercept = y_mean - slope * x_mean
                    x_min = min(self._data.lose_quantities)
                    x_max = max(self._data.lose_quantities)
                    scatter_fig.add_trace(
                        go.Scatter(
                            x=[x_min, x_max],
                            y=[slope * x_min + intercept, slope * x_max + intercept],
                            mode="lines",
                            line=dict(color="#1f1f1f", width=2),
                            name="Fit",
                        )
                    )
            scatter_fig.update_layout(
                title="Win-Lose Unit Quantity Distribution",
                xaxis_title="Lose Unit Quantity",
                yaxis_title="Win Unit Quantity",
                showlegend=False,
            )
            figs.append(scatter_fig)

        if self._data.has_eval:
            per_round: dict[int, dict[str, int]] = {}
            for r, c, w, n in zip(
                self._data.rounds,
                self._data.human_correct,
                self._data.human_wrong,
                self._data.human_neutral,
            ):
                bucket = per_round.setdefault(r, {"correct": 0, "wrong": 0, "neutral": 0})
                bucket["correct"] += c
                bucket["wrong"] += w
                bucket["neutral"] += n

            total_correct = sum(self._data.human_correct)
            total_wrong = sum(self._data.human_wrong)
            total_neutral = sum(self._data.human_neutral)
            total_humans = total_correct + total_wrong + total_neutral
            if total_humans <= 0:
                total_correct = total_wrong = total_neutral = 0
                total_humans = 1
            overall_rates = [
                total_correct / total_humans,
                total_neutral / total_humans,
                total_wrong / total_humans,
            ]

            overall_fig = go.Figure(
                data=[
                    go.Bar(
                        name="Correct",
                        y=["Overall"],
                        x=[overall_rates[0]],
                        orientation="h",
                        marker_color="#2ca02c",
                    ),
                    go.Bar(
                        name="Neutral",
                        y=["Overall"],
                        x=[overall_rates[1]],
                        orientation="h",
                        marker_color="#7f7f7f",
                    ),
                    go.Bar(
                        name="Wrong",
                        y=["Overall"],
                        x=[overall_rates[2]],
                        orientation="h",
                        marker_color="#d62728",
                    ),
                ]
            )
            overall_fig.update_layout(
                title="Overall Human Outcomes",
                barmode="stack",
                xaxis_tickformat=".0%",
                xaxis_title="Rate",
                yaxis_title="",
                legend_orientation="h",
                legend_yanchor="bottom",
                legend_y=1.01,
                legend_xanchor="left",
                legend_x=0,
            )
            figs.append(overall_fig)

            round_keys = sorted(per_round.keys())
            wrong_rate = []
            neutral_rate = []
            correct_rate = []
            for r in round_keys:
                bucket = per_round[r]
                total = bucket["correct"] + bucket["wrong"] + bucket["neutral"]
                if total <= 0:
                    wrong_rate.append(0)
                    neutral_rate.append(0)
                    correct_rate.append(0)
                else:
                    wrong_rate.append(bucket["wrong"] / total)
                    neutral_rate.append(bucket["neutral"] / total)
                    correct_rate.append(bucket["correct"] / total)

            perf_fig = go.Figure(
                data=[
                    go.Bar(name="Wrong", x=[str(r) for r in round_keys], y=wrong_rate, marker_color="#d62728"),
                    go.Bar(name="Neutral", x=[str(r) for r in round_keys], y=neutral_rate, marker_color="#7f7f7f"),
                    go.Bar(name="Correct", x=[str(r) for r in round_keys], y=correct_rate, marker_color="#2ca02c"),
                ]
            )
            perf_fig.update_layout(
                title="Human Outcomes by Round",
                barmode="stack",
                yaxis_title="Rate",
                yaxis_tickformat=".0%",
            )
            figs.append(perf_fig)

        return figs

    def build_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        total_side_slots = self._data.sample_count * 2 if self._data.sample_count > 0 else 0
        for class_id in sorted(self._data.class_stats.keys()):
            stats = self._data.class_stats[class_id]
            appearances = stats.appearances
            total_qty = stats.total_qty
            win_rate = stats.wins / appearances if appearances > 0 else 0.0
            appearance_rate = appearances / total_side_slots if total_side_slots > 0 else 0.0
            avg_qty = total_qty / appearances if appearances > 0 else 0.0

            round_values = stats.rounds
            if round_values:
                median_round = statistics.median(round_values)
                median_round_value: float | None = _format_number(float(median_round))
            else:
                median_round_value = None

            name = self._data.names.get(class_id, f"class_{class_id}")
            rows.append(
                {
                    "class_id": class_id,
                    "name": name,
                    "avatar": _avatar_markdown(name),
                    "appearances": appearances,
                    "appearance_rate": appearance_rate,
                    "median_round": median_round_value,
                    "avg_qty": _format_number(avg_qty),
                    "win_rate": win_rate,
                }
            )

        rows.sort(key=lambda row: int(row["appearances"]), reverse=True)
        return rows


def _build_report(dataset_path: str) -> tuple[str, list[tuple[str, str]], list[go.Figure], list[dict[str, Any]]]:
    builder = ReportBuilder(dataset_path)
    builder.collect()
    title = "Ark Guesser Dataset Report"
    summary = builder.build_summary()
    figs = builder.build_figures()
    rows = builder.build_rows()
    return title, summary, figs, rows


def _build_app(title: str, summary: list[tuple[str, str]], figs: list[go.Figure], rows: list[dict[str, str]]) -> Dash:
    repo_root = Path(__file__).resolve().parents[1]
    assets_dir = repo_root / "assets"
    app = Dash(
        __name__,
        title=title,
        assets_folder=str(assets_dir),
        assets_url_path="/assets",
    )
    app.index_string = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  {{%metas%}}
  <title>{title}</title>
  {{%favicon%}}
  {{%css%}}
  <style>
  :root {{
    color-scheme: light;
    --bg: #f5f3ef;
    --card: #ffffff;
    --text: #1f1f1f;
    --muted: #5a5a5a;
  }}
  body {{
    margin: 0;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    background: radial-gradient(circle at top left, #ffffff, #f1ece6 55%, #e7e0d6 100%);
    color: var(--text);
  }}
  .page {{
    padding: 24px 6vw 32px 6vw;
  }}
  .summary {{
    background: var(--card);
    border-radius: 16px;
    padding: 16px 24px;
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
  }}
  .summary ul {{
    margin: 0;
    padding-left: 20px;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(320px, 1fr));
    gap: 20px;
    margin-top: 20px;
  }}
  .card {{
    background: var(--card);
    border-radius: 16px;
    padding: 12px;
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08);
  }}
  .table-card {{
    background: var(--card);
    border-radius: 16px;
    padding: 12px;
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08);
    margin-top: 20px;
  }}
  .muted {{
    color: var(--muted);
  }}
  </style>
</head>
<body>
  {{%app_entry%}}
  {{%config%}}
  {{%scripts%}}
  {{%renderer%}}
</body>
</html>"""

    table = dash_table.DataTable(
        columns=[
            {
                "name": "Class ID",
                "id": "class_id",
                "type": "numeric",
            },
            {
                "name": "Name",
                "id": "name",
            },
            {
                "name": "Avatar",
                "id": "avatar",
                "presentation": "markdown",
            },
            {
                "name": "Appearances (Both Sides)",
                "id": "appearances",
                "type": "numeric",
            },
            {
                "name": "Appearance Rate",
                "id": "appearance_rate",
                "type": "numeric",
                "format": FormatTemplate.percentage(2),
            },
            {
                "name": "Median Round",
                "id": "median_round",
                "type": "numeric",
            },
            {
                "name": "Avg Quantity",
                "id": "avg_qty",
                "type": "numeric",
                "format": Format.Format(precision=2, scheme="f"),
            },
            {
                "name": "Win Rate",
                "id": "win_rate",
                "type": "numeric",
                "format": FormatTemplate.percentage(2),
            },
        ],  # type: ignore
        data=rows,  # type: ignore
        sort_action="native",
        filter_action="native",
        page_size=100,
        style_table={"overflowX": "auto", "borderRadius": "12px"},
        style_cell={
            "padding": "8px 10px",
            "fontFamily": '"Segoe UI Variable", "Segoe UI", "Noto Sans SC", sans-serif',
            "fontSize": "18px",
            "lineHeight": "1.3",
            "border": "1px solid #e6e0d8",
            "backgroundColor": "#ffffff",
        },
        style_header={
            "fontWeight": "700",
            "backgroundColor": "#f3eee7",
            "fontSize": "16px",
            "border": "1px solid #e6e0d8",
        },
        style_filter={
            "backgroundColor": "#fbf9f6",
            "fontSize": "14px",
            "border": "1px solid #e6e0d8",
        },
        style_data={
            "backgroundColor": "#ffffff",
        },
        style_data_conditional=[
            {"if": {"column_id": "class_id"}, "width": "64px", "maxWidth": "64px"},
            {"if": {"column_id": "name"}, "width": "128px", "maxWidth": "256px"},
            {"if": {"column_id": "avatar"}, "width": "64px"},
            {"if": {"column_id": "appearances"}, "width": "128px"},
            {"if": {"column_id": "appearance_rate"}, "width": "128px"},
            {"if": {"column_id": "median_round"}, "width": "128px"},
            {"if": {"column_id": "avg_qty"}, "width": "128px"},
            {"if": {"column_id": "win_rate"}, "width": "128px"},
            {
                "if": {"column_type": "numeric"},
                "textAlign": "right",
                "fontVariantNumeric": "tabular-nums",
            },
            {
                "if": {"state": "active"},
                "backgroundColor": "#fff8e8",
                "border": "1px solid #e6d7bf",
            },
            {
                "if": {"state": "selected"},
                "backgroundColor": "#fdecc9",
                "border": "1px solid #e6d7bf",
            },
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#faf7f2",
            },
        ],  # type: ignore
        markdown_options={"html": True},
    )

    app.layout = html.Div(
        className="page",
        children=[
            html.H1(title),
            html.P("Interactive dataset report for Ark Guesser AI.", className="muted"),
            html.Div(
                className="summary",
                children=[
                    html.H2("Summary"),
                    html.Ul([html.Li([html.Strong(f"{k}: "), v]) for k, v in summary]),
                ],
            ),
            html.Div(
                className="grid",
                children=[html.Div(dcc.Graph(figure=fig), className="card") for fig in figs],
            ),
            html.Div(
                className="table-card",
                children=[
                    html.H2("Class Metrics"),
                    table,
                ],
            ),
        ],
    )
    return app


def main(dataset_path: str, host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    title, summary, figs, rows = _build_report(dataset_path)
    app = _build_app(title, summary, figs, rows)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    raise SystemExit("Use main.py dataset_visualize to run this module.")
