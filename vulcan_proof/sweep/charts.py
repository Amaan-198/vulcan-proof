"""Phase-4 chart factories with a mandatory simulator-result footer."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..errors import InvariantError
from ..params import P, Params


ChartFactory = Callable[[Any, Params], Figure]


def _rows(data: Any, key: str) -> list[Mapping[str, Any]]:
    """Read a list-valued chart section, accepting an empty section."""
    if isinstance(data, Mapping) and key in data:
        value = data[key]
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    return []


def _figure(title: str, params: Params) -> tuple[Figure, Any]:
    """Create a titled figure and attach the required footer."""
    figure, axis = plt.subplots()
    axis.set_title(title)
    figure.text(1.0 / 2.0, 1.0 / 100.0, str(params["report.simulator_footer"]), ha="center")
    return figure, axis


def figure_footer(figure: Figure) -> str:
    """Return the simulator footer attached to a figure."""
    footer = str(P["report.simulator_footer"])
    for item in figure.texts:
        if item.get_text() == footer:
            return footer
    raise InvariantError("chart is missing the simulator footer")


def _series(rows: Sequence[Mapping[str, Any]], path: Sequence[str], default: float = 0.0) -> np.ndarray:
    """Extract a numeric series from nested row mappings."""
    values: list[float] = []
    for row in rows:
        current: Any = row
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = default
                break
            current = current[key]
        values.append(float(current))
    return np.asarray(values, dtype="float64")


def plot_kappa_net(data: Any, params: Params = P) -> Figure:
    """Plot net value by κ for the available arms."""
    figure, axis = _figure("Net ₹/1,000 vs κ", params)
    rows = _rows(data, "kappa_table")
    x = _series(rows, ("kappa",))
    for name in ("arm1", "arm3", "arm4", "arm5"):
        values = _series(rows, ("net", name))
        if len(values) == len(x) and len(values) > 0:
            axis.plot(x, values, label=name)
    axis.set_xlabel("κ")
    axis.set_ylabel("Net ₹/1,000")
    if axis.lines:
        axis.legend()
    return figure


def plot_kappa_difference(data: Any, params: Params = P) -> Figure:
    """Plot Arm 5 minus Arm 4 with its confidence band."""
    figure, axis = _figure("Arm 5 − Arm 4 vs κ", params)
    rows = _rows(data, "kappa_table")
    x = _series(rows, ("kappa",))
    mean = _series(rows, ("arm5_minus_arm4_mean",))
    low = _series(rows, ("ci_low",))
    high = _series(rows, ("ci_high",))
    if len(x) > 0:
        axis.plot(x, mean, label="Arm 5 − Arm 4")
        axis.fill_between(x, low, high)
    axis.axhline(0.0)
    axis.set_xlabel("κ")
    axis.set_ylabel("₹/1,000")
    return figure


def plot_oat_tornado(data: Any, params: Params = P) -> Figure:
    """Plot low/high OAT Arm-5 minus Arm-4 values ordered by rank."""
    figure, axis = _figure("OAT sensitivity", params)
    rows = _rows(data, "oat_rows")
    names: list[str] = []
    low: list[float] = []
    high: list[float] = []
    for row in rows:
        if str(row.get("level")) == "lo":
            names.append(str(row.get("parameter", "")))
            low.append(float(row["arm5_minus_arm4"]["mean"]))
        if str(row.get("level")) == "hi":
            high.append(float(row["arm5_minus_arm4"]["mean"]))
    if names and len(low) == len(high):
        positions = np.arange(len(names), dtype="float64")
        axis.barh(positions, np.asarray(high) - np.asarray(low))
        axis.set_yticks(positions, names)
    axis.set_xlabel("Δ ₹/1,000")
    return figure


def plot_lhs_scatter(data: Any, params: Params = P) -> Figure:
    """Plot LHS difference against uplift and hourly rate."""
    figure, axes = plt.subplots(ncols=2)
    figure.text(1.0 / 2.0, 1.0 / 100.0, str(params["report.simulator_footer"]), ha="center")
    rows = _rows(data, "lhs_scatter")
    difference = _series(rows, ("mean_difference",))
    axes[0].scatter(_series(rows, ("uplift_multiplier",)), difference)
    axes[1].scatter(_series(rows, ("hourly_rate",)), difference)
    axes[0].set_xlabel("Uplift multiplier")
    axes[1].set_xlabel("Hourly rate")
    for axis in axes:
        axis.set_ylabel("Arm 5 − Arm 4 ₹/1,000")
    return figure


def plot_recommendation_boundaries(data: Any, params: Params = P) -> Figure:
    """Plot the frozen OTP and packing recommendation boundaries."""
    figure, axis = _figure("Recommendation boundaries", params)
    rows = _rows(data, "boundary_rows")
    x = _series(rows, ("x",))
    otp = _series(rows, ("boundaries", "otp_electronics"))
    packing = _series(rows, ("boundaries", "packing_apparel"))
    if len(x) > 0:
        axis.plot(x, otp, label="OTP · Electronics")
        axis.plot(x, packing, label="Packing · Apparel")
        axis.legend()
    axis.set_ylabel("Break-even ₹")
    return figure


def plot_defense_only(data: Any, params: Params = P) -> Figure:
    """Plot headline and carrier-fault defense-only win rates."""
    figure, axis = _figure("Defense-only win rate", params)
    rows = _rows(data, "defense_rows")
    names = [str(row.get("arm", "")) for row in rows]
    headline = [float(row.get("headline", 0.0)) for row in rows]
    carrier = [float(row.get("carrier_fault", 0.0)) for row in rows]
    if names:
        positions = np.arange(len(names), dtype="float64")
        axis.bar(positions, headline, label="Headline")
        axis.plot(positions, carrier, label="Carrier fault")
        axis.set_xticks(positions, names)
        axis.legend()
    axis.set_ylabel("Win rate")
    return figure


def plot_prevention_defense(data: Any, params: Params = P) -> Figure:
    """Plot prevention and defence shares for Arms 4 and 5."""
    figure, axis = _figure("Prevention vs defence", params)
    rows = _rows(data, "split_rows")
    names = [str(row.get("arm", "")) for row in rows]
    prevention = [float(row.get("prevention", 0.0)) for row in rows]
    defence = [float(row.get("defence", 0.0)) for row in rows]
    if names:
        positions = np.arange(len(names), dtype="float64")
        axis.bar(positions, prevention, label="Prevention")
        axis.bar(positions, defence, bottom=prevention, label="Defence")
        axis.set_xticks(positions, names)
        axis.legend()
    axis.set_ylabel("Share")
    return figure


def plot_coverage_friction(data: Any, params: Params = P) -> Figure:
    """Plot physical coverage and friction for Arms 4 and 5."""
    figure, axis = _figure("Coverage and friction", params)
    rows = _rows(data, "coverage_rows")
    names = [str(row.get("arm", "")) for row in rows]
    coverage = [float(row.get("coverage_any", 0.0)) for row in rows]
    friction = [float(row.get("otp_requested_pct", 0.0)) for row in rows]
    if names:
        positions = np.arange(len(names), dtype="float64")
        axis.bar(positions, coverage, label="Any evidence coverage")
        axis.plot(positions, friction, label="OTP requested")
        axis.set_xticks(positions, names)
        axis.legend()
    axis.set_ylabel("Fraction of test orders")
    return figure


def plot_lorenz(data: Any, params: Params = P) -> Figure:
    """Plot Stage-A Lorenz curves for the configured κ anchors."""
    figure, axis = _figure("Stage-A Lorenz curves", params)
    rows = _rows(data, "lorenz_rows")
    for row in rows:
        deciles = row.get("deciles", [])
        if isinstance(deciles, list):
            x = _series(deciles, ("decile",))
            y = _series(deciles, ("share_of_disputes",))
            if len(x) > 0:
                axis.plot(x, y, label=f"κ={row.get('kappa', '')}")
    if axis.lines:
        axis.legend()
    axis.set_xlabel("Decile")
    axis.set_ylabel("Share of disputes")
    return figure


def chart_functions() -> tuple[ChartFactory, ...]:
    """Return all nine PNG-producing chart factories in report order."""
    return (
        plot_kappa_net,
        plot_kappa_difference,
        plot_oat_tornado,
        plot_lhs_scatter,
        plot_recommendation_boundaries,
        plot_defense_only,
        plot_prevention_defense,
        plot_coverage_friction,
        plot_lorenz,
    )


def save_chart(figure: Figure, path: pathlib.Path, params: Params = P) -> pathlib.Path:
    """Verify the footer and save one PNG."""
    footer = str(params["report.simulator_footer"])
    if not any(item.get_text() == footer for item in figure.texts):
        raise InvariantError("refusing to save a chart without the simulator footer")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    return path


def generate_all_charts(data: Mapping[str, Any], output_dir: pathlib.Path, params: Params = P) -> tuple[pathlib.Path, ...]:
    """Generate and save the nine required Phase-4 charts."""
    names = (
        "01_kappa_net.png",
        "02_kappa_arm5_minus_arm4.png",
        "03_oat_tornado.png",
        "04_lhs_scatter.png",
        "05_recommendation_boundaries.png",
        "06_defense_only.png",
        "07_prevention_vs_defense.png",
        "08_coverage_friction.png",
        "09_lorenz.png",
    )
    figures = [factory(data, params) for factory in chart_functions()]
    paths: list[pathlib.Path] = []
    for name, figure in zip(names, figures, strict=True):
        paths.append(save_chart(figure, pathlib.Path(output_dir) / name, params))
        plt.close(figure)
    return tuple(paths)
