#!/usr/bin/env python3
"""Pipeline reproducible para la estación Águilas (AGU), marzo de 2024.

Aplica la NOM-172-SEMARNAT-2023 a PM10, PM2.5 y O3, genera resultados
horarios y diarios, diagnósticos de calidad de datos y visualizaciones.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


CATEGORY_ORDER = [
    "Buena",
    "Aceptable",
    "Mala",
    "Muy mala",
    "Extremadamente mala",
]
CATEGORY_LEVEL = {name: i for i, name in enumerate(CATEGORY_ORDER)}
CATEGORY_COLOR = {
    "Buena": "#00E400",
    "Aceptable": "#FFFF00",
    "Mala": "#FF7E00",
    "Muy mala": "#FF0000",
    "Extremadamente mala": "#8F3F97",
}
ADJUSTMENT_FACTOR = {"PM10": 0.714, "PM2.5": 0.694}


def round_half_up(value: float, decimals: int = 0) -> float:
    """Redondeo decimal convencional solicitado por la NOM."""
    if pd.isna(value):
        return np.nan
    quantum = Decimal("1") if decimals == 0 else Decimal("1." + "0" * decimals)
    return float(Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def category(value: float, cuts: list[float]) -> str | None:
    if pd.isna(value):
        return None
    return CATEGORY_ORDER[sum(value > cut for cut in cuts)]


def category_pm10(value: float) -> str | None:
    # Columna aplicable a partir de enero de 2024.
    return category(value, [45, 60, 132, 213])


def category_pm25(value: float) -> str | None:
    # Columna aplicable a partir de enero de 2024.
    return category(value, [15, 33, 79, 130])


def category_o3(value: float) -> str | None:
    if pd.isna(value):
        return None
    return category(round_half_up(value, 3), [0.058, 0.090, 0.135, 0.175])


def nowcast(series: pd.Series, adjustment: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Promedio móvil ponderado de 12 h con factor de ajuste NOM-172-2023.

    La hora más reciente usa exponente 0. Se mantienen los exponentes de las
    posiciones faltantes y se exige información en al menos dos de las tres
    horas más recientes.
    """
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    weights_used = np.full(len(values), np.nan)
    observations = np.zeros(len(values), dtype=int)

    for idx in range(len(values)):
        window = values[max(0, idx - 11) : idx + 1][::-1]
        if np.sum(~np.isnan(window[:3])) < 2:
            continue
        valid = window[~np.isnan(window)]
        if valid.size == 0:
            continue
        maximum = float(np.max(valid))
        minimum = float(np.min(valid))
        raw_weight = 1.0 if maximum == 0 else 1 - (maximum - minimum) / maximum
        weight = round_half_up(max(0.5, raw_weight), 2)
        exponents = np.arange(len(window))
        mask = ~np.isnan(window)
        weighted_mean = np.sum(window[mask] * weight ** exponents[mask]) / np.sum(
            weight ** exponents[mask]
        )
        result[idx] = round_half_up(weighted_mean * adjustment, 0)
        weights_used[idx] = weight
        observations[idx] = int(mask.sum())
    return result, weights_used, observations


def worst_category(row: pd.Series, fields: dict[str, str]) -> tuple[str | None, str | None]:
    pairs = [(pollutant, row[field]) for pollutant, field in fields.items()]
    pairs = [(p, c) for p, c in pairs if c in CATEGORY_LEVEL]
    if not pairs:
        return None, None
    maximum = max(CATEGORY_LEVEL[c] for _, c in pairs)
    winners = [(p, c) for p, c in pairs if CATEGORY_LEVEL[c] == maximum]
    return winners[0][1], "/".join(p for p, _ in winners)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "axes.edgecolor": "#4A5568",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D9E1E8",
            "grid.alpha": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_completeness(quality: pd.DataFrame, output: Path) -> None:
    selected = quality[quality["variable"].isin(["PM10", "O3", "PM2.5", "ET", "RH", "WS", "WD"])]
    selected = selected.set_index("variable").loc[["PM10", "O3", "PM2.5", "ET", "RH", "WS", "WD"]].reset_index()
    colors = ["#2E9D62" if x == 100 else "#4C78A8" if x > 0 else "#B8C1CC" for x in selected["completitud_pct"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.barh(selected["variable"], selected["completitud_pct"], color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Completitud (%)")
    ax.set_title("Disponibilidad de variables en marzo de 2024")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for bar, value, available in zip(bars, selected["completitud_pct"], selected["disponibles"]):
        ax.text(value + 1.0, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%  ({available}/744)", va="center", fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, output)


def plot_month_overview(hourly: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.8), sharex=True, gridspec_kw={"hspace": 0.18})
    x = hourly["timestamp"]
    axes[0].plot(x, hourly["PM10"], color="#AAB5C1", lw=0.8, label="PM10 horaria")
    axes[0].plot(x, hourly["PM10_NowCast"], color="#D06B21", lw=1.35, label="PM10 NowCast")
    axes[0].axhline(45, color="#2E9D62", ls="--", lw=0.8)
    axes[0].axhline(60, color="#F28E2B", ls="--", lw=0.8)
    axes[0].set_ylabel("µg/m³")
    axes[0].set_title("Evolución horaria de contaminantes e indicador meteorológico")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)

    axes[1].plot(x, hourly["PM2.5"], color="#BFC7CF", lw=0.75, label="PM2.5 horaria")
    axes[1].plot(x, hourly["PM2.5_NowCast"], color="#6B5BA7", lw=1.25, label="PM2.5 NowCast")
    axes[1].axhline(15, color="#2E9D62", ls="--", lw=0.8)
    axes[1].axhline(33, color="#F28E2B", ls="--", lw=0.8)
    axes[1].set_ylabel("µg/m³")
    axes[1].legend(loc="upper right", ncol=2, frameon=False)

    axes[2].plot(x, hourly["O3"], color="#3976A8", lw=1.1)
    axes[2].axhline(0.058, color="#2E9D62", ls="--", lw=0.8, label="Límite Buena")
    axes[2].axhline(0.090, color="#F28E2B", ls="--", lw=0.8, label="Límite Aceptable")
    axes[2].set_ylabel("O₃ (ppm)")
    axes[2].legend(loc="upper right", ncol=2, frameon=False)

    axes[3].plot(x, hourly["ET"], color="#7B8794", lw=1.0)
    axes[3].set_ylabel("Temperatura (°C)")
    axes[3].set_xlabel("Fecha")
    axes[3].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%d mar"))
    for ax in axes:
        ax.grid(axis="y")
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    save_figure(fig, output)


def plot_hourly_heatmap(hourly: pd.DataFrame, output: Path) -> None:
    matrix = hourly.assign(
        day=hourly["timestamp"].dt.day,
        hour=hourly["timestamp"].dt.hour,
        level=hourly["categoria_global_horaria"].map(CATEGORY_LEVEL),
    ).pivot(index="day", columns="hour", values="level")
    fig, ax = plt.subplots(figsize=(10.7, 7.4))
    cmap = ListedColormap([CATEGORY_COLOR[c] for c in CATEGORY_ORDER])
    image = ax.imshow(matrix.values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=4.5)
    ax.set_title("Categoría global por día y hora")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Día de marzo")
    ax.set_xticks(np.arange(0, 24, 2), labels=[str(x) for x in range(0, 24, 2)])
    ax.set_yticks(np.arange(0, 31, 2), labels=[str(x) for x in range(1, 32, 2)])
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOR[c]) for c in CATEGORY_ORDER[:3]]
    ax.legend(handles, CATEGORY_ORDER[:3], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    fig.colorbar(image, ax=ax, ticks=[], fraction=0.018, pad=0.02)
    fig.tight_layout()
    save_figure(fig, output)


def plot_representative_period(hourly: pd.DataFrame, output: Path) -> None:
    period = hourly[(hourly["timestamp"] >= "2024-03-27") & (hourly["timestamp"] < "2024-03-29")].copy()
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8.7), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.65, 1.25, 0.32], "hspace": 0.16})
    x = period["timestamp"]
    axes[0].plot(x, period["PM10"], color="#B8C1CC", lw=1.0, label="Dato horario PM10")
    axes[0].plot(x, period["PM10_NowCast"], color="#D06B21", marker="o", ms=2.5, lw=1.4, label="Indicador NowCast")
    axes[0].axhspan(45, 60, color=CATEGORY_COLOR["Aceptable"], alpha=0.12)
    axes[0].axhspan(60, max(100, period["PM10"].max() + 5), color=CATEGORY_COLOR["Mala"], alpha=0.08)
    axes[0].axhline(45, color="#8A7A21", ls="--", lw=0.8)
    axes[0].axhline(60, color="#B86013", ls="--", lw=0.8)
    axes[0].set_ylabel("PM10 (µg/m³)")
    axes[0].set_title("Simulación de llegada progresiva de datos 27 y 28 de marzo")
    axes[0].legend(loc="upper left", ncol=2, frameon=False)

    axes[1].plot(x, period["O3"], color="#3976A8", marker="o", ms=2.5, lw=1.25)
    axes[1].axhspan(0.058, 0.090, color=CATEGORY_COLOR["Aceptable"], alpha=0.12)
    axes[1].axhspan(0.090, 0.11, color=CATEGORY_COLOR["Mala"], alpha=0.09)
    axes[1].axhline(0.058, color="#8A7A21", ls="--", lw=0.8)
    axes[1].axhline(0.090, color="#B86013", ls="--", lw=0.8)
    axes[1].set_ylabel("O₃ (ppm)")

    axes[2].plot(x, period["ET"], color="#6F7D89", lw=1.2)
    axes[2].set_ylabel("Temperatura (°C)")

    levels = period["categoria_global_horaria"].map(CATEGORY_LEVEL).to_numpy()[None, :]
    axes[3].imshow(levels, aspect="auto", cmap=ListedColormap([CATEGORY_COLOR[c] for c in CATEGORY_ORDER]), vmin=-0.5, vmax=4.5, extent=[mdates.date2num(x.iloc[0]), mdates.date2num(x.iloc[-1] + pd.Timedelta(hours=1)), 0, 1])
    axes[3].set_yticks([])
    axes[3].set_ylabel("Global", rotation=0, ha="right", va="center")
    axes[3].xaxis_date()
    axes[3].xaxis.set_major_locator(mdates.HourLocator(interval=4))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    axes[3].set_xlabel("Fecha y hora")
    for ax in axes[:3]:
        ax.grid(axis="y")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    save_figure(fig, output)


def plot_daily_calendar(daily: pd.DataFrame, output: Path) -> None:
    first_weekday = pd.Timestamp("2024-03-01").weekday()  # lunes=0
    weekdays = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6)
    ax.axis("off")
    for col, label in enumerate(weekdays):
        ax.text(col + 0.5, 5.72, label, ha="center", va="center", fontweight="bold", color="#263746")
    for _, row in daily.iterrows():
        day = int(row["fecha"].day)
        position = first_weekday + day - 1
        week, col = divmod(position, 7)
        y = 4.85 - week
        category_name = row["categoria_global_diaria"]
        color = CATEGORY_COLOR[category_name]
        rect = plt.Rectangle((col + 0.05, y - 0.72), 0.9, 0.82, facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        text_color = "white" if category_name in {"Mala", "Muy mala", "Extremadamente mala"} else "#1A202C"
        ax.text(col + 0.13, y - 0.02, str(day), ha="left", va="top", fontweight="bold", fontsize=11, color=text_color)
        dominant = str(row["contaminante_dominante"]).replace("/", " · ")
        ax.text(col + 0.5, y - 0.43, dominant, ha="center", va="center", fontsize=7.5, color=text_color)
    ax.set_title("Calendario diario de calidad del aire y contaminante dominante", pad=18, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLOR[c]) for c in CATEGORY_ORDER[:3]]
    ax.legend(handles, CATEGORY_ORDER[:3], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    fig.tight_layout()
    save_figure(fig, output)


def plot_numeralia(daily: pd.DataFrame, output: Path) -> None:
    category_counts = daily["categoria_global_diaria"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    pollutant_counts = {
        pollutant: int(daily["contaminante_dominante"].fillna("").str.split("/").map(lambda items: pollutant in items).sum())
        for pollutant in ["PM10", "PM2.5", "O3"]
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), gridspec_kw={"wspace": 0.34})
    cats_shown = CATEGORY_ORDER[:3]
    values = category_counts.loc[cats_shown]
    bars = axes[0].bar(cats_shown, values, color=[CATEGORY_COLOR[x] for x in cats_shown])
    axes[0].set_title("Días por categoría global")
    axes[0].set_ylabel("Número de días")
    axes[0].set_ylim(0, max(values) + 4)
    axes[0].grid(axis="y")
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.35, str(int(value)), ha="center", fontweight="bold")

    labels = list(pollutant_counts)
    vals = list(pollutant_counts.values())
    bars = axes[1].bar(labels, vals, color=["#D06B21", "#6B5BA7", "#3976A8"])
    axes[1].set_title("Días como responsable o corresponsable")
    axes[1].set_ylabel("Número de días")
    axes[1].set_ylim(0, max(vals) + 4)
    axes[1].grid(axis="y")
    for bar, value in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.35, str(value), ha="center", fontweight="bold")
    fig.suptitle("Numeralia de marzo de 2024", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, output)


def plot_diurnal_profile(hourly: pd.DataFrame, output: Path) -> None:
    profile = hourly.groupby(hourly["timestamp"].dt.hour)[["O3", "PM10", "ET"]].mean()
    fig, axes = plt.subplots(3, 1, figsize=(9.8, 7.2), sharex=True, gridspec_kw={"hspace": 0.18})
    axes[0].plot(profile.index, profile["PM10"], color="#D06B21", marker="o", ms=3)
    axes[0].set_ylabel("PM10 (µg/m³)")
    axes[0].set_title("Perfil horario promedio del mes")
    axes[1].plot(profile.index, profile["O3"], color="#3976A8", marker="o", ms=3)
    axes[1].set_ylabel("O₃ (ppm)")
    axes[2].plot(profile.index, profile["ET"], color="#6F7D89", marker="o", ms=3)
    axes[2].set_ylabel("Temperatura (°C)")
    axes[2].set_xlabel("Hora del día")
    axes[2].set_xticks(range(0, 24, 2))
    for ax in axes:
        ax.grid(axis="y")
    fig.tight_layout()
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="upload/BD_2024(2).xlsx")
    parser.add_argument("--output", default="output/agu_marzo_2024")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    data_dir = output_dir / "datos"
    figure_dir = output_dir / "figuras"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    source = pd.read_excel(input_path, sheet_name="Data")
    source.columns = source.columns.str.strip()
    source["timestamp"] = pd.to_datetime(source["DATE"], errors="coerce")
    source["PM2.5"] = pd.to_numeric(source["PM2.5"], errors="coerce")
    station = source[source["STATION"].eq("AGU")].sort_values("timestamp").copy()

    if station["timestamp"].duplicated().any():
        raise ValueError("La estación AGU contiene marcas de tiempo duplicadas.")
    station["PM10_NowCast"], station["factor_W_PM10"], station["n_12h_PM10"] = nowcast(
        station["PM10"], ADJUSTMENT_FACTOR["PM10"]
    )
    station["PM2.5_NowCast"], station["factor_W_PM2.5"], station["n_12h_PM2.5"] = nowcast(
        station["PM2.5"], ADJUSTMENT_FACTOR["PM2.5"]
    )

    hourly = station[
        (station["timestamp"] >= "2024-03-01") & (station["timestamp"] < "2024-04-01")
    ].copy()
    if len(hourly) != 744:
        raise ValueError(f"Se esperaban 744 horas en marzo; se encontraron {len(hourly)}.")
    if not (hourly["timestamp"].dt.hour == hourly["HOUR"]).all():
        raise ValueError("La columna HOUR no coincide con la hora contenida en DATE.")

    hourly["categoria_PM10"] = hourly["PM10_NowCast"].map(category_pm10)
    hourly["categoria_PM2.5"] = hourly["PM2.5_NowCast"].map(category_pm25)
    hourly["categoria_O3"] = hourly["O3"].map(category_o3)
    hourly[["categoria_global_horaria", "contaminante_dominante_horario"]] = hourly.apply(
        lambda row: pd.Series(
            worst_category(
                row,
                {"PM10": "categoria_PM10", "PM2.5": "categoria_PM2.5", "O3": "categoria_O3"},
            )
        ),
        axis=1,
    )

    daily_rows: list[dict[str, object]] = []
    for date, group in hourly.groupby(hourly["timestamp"].dt.date):
        n_pm10 = int(group["PM10"].notna().sum())
        n_pm25 = int(group["PM2.5"].notna().sum())
        n_o3 = int(group["O3"].notna().sum())
        pm10_daily = round_half_up(group["PM10"].mean(), 0) if n_pm10 >= 18 else np.nan
        pm25_daily = round_half_up(group["PM2.5"].mean(), 0) if n_pm25 >= 18 else np.nan
        o3_daily = round_half_up(group["O3"].max(), 3) if n_o3 > 0 else np.nan
        row = {
            "fecha": pd.Timestamp(date),
            "horas_PM10": n_pm10,
            "horas_PM2.5": n_pm25,
            "horas_O3": n_o3,
            "PM10_promedio_24h": pm10_daily,
            "PM2.5_promedio_24h": pm25_daily,
            "O3_maximo_1h": o3_daily,
            "categoria_PM10": category_pm10(pm10_daily),
            "categoria_PM2.5": category_pm25(pm25_daily),
            "categoria_O3": category_o3(o3_daily),
            "temperatura_media_C": round(group["ET"].mean(), 2),
            "temperatura_min_C": round(group["ET"].min(), 2),
            "temperatura_max_C": round(group["ET"].max(), 2),
        }
        row["categoria_global_diaria"], row["contaminante_dominante"] = worst_category(
            pd.Series(row),
            {"PM10": "categoria_PM10", "PM2.5": "categoria_PM2.5", "O3": "categoria_O3"},
        )
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)

    units = {"PM10": "µg/m³", "O3": "ppm", "PM2.5": "µg/m³", "ET": "°C", "RH": "%", "WS": "m/s", "WD": "grados"}
    quality_rows = []
    for variable in ["PM10", "O3", "PM2.5", "ET", "RH", "WS", "WD"]:
        available = int(hourly[variable].notna().sum())
        quality_rows.append(
            {
                "variable": variable,
                "unidad": units[variable],
                "disponibles": available,
                "faltantes": 744 - available,
                "completitud_pct": round(100 * available / 744, 1),
                "minimo": hourly[variable].min(),
                "maximo": hourly[variable].max(),
            }
        )
    quality = pd.DataFrame(quality_rows)

    event = hourly[(hourly["timestamp"] >= "2024-03-27") & (hourly["timestamp"] < "2024-03-29")][
        [
            "timestamp",
            "PM10",
            "PM10_NowCast",
            "categoria_PM10",
            "PM2.5",
            "PM2.5_NowCast",
            "categoria_PM2.5",
            "O3",
            "categoria_O3",
            "ET",
            "categoria_global_horaria",
            "contaminante_dominante_horario",
        ]
    ].copy()

    hourly_output_columns = [
        "timestamp",
        "O3",
        "PM10",
        "PM2.5",
        "ET",
        "RH",
        "WS",
        "WD",
        "PM10_NowCast",
        "factor_W_PM10",
        "n_12h_PM10",
        "PM2.5_NowCast",
        "factor_W_PM2.5",
        "n_12h_PM2.5",
        "categoria_PM10",
        "categoria_PM2.5",
        "categoria_O3",
        "categoria_global_horaria",
        "contaminante_dominante_horario",
    ]
    hourly[hourly_output_columns].to_csv(data_dir / "datos_horarios_procesados_AGU_marzo_2024.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(data_dir / "resumen_diario_AGU_marzo_2024.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(data_dir / "calidad_datos_AGU_marzo_2024.csv", index=False, encoding="utf-8-sig")
    event.to_csv(data_dir / "simulacion_27_28_marzo_AGU.csv", index=False, encoding="utf-8-sig")

    hourly_counts = hourly["categoria_global_horaria"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    daily_counts = daily["categoria_global_diaria"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    dominant_counts = daily["contaminante_dominante"].value_counts().to_dict()
    co_responsible_counts = {
        pollutant: int(daily["contaminante_dominante"].fillna("").str.split("/").map(lambda items: pollutant in items).sum())
        for pollutant in ["PM10", "PM2.5", "O3"]
    }
    correlations = hourly[["O3", "PM10", "PM2.5", "ET"]].corr(method="pearson")
    summary = {
        "station": "Águilas (AGU)",
        "period": "2024-03-01 a 2024-03-31",
        "records": int(len(hourly)),
        "hourly_category_counts": {k: int(v) for k, v in hourly_counts.items()},
        "daily_category_counts": {k: int(v) for k, v in daily_counts.items()},
        "daily_dominant_patterns": {str(k): int(v) for k, v in dominant_counts.items()},
        "daily_co_responsible_counts": co_responsible_counts,
        "maxima": {
            "PM10_raw": float(hourly["PM10"].max()),
            "PM10_NowCast": float(hourly["PM10_NowCast"].max()),
            "PM2.5_raw": float(hourly["PM2.5"].max()),
            "PM2.5_NowCast": float(hourly["PM2.5_NowCast"].max()),
            "O3": float(hourly["O3"].max()),
            "ET": float(hourly["ET"].max()),
        },
        "valid_nowcast_hours": {
            "PM10": int(hourly["PM10_NowCast"].notna().sum()),
            "PM2.5": int(hourly["PM2.5_NowCast"].notna().sum()),
        },
        "pearson_correlations": {
            "O3_ET": round(float(correlations.loc["O3", "ET"]), 3),
            "PM10_PM2.5": round(float(correlations.loc["PM10", "PM2.5"]), 3),
        },
    }
    (data_dir / "resumen_resultados.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_completeness(quality, figure_dir / "01_completitud_variables.png")
    plot_month_overview(hourly, figure_dir / "02_evolucion_horaria_mes.png")
    plot_hourly_heatmap(hourly, figure_dir / "03_matriz_categoria_horaria.png")
    plot_representative_period(hourly, figure_dir / "04_simulacion_27_28_marzo.png")
    plot_daily_calendar(daily, figure_dir / "05_calendario_diario.png")
    plot_numeralia(daily, figure_dir / "06_numeralia.png")
    plot_diurnal_profile(hourly, figure_dir / "07_perfil_horario_promedio.png")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
