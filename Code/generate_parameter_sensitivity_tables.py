"""Generate measured LaTeX OFAT tables from the frozen result CSVs."""

from __future__ import annotations

import csv
import math
from pathlib import Path


LEGACY_PATH = Path("parameter_analysis_results.csv")
GAUSSIAN_PATH = Path(
    "results/final_parameter_analysis/"
    "gaussian_parameter_sensitivity_results.csv"
)
MAIN_OUTPUT = Path(
    "results/final_parameter_analysis/"
    "parameter_sensitivity_main_tables.tex"
)
APPENDIX_OUTPUT = Path(
    "results/final_parameter_analysis/"
    "parameter_sensitivity_appendix_tables.tex"
)

METHODS = {
    1: "Direct squared loss",
    2: "Iterative squared loss",
    3: "Correntropy loss",
}
MAIN_NOISES = (
    ("clean", "Clean"),
    ("gaussian_20", "Gaussian 20 dB"),
    ("saltpepper_5", "SP 5\\%"),
    ("saltpepper_10", "SP 10\\%"),
)
APPENDIX_NOISES = (
    ("gaussian_30", "Gaussian 30 dB"),
    ("gaussian_10", "Gaussian 10 dB"),
)
N_VALUES = (4, 8, 16, 32)
H_VALUES = (1, 2, 3, 4)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return [dict(row) for row in csv.DictReader(source)]


def _finite(row: dict, field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field}: {row}")
    return value


def _find_one(rows: list[dict], predicate, description: str) -> dict:
    matches = [row for row in rows if predicate(row)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for {description}, found {len(matches)}."
        )
    return matches[0]


def _measured_row(
    legacy: list[dict],
    gaussian: list[dict],
    *,
    sweep: str,
    noise: str,
    parameter: int,
    algorithm: int,
) -> dict[str, float | str | int]:
    if sweep == "n":
        experiment = "n_analysis"
        n, h = parameter, 1
    elif sweep == "h":
        experiment = "h_analysis"
        n, h = 8, parameter
    else:
        raise ValueError(f"Unknown sweep: {sweep}")

    if noise == "clean":
        scenario = "no_noise"
    else:
        scenario = noise

    if scenario in {"no_noise", "saltpepper_5", "saltpepper_10"}:
        source = _find_one(
            legacy,
            lambda row: (
                row["experiment"] == experiment
                and row["noise_scenario"] == scenario
                and int(row["n"]) == n
                and int(row["h"]) == h
                and int(row["algorithm"]) == algorithm
            ),
            f"{sweep}/{scenario}/{parameter}/{algorithm}",
        )
        rmse = _finite(source, "gradient_RMSE")
        hog_error = _finite(source, "hog_error")
        cosine = _finite(source, "cosine_similarity")
        runtime = _finite(source, "runtime")
    else:
        snr = float(noise.split("_", 1)[1])
        source = _find_one(
            gaussian,
            lambda row: (
                float(row["target_snr_db"]) == snr
                and int(row["n"]) == n
                and int(row["h"]) == h
                and int(row["algorithm"]) == algorithm
            ),
            f"{sweep}/Gaussian {snr}/{parameter}/{algorithm}",
        )
        rmse = _finite(source, "matched_clean_gradient_RMSE")
        hog_error = _finite(source, "hog_relative_error")
        cosine = _finite(source, "cosine_similarity")
        runtime = _finite(source, "runtime")

    if scenario == "no_noise" and not (
        abs(rmse) <= 1e-12
        and abs(hog_error) <= 1e-12
        and abs(cosine - 1.0) <= 1e-12
    ):
        raise ValueError("Clean matched-reference metrics are not exact.")

    return {
        "method": METHODS[algorithm],
        "n": n,
        "h": h,
        "rmse": rmse,
        "hog_error": hog_error,
        "cosine": cosine,
        "runtime": runtime,
    }


def _table(
    legacy: list[dict],
    gaussian: list[dict],
    *,
    sweep: str,
    noises: tuple[tuple[str, str], ...],
    caption: str,
    label: str,
    appendix: bool,
) -> str:
    parameter_name = "$n$" if sweep == "n" else "$h$"
    values = N_VALUES if sweep == "n" else H_VALUES
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        rf"\begin{{longtable}}{{@{{}}lr lrrrr@{{}}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        rf"Noise & {parameter_name} & Method & "
        r"\shortstack{Matched-clean\\Gradient RMSE} & "
        r"\shortstack{HOG Relative\\Error} & "
        r"\shortstack{Cosine\\Similarity} & Runtime (s) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{7}{l}{\tablename\ \thetable\ (continued)}\\",
        r"\toprule",
        rf"Noise & {parameter_name} & Method & "
        r"\shortstack{Matched-clean\\Gradient RMSE} & "
        r"\shortstack{HOG Relative\\Error} & "
        r"\shortstack{Cosine\\Similarity} & Runtime (s) \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{7}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for noise_index, (noise_key, noise_label) in enumerate(noises):
        for value_index, parameter in enumerate(values):
            for algorithm in METHODS:
                row = _measured_row(
                    legacy,
                    gaussian,
                    sweep=sweep,
                    noise=noise_key,
                    parameter=parameter,
                    algorithm=algorithm,
                )
                lines.append(
                    f"{noise_label} & {parameter} & {row['method']} & "
                    f"{row['rmse']:.3f} & {row['hog_error']:.4f} & "
                    f"{row['cosine']:.4f} & {row['runtime']:.3f} \\\\"
                )
            if value_index != len(values) - 1:
                lines.append(r"\addlinespace[1.2pt]")
        if noise_index != len(noises) - 1:
            lines.extend([r"\addlinespace[2.4pt]", r"\midrule"])
    lines.extend([r"\end{longtable}", r"\endgroup"])
    if appendix:
        lines.extend(
            [
                r"\noindent\footnotesize All rows use $p=3$, seed 42, "
                r"the same $224\times224$ ROI, and identical estimator and "
                r"HOG settings. Runtime is the median of three "
                r"gradient-plus-HOG runs. Correntropy at Gaussian 10~dB "
                r"with $(n,h)=(8,1)$ reached the fixed 500-iteration limit; "
                r"that row reports the measured final iterate.",
                r"\normalsize",
            ]
        )
    else:
        lines.extend(
            [
                r"\noindent\footnotesize All rows use $p=3$, seed 42, "
                r"the same $224\times224$ ROI, and identical estimator and "
                r"HOG settings. Runtime is the median of three "
                r"gradient-plus-HOG runs; solver setup and matched-clean "
                r"reference computation are excluded. Clean rows compare "
                r"each method with its identical clean reference.",
                r"\normalsize",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    legacy = _read(LEGACY_PATH)
    gaussian = _read(GAUSSIAN_PATH)
    if len(gaussian) != 63:
        raise ValueError(f"Expected 63 Gaussian rows, found {len(gaussian)}.")

    main_tables = [
        "% Generated from measured CSV rows; do not edit numerical values.",
        _table(
            legacy,
            gaussian,
            sweep="n",
            noises=MAIN_NOISES,
            caption=(
                r"Neighborhood-count sensitivity at fixed $h=1$ for the "
                r"main noise conditions. Rows are ordered by noise, $n$, "
                r"and method."
            ),
            label="tab:n-sensitivity-main",
            appendix=False,
        ),
        _table(
            legacy,
            gaussian,
            sweep="h",
            noises=MAIN_NOISES,
            caption=(
                r"Sample-spacing sensitivity at fixed $n=8$ for the main "
                r"noise conditions. Rows are ordered by noise, $h$, and "
                r"method."
            ),
            label="tab:h-sensitivity-main",
            appendix=False,
        ),
    ]
    appendix_tables = [
        "% Generated from measured CSV rows; do not edit numerical values.",
        r"\clearpage",
        r"\appendix",
        r"\section*{Additional Gaussian Sensitivity Results}",
        (
            r"Tables~\ref{tab:n-sensitivity-gaussian-appendix} and "
            r"\ref{tab:h-sensitivity-gaussian-appendix} report the full "
            r"Gaussian 30 and 10~dB OFAT rows omitted from the main tables."
        ),
        _table(
            legacy,
            gaussian,
            sweep="n",
            noises=APPENDIX_NOISES,
            caption=(
                r"Neighborhood-count sensitivity at fixed $h=1$ for "
                r"Gaussian 30 and 10~dB."
            ),
            label="tab:n-sensitivity-gaussian-appendix",
            appendix=True,
        ),
        r"\clearpage",
        _table(
            legacy,
            gaussian,
            sweep="h",
            noises=APPENDIX_NOISES,
            caption=(
                r"Sample-spacing sensitivity at fixed $n=8$ for Gaussian "
                r"30 and 10~dB."
            ),
            label="tab:h-sensitivity-gaussian-appendix",
            appendix=True,
        ),
    ]
    MAIN_OUTPUT.write_text(
        "\n\n".join(main_tables) + "\n", encoding="utf-8"
    )
    APPENDIX_OUTPUT.write_text(
        "\n\n".join(appendix_tables) + "\n", encoding="utf-8"
    )
    print(f"Wrote {MAIN_OUTPUT}")
    print(f"Wrote {APPENDIX_OUTPUT}")


if __name__ == "__main__":
    main()
