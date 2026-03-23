import argparse
import math
import os
import pathlib
import sys
from collections import defaultdict

import pyarrow.parquet as pq

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nanochat.dataset import list_parquet_files


def bytes_to_human(nbytes: int) -> str:
    """Human-readable bytes (base-2 units)."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(nbytes)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.2f} {u}"
        v /= 1024.0
    return f"{nbytes} B"

def ascii_histogram(
    hist: dict[int, int],
    bin_width: int,
    max_bins: int = 60,
    log_scale: bool = False,
) -> None:
    """
    Render a compact in-terminal histogram.

    We bucket by bucket start positions (e.g. 0, 32, 64, ...).
    """
    starts = sorted(hist.keys())
    if not starts:
        print("(empty histogram)")
        return

    # Keep it readable even if there are many buckets.
    if len(starts) > max_bins:
        starts = starts[:max_bins]

    max_count = max(hist[s] for s in starts)
    bar_max_width = 50

    scale_label = "log-x" if log_scale else "linear"
    print(f"Text length histogram (ASCII, bin width={bin_width} chars, x-scale={scale_label}):")
    for s in starts:
        c = hist[s]
        bar_len = int((c / max_count) * bar_max_width) if max_count else 0
        bar = "#" * bar_len
        end = s + bin_width - 1
        if log_scale:
            mid = (s + end) / 2.0
            log_mid = math.log10(mid + 1.0)
            print(f"{s}-{end} (log10(mid+1)={log_mid:.3f}) | {bar} {c}")
        else:
            print(f"{s}-{end} | {bar} {c}")


def show_histogram_plot(hist: dict[int, int], bin_width: int, use_mpl: bool, log_scale: bool) -> None:
    """
    Show a histogram diagram.

    Default: in-terminal ASCII histogram (reliable in headless/CI).
    Optional: matplotlib GUI window (best-effort; may not be visible here).
    """
    if not use_mpl:
        ascii_histogram(hist, bin_width, log_scale=log_scale)
        return

    # Best-effort GUI render.
    # If it fails, users still get an ASCII histogram.
    try:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        starts = sorted(hist.keys())
        counts = [hist[s] for s in starts]

        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111)
        # Matplotlib log scale can't include x=0, so plot x positions as (bucket_start + 1).
        x_plot = [s + 1 for s in starts]
        ax.bar(x_plot, counts, width=max(1.0, bin_width * 0.95), align="edge")
        ax.set_xlabel(f"Text length (chars, bucket_start+1), bin_width={bin_width}")
        ax.set_ylabel("Count")
        if log_scale:
            ax.set_xscale("log")
            ax.set_title("Text length histogram (log x)")
        else:
            ax.set_title("Text length histogram")

        # Avoid trying to label every x tick when there are many bins.
        n = len(starts)
        if n > 0:
            tick_step = max(1, n // 10)
            xticks = [starts[i] + 1 for i in range(0, n, tick_step)]
            ax.set_xticks(xticks)

        fig.tight_layout()
        plt.show()
    except Exception as e:
        print(f"(matplotlib display failed: {type(e).__name__}) Falling back to ASCII.")
        ascii_histogram(hist, bin_width, log_scale=log_scale)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics for Parquet files in DATA_DIR.",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=32,
        help="Histogram bin width for len(text) in characters (default: 32).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Limit to the first N parquet files for faster dev checks (default: -1 = all).",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default="",
        help="(Deprecated) kept for compatibility. If set, histogram will be shown (not saved).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a histogram diagram in the terminal (ASCII).",
    )
    parser.add_argument(
        "--mpl",
        action="store_true",
        help="Try to open a matplotlib window when used together with --plot.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Render histogram with logarithmic text-length (x) scale.",
    )
    parser.add_argument(
        "--hide-buckets",
        action="store_true",
        help="If set, suppress printing per-bin histogram counts (useful with --plot-out).",
    )
    args = parser.parse_args()

    if args.bin_width <= 0:
        raise SystemExit("--bin-width must be > 0")

    parquet_paths = list_parquet_files()
    if args.max_files != -1:
        parquet_paths = parquet_paths[: args.max_files]

    num_parquet_files = len(parquet_paths)
    total_size_bytes = 0
    total_rows = 0

    # Histogram: key is the bin start (e.g. 0, 32, 64, ...), value is count.
    hist = defaultdict(int)
    hist_samples = 0

    for filepath in parquet_paths:
        total_size_bytes += os.path.getsize(filepath)

        pf = pq.ParquetFile(filepath)
        total_rows += int(pf.metadata.num_rows)

        # Iterate row groups so we don't have to load the whole file in memory.
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            # Keep this column name consistent with nanochat/dataset.py.
            texts = rg.column("text").to_pylist()
            for t in texts:
                # For this dataset, `t` is expected to be a string, but be defensive.
                if t is None:
                    l = 0
                else:
                    try:
                        l = len(t)
                    except TypeError:
                        l = len(str(t))
                bucket_start = (l // args.bin_width) * args.bin_width
                hist[bucket_start] += 1
                hist_samples += 1

    # Summary
    total_size_gb = total_size_bytes / (1024**3)
    print(f"Parquet files: {num_parquet_files}")
    print(f"Total size (bytes): {total_size_bytes}")
    print(f"Total size (GB): {total_size_gb:.6f} GB")
    print(f"Total size (human): {bytes_to_human(total_size_bytes)}")
    print(f"Total rows: {total_rows}")

    if hist_samples != total_rows:
        print(f"Warning: histogram samples ({hist_samples}) != total rows ({total_rows})")

    # Optional diagram
    if args.plot_out:
        print("Note: --plot-out is deprecated; histogram will be shown instead of saved.")
        args.plot = True

    if args.plot:
        show_histogram_plot(hist, args.bin_width, use_mpl=args.mpl, log_scale=args.log_scale)

    # Histogram (text)
    if not args.hide_buckets and not args.plot:
        print(f"Text length histogram (bin width={args.bin_width} chars):")
        for start in sorted(hist.keys()):
            end = start + args.bin_width - 1
            count = hist[start]
            print(f"{start}-{end}: {count}")


if __name__ == "__main__":
    main()
