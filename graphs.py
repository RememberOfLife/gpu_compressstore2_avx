#!/usr/bin/env python3
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

from datetime import datetime
import csv
import sys
import multiprocessing
import os
import time
import math
import functools
import numpy as np

# columns
APPROACH_COL = 0
THREAD_COUNT_COL = 1
DATA_TYPE_COL = 2
ELEMENT_COUNT_COL = 3
MASK_DISTRIBUTION_KIND_COL = 4
SELECTIVITY_COL = 5
RUNTIME_MS_COL = 6
THROUGHPUT_COL = 7
CASE_COL = 8
COLUMN_COUNT = 9
VIRTUAL_COLUMN_COUNT = 2

COLUMNS = list(range(0, COLUMN_COUNT))


COLORS = [
    'darkorange', 'green', 'gold', 'deepskyblue', 'navy',
    'darkred', 'purple', 'red', 'mediumpurple', 'turquoise',
    'lightgray', 'teal', 'lime', 'seagreen', 'plum', 'lightsteelblue',
    'palevioletred', 'black', 'blue', 'magenta', 'olive', 'goldenrod'
]
COLOR_KEYS: dict[str, str] = {}

MARKERS = ['^', '^', '+', '<', 'x', '1', '2', '*', '>', '2', 'D',
           'o', '+', '3', '4', 'D', 'D', 'd', 'd', 'H', 'h', 'p', 'P', 's']
MARKER_KEYS: dict[str, str] = {}

DATA_TYPE_SIZES = {
    "float": 4,
    "double": 8,
    "uint8_t": 4,
    "uint16_t": 4,
    "uint32_t": 4,
    "uint64_t": 8
}


def get_marker_from_str(s):
    global MARKER_KEYS
    if s not in MARKER_KEYS:
        MARKER_KEYS[s] = MARKERS[len(MARKER_KEYS) % len(MARKERS)]
    return MARKER_KEYS[s]


def get_color_from_str(s):
    global COLOR_KEYS
    if s not in COLOR_KEYS:
        COLOR_KEYS[s] = COLORS[len(COLOR_KEYS) % len(COLORS)]
    return COLOR_KEYS[s]


def classify(data, classifying_col):
    classes = {}
    for row in data:
        c = row[classifying_col]
        if c not in classes:
            classes[c] = [row]
        else:
            classes[c].append(row)
    return classes


def classify_mult(data, classifying_cols):
    classes = {}
    for row in data:
        c = []
        for cc in classifying_cols:
            c.append(row[cc])
        c = tuple(c)
        if c not in classes:
            classes[c] = [row]
        else:
            classes[c].append(row)
    return classes


def highest_in_class(classes, maximize_col):
    highest = {}
    for k, v in classes.items():
        highest[k] = max(v, key=lambda r: r[maximize_col])
    return highest


def lowest_in_class(classes, minimize_col):
    lowest = {}
    for k, v in classes.items():
        lowest[k] = min(v, key=lambda r: r[minimize_col])
    return lowest


def col_vals_l(rows, col):
    return map(lambda r: r[col], rows)


def col_vals(rows, col):
    return list(col_vals_l(rows, col))


def unique_col_vals(rows, col):
    results = {}
    for r in rows:
        results[r[col]] = True
    return list(results.keys())


def col_average(rows, col):
    return sum(col_vals_l(rows, col)) / len(rows)


def average_columns(rows, cols):
    class_cols = list(COLUMNS)
    for c in cols:
        class_cols.remove(c)

    output_rows = []
    for row_group in classify_mult(rows, class_cols).values():
        row = row_group[0]
        for c in cols:
            row[c] = col_average(row_group, c)
        output_rows.append(row)
    return output_rows


def class_with_lowest_average(classes, avg_col):
    return min(
        classes.items(),
        key=lambda kv: col_average(kv[1], avg_col)
    )[0]


def class_with_highest_average(classes, avg_col):
    return max(
        classes.items(),
        key=lambda kv: col_average(kv[1], avg_col)
    )[0]


def highest_class_average(classes, avg_col):
    return max(
        [(col_average(rows, avg_col), cl) for (cl, rows) in classes.items()]
    )


def lowest_class_average(classes, avg_col):
    return min(
        [(col_average(rows, avg_col), cl) for (cl, rows) in classes.items()]
    )


def sort_by_col(rows, sort_col):
    return sorted(rows, key=lambda r: r[sort_col])


def max_col_val(rows, col):
    return max(rows, key=lambda r: r[col])[col]


def min_col_val(rows, col):
    return min(rows, key=lambda r: r[col])[col]


def filter_col_val(rows, col, val):
    return list(filter(lambda r: r[col] == val, rows))


def require_col_vals(rows, col, vals):
    vals = dict.fromkeys(vals, None)
    return list(filter(lambda r: r[col] in vals, rows))


def exclude_col_val(rows, col, val):
    return list(filter(lambda r: r[col] != val, rows))


def contains_col_val_mult(rows, coldict):
    for r in rows:
        match = True
        for (col_id, col_value) in coldict.items():
            if r[col_id] != col_value:
                match = False
                break
        if match:
            return True

    return False


def abort_plot(plot_name):
    print(f"aborting plot!: {plot_name}")
    if os.path.exists(plot_name):
        os.unlink(plot_name)
    pass

# graph generators


def throughput_over_thread_count(data, log=False, use_runtime=False, name_appendage=None, nz=None):
    y_axis_name = ("runtime" if use_runtime else "throughput")
    plot_name = (
        y_axis_name
        + "_over_thread_count"
        + (f"_{name_appendage}" if name_appendage else "")
        + ("_log" if log else "")
        + ("_nz" if nz else "")
        + ".png"
    )
    y_val_col = RUNTIME_MS_COL if use_runtime else THROUGHPUT_COL
    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("thread count")
    if use_runtime:
        ax.set_ylabel("runtime (ms)")
    else:
        ax.set_ylabel("throughput (GiB/s)")

    max_elem_count = max_col_val(data, ELEMENT_COUNT_COL)
    ax.set_title(y_axis_name +
                 " over thread count "
                 + (f"for {name_appendage} " if name_appendage else "")
                 + f"(element count = {max_elem_count})")

    elem_count_filtered = filter_col_val(
        data, ELEMENT_COUNT_COL, max_elem_count)
    if (
        len(elem_count_filtered) == 0
        or len(unique_col_vals(data, THREAD_COUNT_COL)) < 2
    ):
        abort_plot(plot_name)
        return

    by_cases = classify(elem_count_filtered, CASE_COL)

    for case, rows in by_cases.items():
        by_thread_count = classify(rows, THREAD_COUNT_COL)
        thread_counts = sorted(unique_col_vals(rows, THREAD_COUNT_COL))
        x = thread_counts
        y = []
        for s in thread_counts:
            s_rows = by_thread_count[s]
            avg = col_average(s_rows, y_val_col)
            y.append(avg)

        ax.plot(
            x, y,
            marker=get_marker_from_str(case),
            color=get_color_from_str(case),
            markerfacecolor='none',
            label=f"{case}", alpha=0.7)
    ax.set_xticks(unique_col_vals(data, THREAD_COUNT_COL))
    if log:
        ax.set_yscale("log")
        ax.set_ylim(min_col_val(elem_count_filtered, y_val_col))
    else:
        if not nz:
            ax.set_ylim(0)

    fig.subplots_adjust(bottom=0.3, wspace=0.33,
                        left=0.05, right=0.95, top=0.95)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    fig.savefig(plot_name)


def single_threaded_througput_over_thread_count(
    data, log=False, use_runtime=False, name_appendage=None, nz=None
):
    plot_name = (
        "single_threaded_througput"
        + "_over_thread_count"
        + (f"_{name_appendage}" if name_appendage else "")
        + ("_log" if log else "")
        + ("_nz" if nz else "")
        + ".png"
    )
    y_val_col = RUNTIME_MS_COL if use_runtime else THROUGHPUT_COL
    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("thread count")
    if use_runtime:
        ax.set_ylabel("runtime (ms)")
    else:
        ax.set_ylabel("throughput (GiB/s)")

    max_elem_count = max_col_val(data, ELEMENT_COUNT_COL)
    ax.set_title("single threaded througput" +
                 " over thread count "
                 + (f"for {name_appendage} " if name_appendage else "")
                 + f"(element count = {max_elem_count})")

    elem_count_filtered = filter_col_val(
        data, ELEMENT_COUNT_COL, max_elem_count)
    if (
        len(elem_count_filtered) == 0
        or len(unique_col_vals(data, THREAD_COUNT_COL)) < 2
    ):
        abort_plot(plot_name)
        return

    by_cases = classify(elem_count_filtered, CASE_COL)

    for case, rows in by_cases.items():
        by_thread_count = classify(rows, THREAD_COUNT_COL)
        thread_counts = sorted(unique_col_vals(rows, THREAD_COUNT_COL))
        x = thread_counts
        y = []
        for tc in thread_counts:
            s_rows = by_thread_count[tc]
            avg = col_average(s_rows, y_val_col) / tc
            y.append(avg)

        ax.plot(
            x, y,
            marker=get_marker_from_str(case),
            color=get_color_from_str(case),
            markerfacecolor='none',
            label=f"{case}", alpha=0.7)
    ax.set_xticks(unique_col_vals(data, THREAD_COUNT_COL))
    if log:
        ax.set_yscale("log")
        ax.set_ylim(min_col_val(elem_count_filtered, y_val_col))
    else:
        if not nz:
            ax.set_ylim(0)

    fig.subplots_adjust(bottom=0.3, wspace=0.33,
                        left=0.05, right=0.95, top=0.95)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    fig.savefig(plot_name)


def mt_speedup_over_thread_count(
    data, name_appendage=None, nz=None
):
    plot_name = (
        "mt_speedup_over_thread_count"
        + "_over_thread_count"
        + (f"_{name_appendage}" if name_appendage else "")
        + ("_nz" if nz else "")
        + ".png"
    )
    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("thread count")

    ax.set_ylabel("mt speedup")

    max_elem_count = max_col_val(data, ELEMENT_COUNT_COL)
    ax.set_title("multi threading speedup" +
                 " over thread count "
                 + (f"for {name_appendage} " if name_appendage else "")
                 + f"(element count = {max_elem_count})")

    elem_count_filtered = filter_col_val(
        data, ELEMENT_COUNT_COL, max_elem_count)
    if (
        len(elem_count_filtered) == 0
        or len(unique_col_vals(data, THREAD_COUNT_COL)) < 2
    ):
        abort_plot(plot_name)
        return

    by_cases = classify(elem_count_filtered, CASE_COL)

    for case, rows in by_cases.items():
        st_throughput_avg = col_average(filter_col_val(
            rows, THREAD_COUNT_COL, 1), THROUGHPUT_COL)
        by_thread_count = classify(rows, THREAD_COUNT_COL)
        thread_counts = sorted(unique_col_vals(rows, THREAD_COUNT_COL))
        x = thread_counts
        y = []
        for tc in thread_counts:
            s_rows = by_thread_count[tc]
            avg = col_average(s_rows, THROUGHPUT_COL) / st_throughput_avg
            y.append(avg)

        ax.plot(
            x, y,
            marker=get_marker_from_str(case),
            color=get_color_from_str(case),
            markerfacecolor='none',
            label=f"{case}", alpha=0.7)
    ax.set_xticks(unique_col_vals(data, THREAD_COUNT_COL))
    if not nz:
        ax.set_ylim(0)

    fig.subplots_adjust(bottom=0.3, wspace=0.33,
                        left=0.05, right=0.95, top=0.95)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    fig.savefig(plot_name)


def throughput_over_selectivity(data, log=False, use_runtime=False, name_appendage=None, nz=None):
    y_axis_name = ("runtime" if use_runtime else "throughput")
    plot_name = (
        y_axis_name
        + "_over_selectivity"
        + (f"_{name_appendage}" if name_appendage else "")
        + ("_log" if log else "")
        + ("_nz" if nz else "")
        + ".png"
    )
    y_val_col = RUNTIME_MS_COL if use_runtime else THROUGHPUT_COL
    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 7))
    ax.set_xlabel("selectivity")
    if use_runtime:
        ax.set_ylabel("runtime (ms)")
    else:
        ax.set_ylabel("throughput (GiB/s)")

    max_elem_count = max_col_val(data, ELEMENT_COUNT_COL)
    ax.set_title(y_axis_name +
                 " over selectivity "
                 + (f"for {name_appendage} " if name_appendage else "")
                 + f"(element count = {max_elem_count}) (thread count = best in class)")

    elem_count_filtered = filter_col_val(
        data, ELEMENT_COUNT_COL, max_elem_count)
    if (
        len(elem_count_filtered) == 0
        or len(unique_col_vals(data, SELECTIVITY_COL)) < 2
    ):
        abort_plot(plot_name)
        return

    by_cases = classify(elem_count_filtered, CASE_COL)

    for case, rows in by_cases.items():
        tc_classes = classify(rows, THREAD_COUNT_COL)
        tc_max = class_with_highest_average(tc_classes, THREAD_COUNT_COL)
        rows = tc_classes[tc_max]
        # tc_max = max_col_val(rows, THREAD_COUNT_COL)
        tc_label = f" tc = {tc_max}" if tc_max > 1 else ""
        # only use the max thread count
        rows = filter_col_val(rows, THREAD_COUNT_COL, tc_max)
        by_selectivity = classify(rows, SELECTIVITY_COL)
        selectivities = sorted(unique_col_vals(rows, SELECTIVITY_COL))
        x = selectivities
        y = []
        for s in selectivities:
            s_rows = by_selectivity[s]
            avg = col_average(s_rows, y_val_col)
            y.append(avg)

        ax.set_xlabel("selectivity")
        ax.plot(
            x, y,
            marker=get_marker_from_str(case),
            color=get_color_from_str(case),
            markerfacecolor='none',
            label=f"{case}{tc_label}", alpha=0.7)
    ax.set_xticks(unique_col_vals(data, SELECTIVITY_COL))
    if log:
        ax.set_yscale("log")
        ax.set_ylim(min_col_val(elem_count_filtered, y_val_col))
    else:
        if not nz:
            ax.set_ylim(0)

    fig.subplots_adjust(bottom=0.3, wspace=0.33,
                        left=0.05, right=0.95, top=0.95)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    fig.savefig(plot_name)


def read_csv(path):
    data = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=';')
        header = next(reader)  # skip header

        for csv_row in reader:
            data_row = [None] * COLUMN_COUNT
            data_row[APPROACH_COL] = csv_row[APPROACH_COL]
            data_row[THREAD_COUNT_COL] = int(csv_row[THREAD_COUNT_COL])
            data_row[DATA_TYPE_COL] = csv_row[DATA_TYPE_COL]
            data_row[ELEMENT_COUNT_COL] = int(csv_row[ELEMENT_COUNT_COL])
            data_row[MASK_DISTRIBUTION_KIND_COL] = csv_row[MASK_DISTRIBUTION_KIND_COL]
            data_row[SELECTIVITY_COL] = float(csv_row[SELECTIVITY_COL])
            data_row[RUNTIME_MS_COL] = (float(csv_row[RUNTIME_MS_COL]))
            data_row[THROUGHPUT_COL] = (
                # 1/8th byte for the mask bit
                (DATA_TYPE_SIZES[data_row[DATA_TYPE_COL]] + 0.125) *
                data_row[ELEMENT_COUNT_COL]
            ) / data_row[RUNTIME_MS_COL] * 1000 / 2**30
            data_row[CASE_COL] = (
                data_row[APPROACH_COL] + "-"
                + data_row[MASK_DISTRIBUTION_KIND_COL]
                + "-"
                + data_row[DATA_TYPE_COL]
            )
            data.append(data_row)
    return data


def parallel(fns):
    processes = []
    for fn in fns:
        p = multiprocessing.Process(target=fn)
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def sequential(fns):
    for fn in fns:
        fn()


def timestamp(msg):
    global process_start_time
    print(
        "["
        + str(round(10**-6 * (time.time_ns() - process_start_time), 3))
        + " ms]: "
        + msg
    )


def jitter_filter(data, max_dev):
    res = list(data)
    cls_cols = list(COLUMNS)
    cls_cols.remove(THROUGHPUT_COL)
    cls_cols.remove(RUNTIME_MS_COL)
    filtered = 0
    classes = classify_mult(res, cls_cols)
    for c in classes.values():
        avg = col_average(c, THROUGHPUT_COL)
        for r in c:
            if math.fabs(r[THROUGHPUT_COL] - avg) > avg * max_dev:
                res.remove(r)
                filtered += 1
    print(f"filtered {100.0 * filtered/len(data):.2f} % jittered values")
    return res


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "cpu_data.csv"
    data_raw = read_csv(filename)

    data_avg = average_columns(data_raw, [RUNTIME_MS_COL, THROUGHPUT_COL])

    data_avx = list(
        filter(lambda row: "avx" in row[APPROACH_COL].lower(), data_avg))

    # generate graphs (in parallel)
    jobs = [
        lambda: throughput_over_selectivity(data_avg),
        lambda: throughput_over_selectivity(data_avg, True),
        lambda: throughput_over_thread_count(data_avg),
        lambda: throughput_over_thread_count(data_avg, True),
        lambda: single_threaded_througput_over_thread_count(data_avg),
        lambda: mt_speedup_over_thread_count(data_avg)
    ]

    if len(data_avx):
        jobs.extend([
            lambda: throughput_over_selectivity(
                data_avx, name_appendage="avx"),
            lambda: throughput_over_selectivity(
                data_avx, name_appendage="avx", nz=True),
            lambda: throughput_over_selectivity(
                data_avx, True, name_appendage="avx"),
        ])
    else:
        print("no avx benchmarks found.")

    def add_clustering_job(jobs, data, log, name_appendage, nz):
        jobs.append(lambda: throughput_over_selectivity(
            data, log=log, name_appendage=name_appendage, nz=nz
        ))
        jobs.append(lambda: throughput_over_thread_count(
            data, log=log, name_appendage=name_appendage, nz=nz
        ))
        if not log and not nz:
            jobs.append(lambda: single_threaded_througput_over_thread_count(
                data, log=log, name_appendage=name_appendage, nz=nz
            ))
            jobs.append(lambda: mt_speedup_over_thread_count(
                data, name_appendage=name_appendage, nz=nz
            ))

    mdks = unique_col_vals(data_avg, MASK_DISTRIBUTION_KIND_COL)
    for m in mdks:
        for is_log in [False, True]:
            add_clustering_job(
                jobs,
                filter_col_val(data_avg, MASK_DISTRIBUTION_KIND_COL, m),
                is_log,
                m,
                None
            )
            if not is_log:
                add_clustering_job(
                    jobs,
                    filter_col_val(data_avg, MASK_DISTRIBUTION_KIND_COL, m),
                    is_log,
                    m,
                    True
                )
    parallel(jobs)
    # sequential(jobs)


if __name__ == "__main__":
    main()
