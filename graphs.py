#!/usr/bin/env python3
#import seaborn as sns
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
DATA_TYPE_COL = 1
ELEMENT_COUNT_COL = 2
MASK_DISTRIBUTION_KIND_COL = 3
SELECTIVITY_COL = 4
RUNTIME_MS_COL = 5
THROUGHPUT_COL = 6
CASE_COL = 7
COLUMN_COUNT = 8
VIRTUAL_COLUMN_COUNT = 2

COLUMNS = list(range(0, COLUMN_COUNT))


colors = [
    'darkorange', 'green', 'gold', 'deepskyblue', 'yellow', 'navy',
    'darkred', 'purple', 'red', 'mediumpurple', 'turquoise',
    'lightgray', 'teal', 'lime', 'seagreen', 'plum', 'lightsteelblue',
    'palevioletred', 'black', 'blue', 'magenta', 'olive', 'goldenrod'
]

markers = ['^', '^', '+', '<', 'x', '1', '2', '*', '>', '2', 'D',
           'o', '+', '3', '4', 'D', 'D', 'd', 'd', 'H', 'h', 'p', 'P', 's']

data_type_sizes = {
    "float": 4,
    "double": 8,
    "uint8_t": 4,
    "uint16_t": 4,
    "uint32_t": 4,
    "uint64_t": 8
}


def get_marker_from_str(s):
    return markers[hash(s) % len(markers)]


def get_color_from_str(s):
    return colors[hash(s) % len(colors)]


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


def throughput_over_selectivity(data, log=False, use_runtime=False, name_appendage=None):
    y_axis_name = ("runtime" if use_runtime else "throughput")
    plot_name = (
        y_axis_name
        + "_over_selectivity"
        + (f"_{name_appendage}" if name_appendage else "") +
        ("_log" if log else "") + ".png"
    )
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
                 + f"(element count = {max_elem_count})")

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
        by_selectivity = classify(rows, SELECTIVITY_COL)
        selectivities = sorted(unique_col_vals(rows, SELECTIVITY_COL))
        x = selectivities
        y = []
        for s in selectivities:
            s_rows = by_selectivity[s]
            if use_runtime:
                avg = col_average(s_rows, RUNTIME_MS_COL)
            else:
                avg = col_average(s_rows, THROUGHPUT_COL)
            y.append(avg)

        ax.plot(
            x, y,
            marker=get_marker_from_str(case),
            color=get_color_from_str(case),
            markerfacecolor='none',
            label=f"{case}", alpha=0.7)
    ax.set_xticks(unique_col_vals(data, SELECTIVITY_COL))
    if log:
        ax.set_yscale("log")

    else:
        ax.set_ylim(0)
    fig.subplots_adjust(bottom=0.3, wspace=0.33,
                        left=0.05, right=0.95, top=0.95)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    fig.savefig(plot_name)


def read_csv(path):
    data = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=';')
        _header = next(reader)  # skip header

        for csv_row in reader:
            data_row = [None] * COLUMN_COUNT
            data_row[APPROACH_COL] = csv_row[APPROACH_COL]
            data_row[DATA_TYPE_COL] = csv_row[DATA_TYPE_COL]
            data_row[ELEMENT_COUNT_COL] = int(csv_row[ELEMENT_COUNT_COL])
            data_row[MASK_DISTRIBUTION_KIND_COL] = csv_row[MASK_DISTRIBUTION_KIND_COL]
            data_row[SELECTIVITY_COL] = float(csv_row[SELECTIVITY_COL])
            data_row[RUNTIME_MS_COL] = (float(csv_row[RUNTIME_MS_COL]))
            data_row[THROUGHPUT_COL] = (
                # 1/8th byte for the mask bit
                (data_type_sizes[data_row[DATA_TYPE_COL]] + 0.125) *
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

    # average runs since we basically always need this
    data_avg = average_columns(data_raw, [RUNTIME_MS_COL])

    # generate graphs (in parallel)
    jobs = [
        lambda: throughput_over_selectivity(data_avg),
    ]
    mdks = unique_col_vals(data_avg, MASK_DISTRIBUTION_KIND_COL)
    for m in mdks:
        for is_log in [False, True]:
            jobs.append(
                functools.partial(
                    lambda args: throughput_over_selectivity(
                        filter_col_val(
                            data_avg, MASK_DISTRIBUTION_KIND_COL, args[0]),
                        name_appendage=args[0], log=args[1]
                    ),
                    (m, is_log)
                )
            )
    # parallel(jobs)
    sequential(jobs)


if __name__ == "__main__":
    main()
