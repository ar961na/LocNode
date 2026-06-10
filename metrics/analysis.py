"""Compute ATE / RPE of algorithm trajectories against a groundtruth.

Trajectories are timestamp-associated and rigidly aligned (Umeyama) to the
groundtruth, then ATE and RPE are computed. Plotting (-p) is an optional
quick check.
"""

import argparse
import os.path

from tabulate import tabulate

import load
import align
import metrics


def crop_trajectory(timestamps, positions, quaternions, cutoff):
    """Keep only the poses at or after ``cutoff`` (timestamps assumed seconds)."""
    mask = timestamps >= cutoff
    return timestamps[mask], positions[mask], quaternions[mask]


def cropped_path(path):
    """Destination for a cropped copy: source dir, '<name>_cropped.txt'."""
    root, _ = os.path.splitext(path)
    return root + "_cropped.txt"


def main():
    parser = argparse.ArgumentParser(prog="analysis.py", description=__doc__)

    parser.add_argument("gt", help="path to groundtruth trajectory")
    parser.add_argument("alg", nargs="*", help="paths to algorithm trajectories")

    parser.add_argument(
        "-c",
        "--compute",
        action="store_true",
        help="(deprecated, metrics are always computed)",
    )
    parser.add_argument(
        "-d",
        "--distance",
        nargs="+",
        type=float,
        default=[1.0],
        help="sub-trajectory length(s) in meters for RPE",
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="plot aligned trajectories"
    )
    parser.add_argument(
        "-t",
        "--start",
        type=float,
        default=0.0,
        metavar="N",
        help="discard the first N seconds before computing metrics, measured "
        "from the earliest timestamp across all input trajectories (gt + alg). "
        "Lets you skip the initial unstabilized alignment phase.",
    )
    parser.add_argument(
        "--save-cropped",
        action="store_true",
        help="when --start is used, also write each cropped trajectory next to "
        "its source file as a TUM .txt suffixed '_cropped'",
    )
    parser.add_argument(
        "--gt-quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="quaternion component order in the groundtruth file "
        "(trajectories written by the pipeline are TUM format: xyzw)",
    )
    args = parser.parse_args()

    t_gt, p_gt, q_gt = load.load_trajectory(args.gt, quat_order=args.gt_quat_order)

    # Load every algorithm trajectory up front so a time crop can be measured
    # from the earliest timestamp across all inputs (gt + algs).
    alg_trajs = [(path, *load.load_trajectory(path)) for path in args.alg]

    if args.start > 0:
        origin = min([t_gt.min()] + [t_a.min() for _, t_a, _, _ in alg_trajs])
        cutoff = origin + args.start
        print(
            f"# cropping to t >= {cutoff:.6f} "
            f"(earliest {origin:.6f} + {args.start:g}s)"
        )

        t_gt, p_gt, q_gt = crop_trajectory(t_gt, p_gt, q_gt, cutoff)
        if len(t_gt) == 0:
            parser.error("groundtruth is empty after cropping; reduce --start")
        if args.save_cropped:
            load.save_trajectory(cropped_path(args.gt), t_gt, p_gt, q_gt)

        kept = []
        for path, t_a, p_a, q_a in alg_trajs:
            t_a, p_a, q_a = crop_trajectory(t_a, p_a, q_a, cutoff)
            if len(t_a) == 0:
                print(f"# {os.path.basename(path)}: empty after cropping, skipped")
                continue
            if args.save_cropped:
                load.save_trajectory(cropped_path(path), t_a, p_a, q_a)
            kept.append((path, t_a, p_a, q_a))
        alg_trajs = kept

    timestamps = [t_gt]
    positions = [p_gt]
    quaternions = [q_gt]
    results = []

    for path, t_a, p_a, q_a in alg_trajs:
        # resample groundtruth at the algorithm timestamps once; the matched
        # poses serve both the rigid alignment and the error metrics
        _, m_p_gt, m_q_gt = align.match_groundtruth_to_timestamps(t_gt, p_gt, q_gt, t_a)

        # rigidly align the algorithm trajectory to groundtruth (Umeyama)
        delta_R, delta_t = align.umeyama_alignment(m_p_gt, p_a)
        p_a, q_a = align.apply_alignment(p_a, q_a, delta_R, delta_t)

        timestamps.append(t_a)
        positions.append(p_a)
        quaternions.append(q_a)

        ate_pos, ate_rot = metrics.compute_ate(m_p_gt, m_q_gt, p_a, q_a)
        rpe_pos, rpe_rot = metrics.compute_rpe(m_p_gt, m_q_gt, p_a, q_a, args.distance)

        row = [os.path.basename(path), ate_pos, ate_rot]
        for pos_d, rot_d in zip(rpe_pos, rpe_rot):
            row += [pos_d, rot_d]
        results.append(row)

    headers = ["Trajectory", "ATE Pos", "ATE Rot"]
    for d in args.distance:
        headers += [f"RPE Pos (d={d:g})", f"RPE Rot (d={d:g})"]

    gt_row = [os.path.basename(args.gt)] + [0.0] * (len(headers) - 1)
    print(tabulate([gt_row] + results, headers=headers, floatfmt=".6f"))

    if args.plot:
        import plot

        plot.plot_trajectories(timestamps, positions, quaternions)


if __name__ == "__main__":
    main()
