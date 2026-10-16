"""
Per-modality within-patient registration.

Unlike the previous version (which required all 4 modalities present in
a visit before using it), this version treats each modality as its own
independent timeline:

    - For each patient, look at their FLAIR scans across all visits.
      If there are 2+ FLAIR visits, register the later ones onto the
      earliest one.
    - Do the same independently for T2, PRE, and POST.

A visit missing one modality no longer excludes that visit's OTHER
modalities from being used -- only the missing modality itself is
skipped for that visit.

Minimum requirement: a patient needs at least 2 visits of a given
modality for that modality to be used at all (need at least 2 points
in time to predict change).

This script only READS from root_dir and only WRITES into output_dir.
Original data is never modified.

Usage:
    # All patients found in root_dir:
    python3 register_per_modality.py /path/to/Yale-Brain-Mets-Longitudinal /path/to/output_folder

    # One patient only (for per-patient parallel SLURM jobs -- see hpc/register.sbatch):
    python3 register_per_modality.py /path/to/Yale-Brain-Mets-Longitudinal /path/to/output_folder --patient YG_0123ABC

    # Cap to the first N patients (quick local testing):
    python3 register_per_modality.py /path/to/Yale-Brain-Mets-Longitudinal /path/to/output_folder --max-patients 10
"""

import argparse
import csv
import os
import ants

MODALITIES = ["FLAIR", "PRE", "POST", "T2"]


def find_modality_file(visit_path, modality):
    """Return the filepath for one modality in one visit folder, or None if missing."""
    files_in_visit = os.listdir(visit_path)
    match = [f for f in files_in_visit if f.endswith(f"_{modality}.nii.gz")]
    if match:
        return os.path.join(visit_path, match[0])
    return None


def collect_modality_timeline(patient_path, modality):
    """
    For one patient and one modality, return a sorted list of
    (visit_date, filepath) for every visit that has this modality.
    """
    visit_dates = sorted(os.listdir(patient_path))
    visit_dates = [v for v in visit_dates if os.path.isdir(os.path.join(patient_path, v))]

    timeline = []
    for visit_date in visit_dates:
        visit_path = os.path.join(patient_path, visit_date)
        filepath = find_modality_file(visit_path, modality)
        if filepath is not None:
            timeline.append((visit_date, filepath))

    return timeline


def register_modality_timeline(patient_id, modality, timeline, output_root, metrics):
    """
    Register all later scans of one modality onto the earliest scan
    of that same modality, for one patient.
    """
    if len(timeline) < 2:
        print(f"  {modality}: only {len(timeline)} visit(s) with this modality, skipping (need 2+)")
        return

    fixed_date, fixed_path = timeline[0]
    fixed_img = ants.image_read(fixed_path)

    # Save the fixed (reference) visit as-is.
    fixed_out_dir = os.path.join(output_root, patient_id, fixed_date)
    try:
        os.makedirs(fixed_out_dir, exist_ok=True)
    except FileExistsError:
        # A different modality's loop may have already created this same
        # visit-date dir; exist_ok=True can still spuriously re-raise here
        # on hpcwork's networked filesystem due to a stale isdir() check
        # right after mkdir(). The dir existing is exactly what we want.
        pass
    ants.image_write(fixed_img, os.path.join(fixed_out_dir, f"{modality}.nii.gz"))

    print(f"  {modality}: {len(timeline)} visits found, reference = {fixed_date}")

    # Register every later scan of this modality onto the fixed one.
    for visit_date, moving_path in timeline[1:]:
        moving_img = ants.image_read(moving_path)

        result = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform="SyN"
        )
        registered_img = result["warpedmovout"]

        # Record how well this registration actually did. ants.registration
        # returns transforms but NO similarity score, so without this nothing
        # on disk says whether a given registration converged or silently
        # produced a correctly-shaped but misaligned volume -- warpedmovout
        # always inherits the fixed image's grid, so shape checks cannot catch
        # that (see scripts/diagnostics/data/registration_qc.py).
        #
        # `before` resamples the moving image onto the fixed grid by header
        # alone, no alignment: the honest baseline registration must beat.
        # ants.image_mutual_information is a COST -- more negative = more
        # similar -- so an improvement means mi_after < mi_before.
        moving_on_fixed = ants.resample_image_to_target(
            moving_img, fixed_img, interp_type="linear")
        metrics.append({
            "patient_id": patient_id,
            "modality": modality,
            "fixed_date": fixed_date,
            "moving_date": visit_date,
            "mi_before": float(ants.image_mutual_information(fixed_img, moving_on_fixed)),
            "mi_after": float(ants.image_mutual_information(fixed_img, registered_img)),
        })

        visit_out_dir = os.path.join(output_root, patient_id, visit_date)
        try:
            os.makedirs(visit_out_dir, exist_ok=True)
        except FileExistsError:
            pass
        ants.image_write(registered_img, os.path.join(visit_out_dir, f"{modality}.nii.gz"))

        print(f"    registered {visit_date} -> {fixed_date}")


def register_patient(patient_id, patient_path, output_root, metrics):
    """Register all 4 modality timelines independently for one patient."""
    for modality in MODALITIES:
        timeline = collect_modality_timeline(patient_path, modality)
        register_modality_timeline(patient_id, modality, timeline, output_root, metrics)


def parse_args():
    parser = argparse.ArgumentParser(description="Per-modality within-patient registration.")
    parser.add_argument("root_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--patient", action="append", default=None,
                         help="Register only this patient ID. Repeatable for a small explicit list. "
                              "Mutually exclusive with --max-patients.")
    parser.add_argument("--max-patients", type=int, default=None,
                         help="Cap to the first N patients (alphabetical). Mutually exclusive with --patient.")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = args.root_dir
    output_root = args.output_dir
    os.makedirs(output_root, exist_ok=True)

    all_patient_ids = sorted(os.listdir(root_dir))
    all_patient_ids = [p for p in all_patient_ids if os.path.isdir(os.path.join(root_dir, p))]

    if args.patient:
        missing = [p for p in args.patient if p not in all_patient_ids]
        if missing:
            raise SystemExit(f"--patient not found under {root_dir}: {missing}")
        patient_ids = args.patient
    elif args.max_patients is not None:
        patient_ids = all_patient_ids[:args.max_patients]
    else:
        patient_ids = all_patient_ids

    print(f"Found {len(all_patient_ids)} total patients under {root_dir}; registering {len(patient_ids)}.")

    metrics = []
    for i, patient_id in enumerate(patient_ids):
        print(f"\n[{i+1}/{len(patient_ids)}] {patient_id}")
        patient_path = os.path.join(root_dir, patient_id)
        try:
            register_patient(patient_id, patient_path, output_root, metrics)
        except Exception as e:
            print(f"  ERROR on {patient_id}: {e}")
            continue

    # Per-visit registration quality, written next to the output tree. Named
    # per-patient when --patient is used so the per-patient SLURM array jobs
    # (hpc/register.sbatch) do not overwrite each other's file.
    if metrics:
        suffix = f"_{patient_ids[0]}" if args.patient and len(patient_ids) == 1 else ""
        metrics_path = os.path.join(output_root, f"registration_metrics{suffix}.csv")
        write_header = not os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(metrics)
        improved = sum(1 for m in metrics if m["mi_after"] < m["mi_before"])
        print(f"\nRegistration quality: {improved}/{len(metrics)} registrations improved "
              f"similarity (MI is a cost; lower = better). Wrote {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()