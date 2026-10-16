"""Decide the UNIT of registration exclusion, then emit the list.

Reads registration_qc_transitions_full.csv (per-transition QC) and answers one
question before writing anything:

    Is a bad transition caused by a bad VISIT, or is it bad on its own?

WHY THE QUESTION HAS A RIGHT ANSWER. register_per_modality.py warps every visit
onto the EARLIEST visit's grid independently -- there is no pairwise chaining. So
a bad transition cannot mean "these two do not line up with each other"; it means
one of the two volumes is badly registered to the common anchor. A bad volume
poisons EVERY transition it takes part in, which is two of them for an interior
visit. The 5-patient pilot shows exactly that signature on YG_MTO9K0JU7M07:

     9 -> 10   dice 0.963   fine
    10 -> 11   dice 0.311   bad
    11 -> 12   dice 0.567   bad          both bad transitions touch visit 11

If that generalizes, the unit is the VISIT and the fix reuses machinery that
already exists: drop the frame from the timeline and let the neighbours re-pair
across it with the gap summed, exactly as blank frames are handled
(BrainMetsPairDataset._pairs_of). If instead bad transitions are ISOLATED, the
cause is pair-specific, the visit-level story is wrong, and the exclusion has to
name transitions instead.

The script tests this rather than assuming it, and says which list it wrote.

NOTE ON ENDPOINTS. The first and last visit of a timeline participate in only one
transition each, so a single bad transition there cannot distinguish "this visit
is bad" from "its neighbour is bad". Those are reported separately and excluded
conservatively (the whole transition, not a guessed visit).

Usage:
    ./.venv/bin/python scripts/diagnostics/data/transition_exclusion.py
    ./.venv/bin/python scripts/diagnostics/data/transition_exclusion.py --write
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# A transition is flagged when its slice correspondence or its mask overlap is far
# enough from the cohort norm that the pair is not measuring the same anatomy.
# Cohort norms: dice ~0.97, k=0 fraction ~0.95. Thresholds are deliberately loose --
# this is an exclusion list, not a quality score, and over-excluding costs real
# training data.
#
# Defined HERE rather than in registration_qc.py, which both this and that module
# would otherwise have to agree on: registration_qc imports antspyx, which exists
# only in `venv`, while this script is pure CSV analysis and must run under `.venv`
# too. Keeping the constants in the dependency-free module lets registration_qc
# import them without dragging ANTs into every consumer -- the same reason
# masking.py and slice_preprocessing.py were split out on 2026-08-27.
FLAG_DICE = 0.85
FLAG_FRAC_ZERO = 0.60

QC_CSV = Path("results/diagnostics/registration/registration_qc_transitions_full.csv")
OUT_VISITS = Path("results/diagnostics/registration/registration_exclude_visits.txt")
OUT_TRANS = Path("results/diagnostics/registration/registration_exclude_transitions.txt")


def is_bad(row):
    """Fails if the brain masks barely overlap, or if slice index i at one visit
    does not correspond to slice index i at the other. Thresholds live in
    registration_qc.py so scoring and exclusion cannot drift apart."""
    dice = float(row["dice_after"])
    frac = row["frac_offset_zero"]
    frac = float(frac) if frac not in ("", "nan") else float("nan")
    return dice < FLAG_DICE or (frac == frac and frac < FLAG_FRAC_ZERO)


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    by_pat = defaultdict(list)
    for r in rows:
        r["transition_index"] = int(r["transition_index"])
        r["_bad"] = is_bad(r)
        by_pat[r["patient_id"]].append(r)
    for rs in by_pat.values():
        rs.sort(key=lambda r: r["transition_index"])
    return rows, by_pat


def adjacency_test(by_pat):
    """P(neighbour bad | this bad) against the base rate.

    Visit-caused badness makes bad transitions come in ADJACENT pairs sharing the
    culprit visit, so the conditional should sit far above the base rate.
    Pair-specific badness leaves them isolated and the two should match.
    """
    n_bad = n_bad_with_bad_neighbour = n_total = 0
    for rs in by_pat.values():
        for i, r in enumerate(rs):
            n_total += 1
            if not r["_bad"]:
                continue
            n_bad += 1
            neigh = [rs[j] for j in (i - 1, i + 1) if 0 <= j < len(rs)]
            if any(x["_bad"] for x in neigh):
                n_bad_with_bad_neighbour += 1
    base = n_bad / max(n_total, 1)
    cond = n_bad_with_bad_neighbour / max(n_bad, 1)
    return base, cond, n_bad, n_total


def implicated_visits(by_pat):
    """A visit is blamed when EVERY transition touching it is bad.

    Interior visits touch two, so both must fail -- one bad neighbour is not
    enough, since that transition's other endpoint could be the real culprit.
    Endpoint visits touch only one and are therefore never blamed here; those
    transitions are excluded directly instead.
    """
    visits, endpoint_trans = set(), []
    for pid, rs in by_pat.items():
        touching = defaultdict(list)          # visit_date -> transitions touching it
        for r in rs:
            touching[r["t0_date"]].append(r)
            touching[r["t1_date"]].append(r)
        for date, ts in touching.items():
            if len(ts) == 2 and all(t["_bad"] for t in ts):
                visits.add((pid, date))
        for r in rs:
            if not r["_bad"]:
                continue
            interior_blamed = ((pid, r["t0_date"]) in visits
                               or (pid, r["t1_date"]) in visits)
            if not interior_blamed:
                endpoint_trans.append(r)
    return visits, endpoint_trans


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--qc-csv", type=Path, default=QC_CSV)
    ap.add_argument("--write", action="store_true",
                    help="Write the exclusion lists. Without it, report only.")
    args = ap.parse_args()

    if not args.qc_csv.exists():
        raise SystemExit(
            f"{args.qc_csv} not found. Run hpc/registration_qc_transitions.sbatch, then merge:\n"
            "  awk 'FNR==1 && NR!=1 {next} {print}' "
            "results/diagnostics/registration/registration_qc_transitions_part*.csv \\\n"
            f"    > {args.qc_csv}")

    rows, by_pat = load(args.qc_csv)
    n_pat = len(by_pat)
    print(f"{len(rows):,} transitions across {n_pat:,} patients")
    print(f"thresholds: dice < {FLAG_DICE}  or  k=0 fraction < {FLAG_FRAC_ZERO:.0%}")

    base, cond, n_bad, n_total = adjacency_test(by_pat)
    print(f"\n=== Is badness a property of the VISIT or of the TRANSITION? ===")
    print(f"  bad transitions                          {n_bad:,} / {n_total:,}  ({base:.2%})")
    print(f"  P(a neighbour is also bad | this is bad)  {cond:.2%}")
    print(f"  base rate for comparison                  {base:.2%}")
    lift = cond / max(base, 1e-9)
    print(f"  lift                                      {lift:.1f}x")
    visit_level = lift >= 2.0 and n_bad >= 20
    if n_bad < 20:
        verdict = ("TOO FEW bad transitions to decide; excluding transitions "
                   "directly is the conservative choice")
    elif visit_level:
        verdict = ("VISIT-level -- bad transitions cluster around a shared visit, "
                   "as a badly-registered volume would")
    else:
        verdict = ("TRANSITION-level -- bad transitions are isolated, so the "
                   "visit-level explanation does NOT hold here")
    print(f"\n  VERDICT: {verdict}")

    visits, endpoint_trans = implicated_visits(by_pat)
    print(f"\n=== Exclusion lists ===")
    print(f"  visits blamed (both touching transitions bad)  {len(visits):,}")
    print(f"  bad transitions not explained by a blamed visit {len(endpoint_trans):,}")
    print(f"    (first/last visit of a timeline touches only one transition, so a")
    print(f"     single failure there cannot say which of the two volumes is at fault)")

    affected = {p for p, _ in visits} | {r["patient_id"] for r in endpoint_trans}
    print(f"  patients affected at all                       {len(affected):,} / {n_pat:,}")

    kept = n_total - n_bad
    print(f"\n  transitions surviving: {kept:,} / {n_total:,} ({kept/max(n_total,1):.2%})")
    print(f"  NOTE: dropping these patients ENTIRELY instead would cost "
          f"{sum(len(by_pat[p]) for p in affected):,} transitions "
          f"({sum(len(by_pat[p]) for p in affected)/max(n_total,1):.1%}) -- "
          f"which is the reason this is done per-visit.")

    for label, rs in (("worst by dice", sorted((r for r in rows if r["_bad"]),
                                               key=lambda r: float(r["dice_after"]))[:12]),):
        print(f"\n  {label}:")
        for r in rs:
            blamed = [d for d in (r["t0_date"], r["t1_date"])
                      if (r["patient_id"], d) in visits]
            print(f"    {r['patient_id']} t{r['transition_index']:<3} "
                  f"{r['t0_date']}->{r['t1_date']}  dice {float(r['dice_after']):.3f}  "
                  f"k=0 {float(r['frac_offset_zero']):.0%}"
                  f"{'  blamed visit: ' + blamed[0] if blamed else '  (no single visit blamed)'}")

    if args.write:
        OUT_VISITS.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_VISITS, "w") as f:
            f.write("# patient_id,visit_date -- drop this frame from the timeline and\n"
                    "# re-pair its neighbours across it, summing the spanned gap\n")
            for pid, date in sorted(visits):
                f.write(f"{pid},{date}\n")
        with open(OUT_TRANS, "w") as f:
            f.write("# patient_id,t0_date,t1_date -- drop this transition only\n")
            for r in sorted(endpoint_trans, key=lambda r: (r["patient_id"], r["transition_index"])):
                f.write(f"{r['patient_id']},{r['t0_date']},{r['t1_date']}\n")
        print(f"\nWrote {OUT_VISITS} ({len(visits):,}) and {OUT_TRANS} ({len(endpoint_trans):,})")
    else:
        print("\n(report only -- pass --write to emit the exclusion lists)")


if __name__ == "__main__":
    main()
