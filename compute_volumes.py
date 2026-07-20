"""Compute hyperbolic volumes for the data_j2+j3 knots with SnapPy.

Run with the DEDICATED venv (knot SnapPy clashes with python-snappy, which
numbers-parser needs, so it must NOT live in the nsga2 env):

    /scratch/cb5330/snappy-env/bin/python compute_volumes.py

Name mapping: <=10 crossings use Rolfsen names as-is ('8_20'); 11-12 crossing
names '11a_123' / '12n_45' map to Hoste-Thistlethwaite census names 'K11a123' /
'K12n45'.  Torus knots are skipped (not hyperbolic).  Only geometric solutions
are accepted.  Writes data_j2+j3/volumes.json = {knot_name: volume}.
"""
import json
import os
import re

import snappy

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data_j2+j3", "dataset_cache.json")
OUT = os.path.join(HERE, "data_j2+j3", "volumes.json")

TORUS_KNOTS = {"3_1", "5_1", "7_1", "9_1", "11a_367", "8_19", "10_124"}


def snappy_name(name: str) -> str:
    m = re.fullmatch(r"(\d+)([an])_(\d+)", name)
    if m:  # 11a_1 -> K11a1
        return f"K{m.group(1)}{m.group(2)}{m.group(3)}"
    return name  # Rolfsen '8_20'


def main():
    with open(CACHE) as fh:
        names = sorted({row["knot"] for row in json.load(fh)})

    vols, failed = {}, []
    for name in names:
        if name in TORUS_KNOTS:
            continue
        try:
            M = snappy.Manifold(snappy_name(name))
            sol = M.solution_type()
            if "positively oriented" not in sol:
                M = snappy.Manifold(snappy_name(name))
                M.randomize()
                sol = M.solution_type()
            v = float(M.volume())
            if "positively oriented" not in sol or v <= 0.5:
                failed.append((name, sol, v))
                continue
            vols[name] = v
        except Exception as e:  # census lookup failure etc.
            failed.append((name, "error", str(e)[:60]))

    with open(OUT, "w") as fh:
        json.dump(vols, fh, indent=1)
    print(f"volumes: {len(vols)} knots -> {OUT}")
    print(f"skipped torus: {len(TORUS_KNOTS)}   failed: {len(failed)}")
    for f in failed[:10]:
        print("  FAIL:", f)


if __name__ == "__main__":
    main()
