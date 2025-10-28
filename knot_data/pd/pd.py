import os
import json
import random
import hashlib
from pathlib import Path
from multiprocessing import get_context

from spherogram import random_link
from sage.knots.link import Link
from PIL import Image

unknots_src  = 'GoogleDeepmind_hard_unknots.csv'
dist_json    = 'GoogleDeepmind_hard_unknots_dist.json'
knots_json   = 'SeeingTheUnknot_non_trivial_knots.json'

LOWER_NC           = 12      # Lower limit for number of crossings (NC)
UPPER_NC           = 40      # Upper limit for NC
CROSSING_GAP       = 0.3  
STRAND_THICKNESS   = 1.5
MAX_RETRIES        = 100
CONTRIBUTION_LIMIT = 40_000  # Maximum diagram count contribution of any NC dataset
OUT_ROOT           = Path("../diagram/non-trivial-knots")


def _canonical_pd(pd):
    """Canonical, hashable representation: tuple of tuples (keep row order)."""
    rows = [tuple(int(x) for x in row) for row in pd]
    return tuple(rows)


def _pd_hash(cpd):
    """Small, stable content hash to minimize shared memory traffic."""
    s = json.dumps(cpd, separators=(",", ":"), ensure_ascii=False)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()


def _worker(args):
    """
    args = (shared_seen_dict, shared_lock, (NC, sample_id, alternating))
    """
    shared_seen, shared_lock, task = args
    NC, sample_id, alternating     = task
    file_key                       = f"{NC}{'a' if alternating else 'n'}{sample_id}"
    cpd                            = None

    try:
        L = random_link(
            crossings=NC,
            alternating=alternating,
            consistent_twist_regions=True,
            max_tries=MAX_RETRIES
        )

        pd = L.PD_code()
        pd = [list(tup) for tup in pd]
        if any(0 in row for row in pd):
            pd = [[e + 1 for e in row] for row in pd]

        cpd = _canonical_pd(pd)
        key = _pd_hash(cpd)

        with shared_lock:
            if key in shared_seen:
                return (0, NC, file_key, cpd, "duplicate PD (skipped)")
            shared_seen[key] = 1

        link = Link([list(r) for r in cpd])

        if link.jones_polynomial() == 1:
            return (1, NC, file_key, cpd, "Unknot detected (Jones=1)")

        p = link.plot(
            gap=CROSSING_GAP,
            thickness=STRAND_THICKNESS,
            color='black'
        )

        nc_dir = OUT_ROOT / f"{NC}"
        nc_dir.mkdir(parents=True, exist_ok=True)

        tmp = nc_dir / f"__tmp_{NC}_{sample_id}_{os.getpid()}.png"
        p.save(str(tmp), dpi=300)

        img = Image.open(str(tmp))
        img_resized = img.resize((224, 224), Image.LANCZOS)
        
        final_name = f"{file_key}.png"
        
        out_path = nc_dir / final_name
        img_resized.save(str(out_path))

        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

        return (
            2,         # ok status
            NC,           # number of crossings
            file_key,     # knot identifier
            cpd,          # PD code
            str(out_path) # diagram location
        )

    except Exception as e:
        return (
            3, 
            NC, 
            file_key, 
            cpd, 
            f"{type(e).__name__}: {e}"
        )


def main():
    with open(dist_json, "r") as f:
        hard_unknots = json.load(f)

    hard_unknots = {
        int(nc): int(num_pd_codes)
        for nc, num_pd_codes in hard_unknots.items()
        if LOWER_NC <= int(nc) <= UPPER_NC
    }

    if Path(knots_json).exists():
        try:
            with open(knots_json, "r") as f:
                all_knots = json.load(f)
            if "non-trivial-knots" not in all_knots:
                 all_knots["non-trivial-knots"] = {}
        except json.JSONDecodeError:
            all_knots = {"non-trivial-knots": {}}
    else:
        all_knots = {"non-trivial-knots": {}}

    for nc in range(LOWER_NC, UPPER_NC + 1):
        if str(nc) not in all_knots["non-trivial-knots"]:
            all_knots["non-trivial-knots"][str(nc)] = {}

    tasks = []
    for NC in range(LOWER_NC, UPPER_NC + 1):
        have = hard_unknots.get(NC, 0)
        target = min((have + 1) if have > 0 else 0, CONTRIBUTION_LIMIT)
        (OUT_ROOT / f"{NC}").mkdir(parents=True, exist_ok=True)
        for sample_id in range(target):
            alternating = random.choice([True, False])
            tasks.append((NC, sample_id, alternating))

    if not tasks:
        return

    ctx = get_context("fork")
    manager = ctx.Manager()
    seen = manager.dict()
    
    for nc_dict in all_knots["non-trivial-knots"].values():
        for pd_code in nc_dict.values():
            cpd = _canonical_pd(pd_code)
            key = _pd_hash(cpd)
            seen[key] = 1
    
    lock = manager.Lock()

    arg_iter = ((seen, lock, t) for t in tasks)

    new_knot_entries = []

    with ctx.Pool(processes=os.cpu_count() - 2) as pool:
        done = 0
        total = len(tasks)
        for status, NC, file_key, cpd, msg in pool.imap_unordered(_worker, arg_iter, chunksize=1):
            done += 1
            if status == 0:
                print(f"SKIP dup  NC={NC} id={file_key}: {msg}")
            elif status == 1:
                print(f"SKIP unk  NC={NC} id={file_key}: {msg}")
            elif status == 2:
                new_knot_entries.append((NC, file_key, cpd))
                if done % 100 == 0 or done == total:
                    print(f"[{done}/{total}] Wrote {msg} - Knot Entries: {len(new_knot_entries)}")
            elif status == 3:
                print(f"FAIL (NC={NC}): {msg}")

    if not new_knot_entries:
        return

    count = 0
    for NC, file_key, cpd in new_knot_entries:
        nc_str = str(NC)
        
        if file_key not in all_knots["non-trivial-knots"][nc_str]:
            pd_list = [list(row) for row in cpd]
            all_knots["non-trivial-knots"][nc_str][file_key] = pd_list
            count += 1
            print(count)
    
    if count == 0:
        return

    try:
        with open(knots_json, "w") as f:
            json.dump(all_knots, f, indent=4)
    except Exception as e:
        pass


def mk_dir():
    """Creates diagrammatic crossing count directories."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for nc in range(LOWER_NC, UPPER_NC + 1):
        os.makedirs(OUT_ROOT / str(nc), exist_ok=True)


if __name__ == "__main__":
    mk_dir()
    main()