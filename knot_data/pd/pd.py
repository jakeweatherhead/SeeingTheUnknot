# The MIT License (MIT)
# Copyright © 2025 Jake Weatherhead

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# To run: $ ./run_sage pd.py

import os
import json
import random
import hashlib
import gc
from pathlib import Path
from multiprocessing import get_context

from spherogram import random_link
from sage.knots.link import Link

unknots_src  = 'GoogleDeepmind_hard_unknots.csv'
dist_json    = 'GoogleDeepmind_hard_unknots_dist.json'
knots_json   = 'SeeingTheUnknot_non_trivial_knots.json'

LOWER_NC           = 20
UPPER_NC           = 40
MAX_RETRIES        = 100
CONTRIBUTION_LIMIT = 20_000

PROCESSES          = os.cpu_count() - 3
MAXTASKSPERCHILD   = 100
CHUNKSIZE          = 8
FLUSH_EVERY        = 500

_GLOBAL = {"seen": None, "lock": None}


def _canonical_pd(pd):
    return tuple(tuple(int(x) for x in row) for row in pd)

def _pd_hash(cpd):
    s = json.dumps(cpd, separators=(",", ":"), ensure_ascii=False)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()

def _init_pool(seen_proxy, lock_proxy):
    _GLOBAL["seen"] = seen_proxy
    _GLOBAL["lock"] = lock_proxy

def _worker(task):
    seen = _GLOBAL["seen"]
    lock = _GLOBAL["lock"]

    NC, sample_id, alternating = task
    file_key = f"{NC}{'a' if alternating else 'n'}{sample_id}"
    cpd = None

    try:
        L = random_link(
            crossings=NC,
            alternating=alternating,
            consistent_twist_regions=True,
            max_tries=MAX_RETRIES
        )

        pd = L.PD_code()

        if any(0 in row for row in pd): # for SageMath compatibility
            pd = [[e + 1 for e in row] for row in pd]

        cpd = _canonical_pd(pd)
        key = _pd_hash(cpd)

        with lock:
            if key in seen:
                return (0, NC, file_key, None, "duplicate PD (skipped)")
            seen[key] = 1

        try:
            jp = L.jones_polynomial()
        except Exception as _e:
            return (3, NC, file_key, None, f"JonesPolynomialError: {_e}")

        if jp == 1:
            del L
            gc.collect()
            return (1, NC, file_key, None, "Unknot detected (Jones=1)")

        del L, pd
        gc.collect()

        return (2, NC, file_key, cpd, "non-trivial PD stored")

    except Exception as e:
        try:
            del L
        except Exception:
            pass
        gc.collect()
        return (3, NC, file_key, None, f"{type(e).__name__}: {e}")

def _flush_to_json(entries, all_knots):
    if not entries:
        return 0

    count = 0
    for NC, file_key, cpd in entries:
        nc_str = str(NC)
        bucket = all_knots["non-trivial-knots"].setdefault(nc_str, {})
        if file_key not in bucket:
            bucket[file_key] = str([list(row) for row in cpd]).replace(' ', '')
            count += 1

    if count:
        with open(knots_json, "w") as f:
            json.dump(all_knots, f, indent=4)

    return count

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
        all_knots["non-trivial-knots"].setdefault(str(nc), {})

    tasks = []
    for NC in range(LOWER_NC, UPPER_NC + 1):
        have = hard_unknots.get(NC, 0)
        target = min((have + 1) if have > 0 else 0, CONTRIBUTION_LIMIT)
        for sample_id in range(target):
            alternating = random.choice([True, False])
            tasks.append((NC, sample_id, alternating))

    if not tasks:
        return

    ctx = get_context("fork")
    manager = ctx.Manager()
    seen = manager.dict()
    lock = manager.Lock()

    for nc_dict in all_knots["non-trivial-knots"].values():
        for pd_code_str in nc_dict.values():
            try:
                pd_code = json.loads(pd_code_str.replace("'", '"'))
                cpd = _canonical_pd(pd_code)
                key = _pd_hash(cpd)
                seen[key] = 1
            except Exception:
                continue

    new_knot_entries = []
    done = 0
    total = len(tasks)

    with ctx.Pool(
        processes=PROCESSES,
        maxtasksperchild=MAXTASKSPERCHILD,
        initializer=_init_pool,
        initargs=(seen, lock),
    ) as pool:
        for status, NC, file_key, cpd, msg in pool.imap_unordered(_worker, tasks, chunksize=CHUNKSIZE):
            done += 1

            if status == 0:
                if done % 250 == 0 or done == total:
                    print(f"[{done}/{total}] SKIP dup  NC={NC} id={file_key}: {msg}")
            elif status == 1:
                if done % 250 == 0 or done == total:
                    print(f"[{done}/{total}] SKIP unk  NC={NC} id={file_key}: {msg}")
            elif status == 2:
                new_knot_entries.append((NC, file_key, cpd))
                if (done % 100 == 0) or (done == total):
                    print(f"[{done}/{total}] STORED PD  NC={NC} id={file_key}: {msg} - Pending entries: {len(new_knot_entries)}")
                if len(new_knot_entries) >= FLUSH_EVERY:
                    _flush_to_json(new_knot_entries, all_knots)
                    new_knot_entries.clear()
                    gc.collect()
            elif status == 3:
                print(f"[{done}/{total}] FAIL (NC={NC} id={file_key}): {msg}")

    if new_knot_entries:
        _flush_to_json(new_knot_entries, all_knots)
        new_knot_entries.clear()

def mk_dir():
    pass

if __name__ == "__main__":
    mk_dir()
    main()