import os
import ast
import gc
import json
import random
import argparse
import tempfile
import multiprocessing as mp
from PIL import Image
from sage.knots.link import Link
from spherogram import Link as sLink

try:
    LANCZOS = Image.Resampling.LANCZOS
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    LANCZOS = Image.LANCZOS
    BICUBIC = Image.BICUBIC


def _seed_rng(knot_id: str):
    try:
        pid = os.getpid()
        h = hash((pid, knot_id))
        random.seed(h & 0xFFFFFFFF)
    except Exception:
        random.seed()


def _render_and_augment(knot_id: str, pd_code, output_dir: str):
    _seed_rng(knot_id)
    os.makedirs(output_dir, exist_ok=True)

    L = Link(pd_code)
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{knot_id}_",
        suffix=".png", 
        delete=False
    )
    tmp_png_path = tmp.name
    tmp.close()

    try:
        p = L.plot(gap=0.3, thickness=1.4, color='black')
        p.save(tmp_png_path, dpi=300)
        del p

        with Image.open(tmp_png_path) as img:
            img_resized = img.resize((224, 224), LANCZOS)

            original_width, original_height = img_resized.size

            ZOOM_BOUNDS = (1.4, 2.0)
            ROTATE_BOUNDS = (30, 330)
            
            zoom_coefficient = random.uniform(*ZOOM_BOUNDS)
            rotate_angle = random.uniform(*ROTATE_BOUNDS)

            super_scale_factor = 4.0
            super_size = (
                int(original_width * super_scale_factor),
                int(original_height * super_scale_factor)
            )
            super_img = img_resized.resize(super_size, LANCZOS)

            rotated_super = super_img.rotate(
                rotate_angle,
                expand=True,
                fillcolor='white',
                resample=BICUBIC
            )

            target_width = int(rotated_super.size[0] / super_scale_factor / zoom_coefficient)
            target_height = int(rotated_super.size[1] / super_scale_factor / zoom_coefficient)
            final_rotated = rotated_super.resize((target_width, target_height), LANCZOS)

            canvas   = Image.new('RGB', (original_width, original_height), 'white')
            x_offset = max(0, (original_width - target_width) // 2)
            y_offset = max(0, (original_height - target_height) // 2)
            canvas.paste(final_rotated, (x_offset, y_offset))

            out_path = os.path.join(
                output_dir,
                f"{'K' if output_dir.endswith('/knot/') else 'U'}_{knot_id}_zoom{zoom_coefficient:.1f}_rot{rotate_angle:.1f}.png"
            )
            canvas.save(out_path, format='PNG', optimize=False, compress_level=1)

            del super_img, rotated_super, final_rotated, canvas, img_resized

        del L
        return out_path
    finally:
        try:
            os.remove(tmp_png_path)
        except Exception:
            pass
        gc.collect()


def _read_jobs(source_json: str, output_dir: str):
    with open(source_json, 'r') as f:
        data = json.load(f)

    # category     = "unknots" OR "non-trivial-knots" depending on source file
    # crossing_num = [20, 40]
    for category, crossings_dict in data.items():
        for crossing_num, knots_dict in crossings_dict.items():
            CONTRIBUTION_LIMIT = 17_500
            contribution = 0
            for knot_id, pd_code_str in knots_dict.items():
                if contribution == CONTRIBUTION_LIMIT:
                    break
                pd_code = ast.literal_eval(pd_code_str)
                contribution += 1
                yield (knot_id, pd_code, output_dir)

def _run_one(job):
    knot_id, pd_code, outdir = job
    return _render_and_augment(knot_id, pd_code, outdir)

def process_pd_codes(source_file: str, output_dir: str, n_workers: int):
    jobs = _read_jobs(source_file, output_dir)

    print(f"[{source_file}] Streaming jobs -> {output_dir} with {n_workers} workers…")

    ctx = mp.get_context()
    with ctx.Pool(processes=n_workers, maxtasksperchild=200) as pool:
        done = 0
        for out_path in pool.imap_unordered(_run_one, jobs, chunksize=4):
            done += 1
            if done % 50 == 0:
                print(f"  [{source_file}] {done}… last saved: {out_path}")

    print(f"[{source_file}] Completed {done} jobs.")


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("SAGE_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    src: str = "SeeingTheUnknot_unknots.json"
    out: str = "/home/jake/Personal/SeeingTheUnknot/knot_data/diagram/unknot/"
    process_pd_codes(src, out, n_workers=os.cpu_count() - 3)


def single_plot():
    L = random_link(
            crossings=5000,
            num_components=1,  # Prevent n-component links (n >= 2)
            alternating=True,
            consistent_twist_regions=True,
            max_tries=1_000
        )

    pd = L.PD_code()

    if any(0 in row for row in pd):  # for SageMath compatibility
                pd = [[e + 1 for e in row] for row in pd]

    L = Link(pd)
    p = L.plot(gap=0.25, thickness=1.5, color='black')
    p.save(f'5000.png', dpi=300)

if __name__ == "__main__":
    single_plot()
    main()