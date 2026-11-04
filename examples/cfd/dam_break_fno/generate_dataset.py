# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a dam-break dataset with OpenFOAM interFoam.

Each sample is the OpenFOAM ``damBreak`` tutorial
(``multiphase/interFoam/laminar/damBreak/damBreak``) with a different initial
water column. The column width and height are drawn from a uniform
distribution, the case is solved with ``interFoam`` and the water volume
fraction ``alpha.water`` is sampled onto a uniform ``resolution x resolution``
grid at every write time. The obstacle at the bottom of the tank is stored as a
binary mask.

The output is one HDF5 file per split with the datasets

* ``alpha``: (num_cases, num_times, resolution, resolution), float32
* ``mask``: (resolution, resolution), 1 inside the obstacle
* ``params``: (num_cases, 2), column width and height in metres
* ``times``: (num_times,), write times in seconds

Example::

    python generate_dataset.py --num-cases 100 --output-dir ./data
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

# Geometry of the damBreak tutorial (blockMeshDict uses scale 0.146).
SCALE = 0.146
DOMAIN = 4.0 * SCALE  # square tank side, m
OBSTACLE_X = (2.0 * SCALE, 2.16438 * SCALE)  # obstacle left/right faces, m
OBSTACLE_H = 0.32876 * SCALE  # obstacle height, m

TUTORIAL = "multiphase/interFoam/laminar/damBreak/damBreak"


def find_template(path: str | None) -> Path:
    """Return the damBreak tutorial directory."""
    candidates = []
    if path:
        candidates.append(Path(path))
    if os.environ.get("FOAM_TUTORIALS"):
        candidates.append(Path(os.environ["FOAM_TUTORIALS"]) / TUTORIAL)
    if os.environ.get("WM_PROJECT_DIR"):
        candidates.append(Path(os.environ["WM_PROJECT_DIR"]) / "tutorials" / TUTORIAL)
    for candidate in candidates:
        if (candidate / "system" / "blockMeshDict").is_file():
            return candidate
    raise FileNotFoundError(
        "damBreak tutorial not found. Source the OpenFOAM bashrc or pass "
        "--template <path to damBreak/damBreak>."
    )


def prepare_case(
    template: Path,
    case: Path,
    width: float,
    height: float,
    refine: int,
    end_time: float,
    write_interval: float,
) -> None:
    """Copy the tutorial and set the column size, mesh density and output times."""
    shutil.copytree(template, case)

    block_mesh = case / "system" / "blockMeshDict"
    text = block_mesh.read_text()
    text = re.sub(
        r"(hex \([\d ]+\)) \((\d+) (\d+) 1\)",
        lambda m: (
            f"{m.group(1)} ({int(m.group(2)) * refine} {int(m.group(3)) * refine} 1)"
        ),
        text,
    )
    block_mesh.write_text(text)

    set_fields = case / "system" / "setFieldsDict"
    text = set_fields.read_text()
    text = re.sub(
        r"box \(0 0 -1\) \([^)]*\);",
        f"box (0 0 -1) ({width:.6f} {height:.6f} 1);",
        text,
    )
    set_fields.write_text(text)

    control = case / "system" / "controlDict"
    text = control.read_text()
    # The tutorial's optional sampling function objects are not needed here and
    # fail to load in some packaged OpenFOAM builds.
    text = re.sub(r"^#sinclude.*sampling.*$", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^endTime\s+[^;]+;", f"endTime         {end_time};", text, flags=re.M
    )
    text = re.sub(
        r"^writeInterval\s+[^;]+;",
        f"writeInterval   {write_interval};",
        text,
        flags=re.M,
    )
    control.write_text(text)

    shutil.copyfile(case / "0" / "alpha.water.orig", case / "0" / "alpha.water")


def run(cmd: list[str], case: Path) -> None:
    """Run an OpenFOAM utility in ``case`` and raise with the log tail on failure."""
    log = case / f"log.{cmd[0]}"
    with open(log, "w") as f:
        result = subprocess.run(cmd, cwd=case, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = log.read_text().splitlines()[-20:]
        raise RuntimeError(f"{' '.join(cmd)} failed in {case}:\n" + "\n".join(tail))


def read_internal_field(path: Path, num_cells: int, components: int = 1) -> np.ndarray:
    """Read the internalField of an ASCII OpenFOAM field file."""
    text = path.read_text()
    uniform = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if uniform:
        value = np.array(uniform.group(1).strip("() ").split(), dtype=np.float64)
        return np.tile(value, (num_cells, 1)).squeeze()
    match = re.search(r"internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\(", text)
    if match is None:
        raise ValueError(f"cannot parse internalField in {path}")
    count = int(match.group(1))
    start = match.end()
    end = text.index("\n)", start)
    values = np.array(
        text[start:end].replace("(", " ").replace(")", " ").split(), dtype=np.float64
    )
    return values.reshape(count, components).squeeze()


def grid_points(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates of the output grid and the obstacle mask."""
    h = DOMAIN / resolution
    centres = (np.arange(resolution) + 0.5) * h
    yy, xx = np.meshgrid(centres, centres, indexing="ij")  # row = y, column = x
    mask = (xx > OBSTACLE_X[0]) & (xx < OBSTACLE_X[1]) & (yy < OBSTACLE_H)
    return np.stack([xx.ravel(), yy.ravel()], axis=1), mask


def simulate(task: dict) -> tuple[int, np.ndarray, np.ndarray]:
    """Run one case and return its gridded alpha.water history and write times."""
    index = task["index"]
    with tempfile.TemporaryDirectory(dir=task["work_dir"]) as tmp:
        case = Path(tmp) / f"case_{index:04d}"
        prepare_case(
            Path(task["template"]),
            case,
            task["width"],
            task["height"],
            task["refine"],
            task["end_time"],
            task["write_interval"],
        )
        run(["blockMesh"], case)
        run(["setFields"], case)
        run(["interFoam"], case)

        centres = cell_centres(case / "system" / "blockMeshDict")
        # Consistency check: setFields marks exactly the cells whose centre lies in
        # the column, so the initial field must agree with the computed centres.
        alpha0 = read_internal_field(case / "0" / "alpha.water", len(centres))
        inside = (centres[:, 0] < task["width"]) & (centres[:, 1] < task["height"])
        if not np.array_equal(alpha0 > 0.5, inside):
            raise RuntimeError(f"cell centres do not match the mesh of {case}")
        points, mask = grid_points(task["resolution"])
        _, nearest = cKDTree(centres).query(points)

        times = sorted(
            (
                p
                for p in case.iterdir()
                if re.fullmatch(r"\d+(\.\d+)?(e-?\d+)?", p.name)
            ),
            key=lambda p: float(p.name),
        )
        frames = []
        for time_dir in times:
            alpha = read_internal_field(time_dir / "alpha.water", len(centres))
            frame = alpha[nearest].reshape(task["resolution"], task["resolution"])
            frame[mask] = 0.0
            frames.append(np.clip(frame, 0.0, 1.0))
        return (
            index,
            np.stack(frames).astype(np.float32),
            np.array([float(t.name) for t in times]),
        )


def cell_centres(block_mesh_dict: Path) -> np.ndarray:
    """Cell centres of a 2D blockMesh made of uniform, axis-aligned hex blocks.

    blockMesh numbers cells block by block, with the first block direction
    varying fastest, which is the order of the internalField of every field.
    """
    text = block_mesh_dict.read_text()
    scale = float(
        re.search(r"^(?:scale|convertToMeters)\s+([^;]+);", text, re.M).group(1)
    )
    vertex_block = re.search(r"vertices\s*\((.*?)\n\);", text, re.S).group(1)
    vertices = np.array(
        re.findall(r"\(([-\d.e]+) ([-\d.e]+) ([-\d.e]+)\)", vertex_block), dtype=float
    )
    centres = []
    for corners, nx, ny in re.findall(r"hex \(([\d ]+)\) \((\d+) (\d+) 1\)", text):
        ids = [int(c) for c in corners.split()]
        x0, y0 = vertices[ids[0], :2]
        x1, y1 = vertices[ids[2], :2]
        nx, ny = int(nx), int(ny)
        xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
        yy, xx = np.meshgrid(ys, xs, indexing="ij")  # x varies fastest
        centres.append(np.stack([xx.ravel(), yy.ravel()], axis=1))
    return np.concatenate(centres) * scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--num-cases", type=int, default=100)
    parser.add_argument("--output-dir", default="./data")
    parser.add_argument("--template", default=None, help="damBreak tutorial directory")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--refine", type=int, default=2, help="mesh refinement factor")
    parser.add_argument("--end-time", type=float, default=1.0)
    parser.add_argument("--write-interval", type=float, default=0.025)
    parser.add_argument("--width-range", type=float, nargs=2, default=(0.08, 0.20))
    parser.add_argument("--height-range", type=float, nargs=2, default=(0.15, 0.45))
    parser.add_argument("--splits", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    template = find_template(args.template)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    params = np.stack(
        [
            rng.uniform(*args.width_range, args.num_cases),
            rng.uniform(*args.height_range, args.num_cases),
        ],
        axis=1,
    )
    tasks = [
        {
            "index": i,
            "template": str(template),
            "work_dir": str(output_dir),
            "width": float(w),
            "height": float(h),
            "refine": args.refine,
            "end_time": args.end_time,
            "write_interval": args.write_interval,
            "resolution": args.resolution,
        }
        for i, (w, h) in enumerate(params)
    ]

    alpha = [None] * args.num_cases
    times = None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (index, frames, case_times) in enumerate(
            pool.map(simulate, tasks), start=1
        ):
            if times is None:
                times = case_times
            if len(case_times) != len(times) or not np.allclose(case_times, times):
                raise RuntimeError(f"case {index} wrote different times: {case_times}")
            alpha[index] = frames
            print(f"[{done}/{args.num_cases}] case {index} done", flush=True)

    _, mask = grid_points(args.resolution)
    alpha = np.stack(alpha)
    bounds = np.cumsum(np.round(np.array(args.splits) * args.num_cases)).astype(int)
    splits = {
        "train": slice(0, bounds[0]),
        "validation": slice(bounds[0], bounds[1]),
        "test": slice(bounds[1], args.num_cases),
    }
    for name, part in splits.items():
        with h5py.File(output_dir / f"dam_break_{name}.h5", "w") as f:
            f.create_dataset("alpha", data=alpha[part], compression="gzip")
            f.create_dataset("mask", data=mask.astype(np.float32))
            f.create_dataset("params", data=params[part])
            f.create_dataset("times", data=times)
        print(
            f"{name}: {alpha[part].shape[0]} cases -> {output_dir}/dam_break_{name}.h5"
        )


if __name__ == "__main__":
    main()
