"""Per-event generation timing, in the layout of the ToyG4 timing benchmark.

For every (particle, kinetic energy) point, draws the hit count N and response
R from the global model and generates events at that fixed energy with the
point model, exactly as `lardiff.generator --n-source global` does.  No cache
is read: the conditioning is the fixed energy alone.

Only sampling is timed.  Model loading and the warm-up batches that trigger
torch.compile are excluded, as the Geant4 run timer excludes initialisation.

    python scripts/benchmark_timing.py [--out timing.md] [--batch-size 100]

Rows default to the two trained point models.  Any other species the global
model knows can be timed through an existing point model, which is a timing
proxy only (the hits are drawn from that model's species):

    --point-model=-11=results/20260818_121435_LAr-electron-CNF-v4-cont
"""

import argparse
import dataclasses
import datetime
import json
import logging
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch import Tensor

from lardiff.generator import LArGenerator
from lardiff.global_model import PDG_CODES, GlobalSampler

logger = logging.getLogger("benchmark_timing")

REPO_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_POINT_MODELS = [
    "11=results/20260818_121435_LAr-electron-CNF-v4-cont",
    "13=results/20260908_211332_LAr-muon-CNF-v4",
]
DEFAULT_GLOBAL_MODEL = "results/global_all_species_v5"
DEFAULT_ENERGIES_GEV = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

PARTICLE_LABELS = {
    11: "e⁻ (11)",
    -11: "e⁺ (−11)",
    22: "γ (22)",
    13: "μ⁻ (13)",
    -13: "μ⁺ (−13)",
    211: "π⁺ (211)",
    -211: "π⁻ (−211)",
    2212: "p (2212)",
    2112: "n (2112)",
}


@dataclasses.dataclass
class PointTiming:
    pdg: int
    energy_gev: float
    num_events: int
    batch_size: int
    global_seconds_per_batch: list[float]
    point_seconds_per_batch: list[float]
    mean_num_points: float

    @property
    def total_seconds(self) -> float:
        return sum(self.global_seconds_per_batch) + sum(self.point_seconds_per_batch)

    @property
    def milliseconds_per_event(self) -> float:
        return 1e3 * self.total_seconds / self.num_events

    @property
    def global_milliseconds_per_event(self) -> float:
        return 1e3 * sum(self.global_seconds_per_batch) / self.num_events


def parse_point_models(point_model_entries: list[str]) -> dict[int, str]:
    point_model_by_pdg: dict[int, str] = {}
    for entry in point_model_entries:
        pdg_text, separator, run_directory = entry.partition("=")
        if not separator or not run_directory:
            raise SystemExit(f"--point-model expects PDG=RUN_DIR, got {entry!r}")
        pdg = int(pdg_text)
        if pdg not in PDG_CODES:
            raise SystemExit(f"PDG {pdg} is not covered by the global model {PDG_CODES}")
        if not os.path.isdir(run_directory):
            raise SystemExit(f"no such run directory: {run_directory}")
        point_model_by_pdg[pdg] = run_directory
    return point_model_by_pdg


def run_git(arguments: list[str]) -> str:
    try:
        completed_process = subprocess.run(
            ["git", "-C", REPO_DIRECTORY, *arguments],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed_process.stdout.strip()


def describe_code() -> str:
    remote_url = run_git(["remote", "get-url", "origin"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = run_git(["rev-parse", "--short", "HEAD"])
    dirty_suffix = "-dirty" if run_git(["status", "--porcelain", "--untracked-files=no"]) else ""
    return f"{remote_url}, branch {branch}, commit {commit}{dirty_suffix}"


def describe_cpu() -> str:
    try:
        with open("/proc/cpuinfo") as cpuinfo_file:
            for line in cpuinfo_file:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def describe_software() -> str:
    software_parts = [f"Python {platform.python_version()}", f"torch {torch.__version__}"]
    if torch.version.cuda is not None:
        software_parts.append(f"CUDA {torch.version.cuda}")
    if torch.backends.cudnn.is_available():
        software_parts.append(f"cuDNN {torch.backends.cudnn.version()}")
    return ", ".join(software_parts)


def describe_machine(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(device)} (1 GPU), host CPU {describe_cpu()}"
    return f"{describe_cpu()}, {torch.get_num_threads()} torch threads"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate_batch(
    generator: LArGenerator,
    global_sampler: GlobalSampler,
    energy_mev: float,
    pdg: int,
    batch_size: int,
    device: torch.device,
    renormalize: bool,
) -> tuple[float, float, Tensor]:
    """Return (global seconds, point seconds, conditioned hit counts) for one batch."""
    energies = torch.full((batch_size,), energy_mev, dtype=torch.float32)
    needs_ratio = generator.cond_dim > 2 or renormalize

    synchronize(device)
    global_start = time.perf_counter()
    num_points, ratios = global_sampler.sample(
        energies, pdg=pdg, max_num_points=generator.max_points
    )
    synchronize(device)
    point_start = time.perf_counter()
    samples = generator(
        energies.to(device),
        num_points.to(device),
        ratio=ratios.to(device) if needs_ratio else None,
        renormalize=renormalize,
    ).cpu()
    synchronize(device)
    point_end = time.perf_counter()

    logger.debug(
        "pdg %d, %g MeV: global %.4f s, point %.4f s, mean N %.1f, samples %s",
        pdg, energy_mev, point_start - global_start, point_end - point_start,
        num_points.float().mean().item(), tuple(samples.shape),
    )
    return point_start - global_start, point_end - point_start, num_points


def time_species(
    pdg: int,
    run_directory: str,
    global_sampler: GlobalSampler,
    parsed_arguments: argparse.Namespace,
    device: torch.device,
) -> tuple[list[PointTiming], int]:
    """Return the timings at every energy and the model's padded event length."""
    logger.info("pdg %d: loading point model %s", pdg, run_directory)
    generator = LArGenerator(
        run_dir=run_directory,
        num_timesteps=parsed_arguments.num_timesteps,
        compile=parsed_arguments.compile,
        solver=parsed_arguments.solver,
    ).to(device)
    generator.eval()
    if generator.num_particles > 1:
        raise SystemExit(
            f"{run_directory} is a multi-species model; its particle labels come "
            "from the cache's species list, which this benchmark does not read"
        )

    # alternate between the extreme energies, so the compiled graph has seen
    # both the sparsest and the densest block masks before timing starts
    for warmup_index in range(parsed_arguments.warmup_batches):
        warmup_energy_gev = parsed_arguments.energies[-1 if warmup_index % 2 else 0]
        logger.info("pdg %d: warm-up batch %d at %g GeV", pdg, warmup_index, warmup_energy_gev)
        warmup_global_seconds, warmup_point_seconds, _ = generate_batch(
            generator, global_sampler, 1e3 * warmup_energy_gev, pdg,
            parsed_arguments.batch_size, device, parsed_arguments.renormalize,
        )
        logger.debug(
            "pdg %d: warm-up batch %d took %.2f s",
            pdg, warmup_index, warmup_global_seconds + warmup_point_seconds,
        )

    num_batches = parsed_arguments.events_per_point // parsed_arguments.batch_size
    point_timings: list[PointTiming] = []
    for energy_gev in parsed_arguments.energies:
        global_seconds_per_batch: list[float] = []
        point_seconds_per_batch: list[float] = []
        conditioned_counts: list[Tensor] = []
        for _ in range(num_batches):
            global_seconds, point_seconds, num_points = generate_batch(
                generator, global_sampler, 1e3 * energy_gev, pdg,
                parsed_arguments.batch_size, device, parsed_arguments.renormalize,
            )
            global_seconds_per_batch.append(global_seconds)
            point_seconds_per_batch.append(point_seconds)
            conditioned_counts.append(num_points)

        point_timing = PointTiming(
            pdg=pdg,
            energy_gev=energy_gev,
            num_events=parsed_arguments.events_per_point,
            batch_size=parsed_arguments.batch_size,
            global_seconds_per_batch=global_seconds_per_batch,
            point_seconds_per_batch=point_seconds_per_batch,
            mean_num_points=torch.cat(conditioned_counts).float().mean().item(),
        )
        logger.info(
            "pdg %d, %g GeV: %.3f ms/event (global model %.3f ms/event), mean N %.1f",
            pdg, energy_gev, point_timing.milliseconds_per_event,
            point_timing.global_milliseconds_per_event, point_timing.mean_num_points,
        )
        point_timings.append(point_timing)

    max_points = generator.max_points
    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return point_timings, max_points


def format_table(
    title: str,
    point_timings: list[PointTiming],
    pdg_order: list[int],
    energies_gev: list[float],
    value_format: str,
    value_name: str,
) -> list[str]:
    value_by_point = {
        (point_timing.pdg, point_timing.energy_gev): getattr(point_timing, value_name)
        for point_timing in point_timings
    }
    header_cells = ["Particle"] + [f"{energy_gev:g} GeV" for energy_gev in energies_gev]
    table_lines = [
        title,
        "",
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(["---"] + ["---:"] * len(energies_gev)) + "|",
    ]
    for pdg in pdg_order:
        row_cells = [PARTICLE_LABELS.get(pdg, str(pdg))] + [
            format(value_by_point[(pdg, energy_gev)], value_format)
            for energy_gev in energies_gev
        ]
        table_lines.append("| " + " | ".join(row_cells) + " |")
    return table_lines


def format_report(
    point_timings: list[PointTiming],
    point_model_by_pdg: dict[int, str],
    parsed_arguments: argparse.Namespace,
    device: torch.device,
    max_points_by_pdg: dict[int, int],
) -> str:
    pdg_order = list(point_model_by_pdg)
    point_model_description = "; ".join(
        f"{PARTICLE_LABELS.get(pdg, str(pdg))} ← {os.path.basename(os.path.normpath(run_directory))}"
        f" (max {max_points_by_pdg[pdg]} points)"
        for pdg, run_directory in point_model_by_pdg.items()
    )
    compile_description = "torch.compile on" if parsed_arguments.compile else "torch.compile off"
    renormalize_description = "renormalized to drawn R" if parsed_arguments.renormalize else "not renormalized"

    report_lines = [
        "LArFlow (lardiff) per-event timing benchmark",
        "",
        f"- **Code:** {describe_code()}",
        f"- **Software:** {describe_software()}",
        f"- **Machine:** {describe_machine(device)}",
        f"- **Setup:** point models {point_model_description}; global model "
        f"{os.path.basename(os.path.normpath(parsed_arguments.global_model))}; "
        f"{parsed_arguments.solver} solver, {parsed_arguments.num_timesteps} steps; "
        f"float32, matmul precision \"high\" (TF32 on Ampere+); {compile_description}; "
        f"{renormalize_description}; "
        f"seed {parsed_arguments.seed}",
        f"- **Sample:** {parsed_arguments.events_per_point} events per (particle, energy) "
        f"point, fixed kinetic energy, batch size {parsed_arguments.batch_size}, "
        "N and R drawn from the global model",
        "- **Timer:** wall clock (time.perf_counter, CUDA synchronized) around global-model "
        "sampling, point-model sampling and the copy to host; excludes model loading and "
        f"{parsed_arguments.warmup_batches} warm-up batches per particle (torch.compile)",
        "",
    ]
    report_lines += format_table(
        "Wall time per event (ms)", point_timings, pdg_order,
        parsed_arguments.energies, ".3f", "milliseconds_per_event",
    )
    report_lines.append("")
    report_lines += format_table(
        "of which global model (ms)", point_timings, pdg_order,
        parsed_arguments.energies, ".3f", "global_milliseconds_per_event",
    )
    report_lines.append("")
    report_lines += format_table(
        "Mean hits per event (N from the global model)", point_timings, pdg_order,
        parsed_arguments.energies, ".1f", "mean_num_points",
    )
    return "\n".join(report_lines) + "\n"


def get_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--point-model", action="append", default=None, metavar="PDG=RUN_DIR",
        help="point model to time for a species; repeat for each row. Write "
             "negative codes as --point-model=-11=RUN_DIR. default: "
             + ", ".join(DEFAULT_POINT_MODELS),
    )
    parser.add_argument("--global-model", default=DEFAULT_GLOBAL_MODEL)
    parser.add_argument(
        "--energies", type=float, nargs="+", default=DEFAULT_ENERGIES_GEV,
        help="kinetic energies in GeV. default: %(default)s",
    )
    parser.add_argument("-n", "--events-per-point", type=int, default=100)
    parser.add_argument(
        "-b", "--batch-size", type=int, default=100,
        help="events per generator call; must divide --events-per-point so "
             "every batch has the compiled shape. default: 100",
    )
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--num-timesteps", type=int, default=200)
    parser.add_argument("--solver", default="heun")
    parser.add_argument(
        "--no-renormalize", dest="renormalize", action="store_false",
        help="skip rescaling each event to the drawn R (on by default, as in "
             "the production generation commands)",
    )
    parser.add_argument(
        "--no-compile", dest="compile", action="store_false",
        help="disable torch.compile; it is on by default on GPU, as in lardiff.generator",
    )
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", default=None,
        help="markdown report path; a .json with per-batch timings is written "
             "beside it. default: timing_<timestamp>.md in the working directory",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args(arguments)


@torch.inference_mode()
def main(arguments: list[str] | None = None) -> None:
    parsed_arguments = get_arguments(arguments)
    logging.basicConfig(
        level=parsed_arguments.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if parsed_arguments.events_per_point % parsed_arguments.batch_size != 0:
        raise SystemExit(
            f"--batch-size {parsed_arguments.batch_size} does not divide "
            f"--events-per-point {parsed_arguments.events_per_point}; a ragged "
            "last batch would recompile inside the timed region"
        )
    point_model_by_pdg = parse_point_models(parsed_arguments.point_model or DEFAULT_POINT_MODELS)
    if not os.path.isdir(parsed_arguments.global_model):
        raise SystemExit(f"no such global model directory: {parsed_arguments.global_model}")

    if parsed_arguments.device:
        device = torch.device(parsed_arguments.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type != "cuda":
        logger.warning("running on %s; torch.compile disabled", device)
        parsed_arguments.compile = False
    logger.info("device: %s", device)
    logger.debug("arguments: %s", vars(parsed_arguments))

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(parsed_arguments.seed)
    np.random.seed(parsed_arguments.seed)

    logger.info("loading global model %s", parsed_arguments.global_model)
    global_sampler = GlobalSampler(parsed_arguments.global_model, device=device)

    point_timings: list[PointTiming] = []
    max_points_by_pdg: dict[int, int] = {}
    for pdg, run_directory in point_model_by_pdg.items():
        species_timings, max_points_by_pdg[pdg] = time_species(
            pdg, run_directory, global_sampler, parsed_arguments, device
        )
        point_timings += species_timings

    report = format_report(
        point_timings, point_model_by_pdg, parsed_arguments, device, max_points_by_pdg
    )
    print()
    print(report)

    report_path = parsed_arguments.out or (
        f"timing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    report_directory = os.path.dirname(os.path.abspath(report_path))
    os.makedirs(report_directory, exist_ok=True)
    with open(report_path, "w") as report_file:
        report_file.write(report)

    json_path = os.path.splitext(report_path)[0] + ".json"
    with open(json_path, "w") as json_file:
        json.dump(
            {
                "arguments": vars(parsed_arguments),
                "point_models": {str(pdg): path for pdg, path in point_model_by_pdg.items()},
                "code": describe_code(),
                "software": describe_software(),
                "machine": describe_machine(device),
                "points": [
                    dataclasses.asdict(point_timing)
                    | {"milliseconds_per_event": point_timing.milliseconds_per_event}
                    for point_timing in point_timings
                ],
            },
            json_file,
            indent=2,
        )
    logger.info("wrote %s and %s", report_path, json_path)


if __name__ == "__main__":
    main()
