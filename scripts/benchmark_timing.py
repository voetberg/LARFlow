"""Point-model timing against the hit count N, with no weights and no data.

The cost of one event is set by the architecture and by its hit count, never
by the weight values: the per-point layers run on every one of the
`max_num_points` padded slots, while attention runs only on the
ceil(N / 128)^2 blocks of real hits.  So a randomly initialised model with
the run's architecture times the same as the trained one, provided the
events have the right N.  Without the global model or data there is no
realistic N per energy, so this sweeps N instead; map energy to N afterwards
(e.g. from Geant4 output) to read off a per-energy time.

The architecture, padding length and trafo layout come from a run conf in
`conf/`.  A temporary run directory with random weights and stand-in trafos is
written, and the timed code is `lardiff.generator.LArGenerator` itself.  The
global model is not timed: it is a small MLP over two numbers per event.

    python scripts/benchmark_timing.py [--out timing.md]
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
import tempfile
import time

import torch
import torch._dynamo
import yaml

from lardiff import flow_matching as fm
from lardiff import transformer
from lardiff.generator import LArGenerator
from lardiff.preprocessing import compose

logger = logging.getLogger("benchmark_timing")

REPO_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# conf/lar_muon_v4.yaml has the identical model and max_num_points
DEFAULT_CONF = os.path.join(REPO_DIRECTORY, "conf", "lar_electron_v4_cont.yaml")
DEFAULT_NUM_POINTS = [128, 256, 512, 1024, 2048, 4096, 8192]

# the generator's conditioning check reads log E_dep from the cond scaler, so
# the stand-in conditioning has to span real energies in MeV
STANDIN_ENERGY_RANGE_MEV = (10.0, 10_000.0)
STANDIN_ENERGY_MEV = 1_000.0


@dataclasses.dataclass
class SweepTiming:
    batch_size: int
    num_points: int
    num_events: int
    seconds_per_batch: list[float]

    @property
    def milliseconds_per_event(self) -> float:
        return 1e3 * sum(self.seconds_per_batch) / self.num_events


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


def write_random_run_directory(conf_path: str, run_directory: str, seed: int) -> dict:
    """Write conf.yaml, random weights/best.pt and stand-in preprocessing/trafos.pt."""
    with open(conf_path) as conf_file:
        run_conf = yaml.safe_load(conf_file)
    os.makedirs(os.path.join(run_directory, "weights"))
    os.makedirs(os.path.join(run_directory, "preprocessing"))
    with open(os.path.join(run_directory, "conf.yaml"), "w") as conf_file:
        yaml.safe_dump(run_conf, conf_file)

    torch.manual_seed(seed)
    network_parameters = {
        key: value for key, value in run_conf["model"].items() if key != "flow_config"
    }
    flow_parameters = run_conf["model"].get("flow_config", {})
    flow = fm.CNF(transformer.Transformer(**network_parameters), **flow_parameters)
    torch.save(flow.state_dict(), os.path.join(run_directory, "weights", "best.pt"))
    logger.debug(
        "random point model: %d parameters", sum(p.numel() for p in flow.parameters())
    )

    # the trafo values only shift and scale the output; fitting on synthetic
    # data gives buffers of the right shapes, which is all the timing needs
    data_conf = run_conf["data"]
    standin_points = torch.rand(16, 256, 4) + 0.1
    standin_mask = torch.ones(16, 256, 1, dtype=torch.bool)
    low_energy_mev, high_energy_mev = STANDIN_ENERGY_RANGE_MEV
    standin_cond = low_energy_mev + (high_energy_mev - low_energy_mev) * torch.rand(1024, 3)

    samples_coordinate_trafo = compose(data_conf.get("samples_coordinate_trafo"))
    samples_coordinate_trafo.fit(standin_points[:, :, :3], standin_mask)
    samples_energy_trafo = compose(data_conf.get("samples_energy_trafo"))
    samples_energy_trafo.fit(standin_points[:, :, 3], standin_mask.squeeze(-1))
    cond_trafo = compose(data_conf.get("cond_trafo"))
    cond_trafo.fit(standin_cond)
    torch.save(
        {
            "samples_coordinate_trafo": samples_coordinate_trafo.state_dict(),
            "samples_energy_trafo": samples_energy_trafo.state_dict(),
            "cond_trafo": cond_trafo.state_dict(),
        },
        os.path.join(run_directory, "preprocessing", "trafos.pt"),
    )
    return run_conf


def generate_batch(
    generator: LArGenerator,
    num_points: int,
    batch_size: int,
    device: torch.device,
    renormalize: bool,
) -> float:
    energies = torch.full((batch_size,), STANDIN_ENERGY_MEV, device=device)
    num_points_per_event = torch.full((batch_size,), num_points, dtype=torch.long, device=device)
    ratios = torch.ones(batch_size, device=device)
    needs_ratio = generator.cond_dim > 2 or renormalize

    synchronize(device)
    batch_start = time.perf_counter()
    samples = generator(
        energies,
        num_points_per_event,
        ratio=ratios if needs_ratio else None,
        renormalize=renormalize,
    ).cpu()
    synchronize(device)
    batch_seconds = time.perf_counter() - batch_start

    logger.debug(
        "batch %d, N %d: %.4f s, samples %s",
        batch_size, num_points, batch_seconds, tuple(samples.shape),
    )
    return batch_seconds


def time_batch_size(
    run_directory: str,
    batch_size: int,
    parsed_arguments: argparse.Namespace,
    device: torch.device,
) -> list[SweepTiming]:
    # each batch size is its own compiled shape; resetting stops dynamo from
    # switching to dynamic-shape kernels after it has seen a second size
    torch._dynamo.reset()
    generator = LArGenerator(
        run_dir=run_directory,
        num_timesteps=parsed_arguments.num_timesteps,
        compile=parsed_arguments.compile,
        solver=parsed_arguments.solver,
    ).to(device)
    generator.eval()

    # alternate between the extreme N, so the compiled graph has seen both
    # the sparsest and the densest block masks before timing starts
    for warmup_index in range(parsed_arguments.warmup_batches):
        warmup_num_points = parsed_arguments.num_points[-1 if warmup_index % 2 else 0]
        warmup_seconds = generate_batch(
            generator, warmup_num_points, batch_size, device, parsed_arguments.renormalize
        )
        logger.info(
            "batch %d: warm-up %d at N %d took %.2f s",
            batch_size, warmup_index, warmup_num_points, warmup_seconds,
        )

    num_batches = parsed_arguments.events_per_point // batch_size
    sweep_timings: list[SweepTiming] = []
    for num_points in parsed_arguments.num_points:
        seconds_per_batch = [
            generate_batch(generator, num_points, batch_size, device, parsed_arguments.renormalize)
            for _ in range(num_batches)
        ]
        sweep_timing = SweepTiming(
            batch_size=batch_size,
            num_points=num_points,
            num_events=parsed_arguments.events_per_point,
            seconds_per_batch=seconds_per_batch,
        )
        logger.info(
            "batch %d, N %d: %.3f ms/event",
            batch_size, num_points, sweep_timing.milliseconds_per_event,
        )
        sweep_timings.append(sweep_timing)

    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sweep_timings


def format_report(
    sweep_timings: list[SweepTiming],
    run_conf: dict,
    parsed_arguments: argparse.Namespace,
    device: torch.device,
) -> str:
    model_conf = run_conf["model"]
    compile_description = "torch.compile on" if parsed_arguments.compile else "torch.compile off"
    renormalize_description = "renormalized" if parsed_arguments.renormalize else "not renormalized"
    absolute_conf_path = os.path.abspath(parsed_arguments.conf)
    conf_description = (
        os.path.relpath(absolute_conf_path, REPO_DIRECTORY)
        if absolute_conf_path.startswith(REPO_DIRECTORY + os.sep)
        else absolute_conf_path
    )
    milliseconds_by_point = {
        (sweep_timing.batch_size, sweep_timing.num_points): sweep_timing.milliseconds_per_event
        for sweep_timing in sweep_timings
    }

    report_lines = [
        "LArFlow (lardiff) point-model timing against hit count",
        "",
        f"- **Code:** {describe_code()}",
        f"- **Software:** {describe_software()}",
        f"- **Machine:** {describe_machine(device)}",
        f"- **Setup:** architecture of {conf_description} "
        f"({model_conf['num_blocks']} blocks, {model_conf['num_head']} heads, embedding "
        f"{model_conf['dim_embedding']}, feedforward {model_conf['dim_feedforward']}, "
        f"padded to {run_conf['data']['max_num_points']} points), random weights; "
        f"{parsed_arguments.solver} solver, {parsed_arguments.num_timesteps} steps; "
        f"float32, matmul precision \"high\" (TF32 on Ampere+); {compile_description}; "
        f"{renormalize_description}; seed {parsed_arguments.seed}",
        f"- **Sample:** {parsed_arguments.events_per_point} events per (batch size, N) point, "
        "every event in a batch with the same N",
        "- **Timer:** wall clock (time.perf_counter, CUDA synchronized) around point-model "
        "sampling and the copy to host; excludes model building, "
        f"{parsed_arguments.warmup_batches} warm-up batches per batch size (torch.compile), "
        "and the global model",
        "",
        "Wall time per event (ms)",
        "",
        "| Batch size | " + " | ".join(f"N = {num_points}" for num_points in parsed_arguments.num_points) + " |",
        "|" + "|".join(["---"] + ["---:"] * len(parsed_arguments.num_points)) + "|",
    ]
    for batch_size in parsed_arguments.batch_sizes:
        row_cells = [str(batch_size)] + [
            f"{milliseconds_by_point[(batch_size, num_points)]:.3f}"
            for num_points in parsed_arguments.num_points
        ]
        report_lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(report_lines) + "\n"


def get_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--conf", default=DEFAULT_CONF,
        help="run conf whose architecture is timed. default: %(default)s",
    )
    parser.add_argument(
        "--num-points", type=int, nargs="+", default=DEFAULT_NUM_POINTS,
        help="hit counts to sweep, each at most the conf's max_num_points. default: %(default)s",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[100],
        help="one table row per batch size; each must divide --events-per-point. "
             "default: %(default)s",
    )
    parser.add_argument("-n", "--events-per-point", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--num-timesteps", type=int, default=200)
    parser.add_argument("--solver", default="heun")
    parser.add_argument(
        "--no-renormalize", dest="renormalize", action="store_false",
        help="skip the per-event energy rescaling (on by default, as in production)",
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

    for batch_size in parsed_arguments.batch_sizes:
        if parsed_arguments.events_per_point % batch_size != 0:
            raise SystemExit(
                f"batch size {batch_size} does not divide --events-per-point "
                f"{parsed_arguments.events_per_point}; a ragged last batch would "
                "recompile inside the timed region"
            )

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

    sweep_timings: list[SweepTiming] = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_directory = os.path.join(temporary_directory, "random_run")
        run_conf = write_random_run_directory(
            parsed_arguments.conf, run_directory, parsed_arguments.seed
        )
        max_num_points = run_conf["data"]["max_num_points"]
        if max(parsed_arguments.num_points) > max_num_points:
            raise SystemExit(
                f"--num-points goes above the conf's max_num_points {max_num_points}; "
                "the generator would clamp those events"
            )
        torch.manual_seed(parsed_arguments.seed)
        for batch_size in parsed_arguments.batch_sizes:
            sweep_timings += time_batch_size(run_directory, batch_size, parsed_arguments, device)

    report = format_report(sweep_timings, run_conf, parsed_arguments, device)
    print()
    print(report)

    report_path = parsed_arguments.out or (
        f"timing_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as report_file:
        report_file.write(report)

    json_path = os.path.splitext(report_path)[0] + ".json"
    with open(json_path, "w") as json_file:
        json.dump(
            {
                "arguments": vars(parsed_arguments),
                "code": describe_code(),
                "software": describe_software(),
                "machine": describe_machine(device),
                "points": [
                    dataclasses.asdict(sweep_timing)
                    | {"milliseconds_per_event": sweep_timing.milliseconds_per_event}
                    for sweep_timing in sweep_timings
                ],
            },
            json_file,
            indent=2,
        )
    logger.info("wrote %s and %s", report_path, json_path)


if __name__ == "__main__":
    main()
