"""
Downsamples images in the given directory for the given number of times.
"""
import subprocess
import sys
from typing import Optional
from pathlib import Path, PurePath, PosixPath

from contextlib import nullcontext
from typing import Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
import os

CONSOLE = Console(width=120)


def status(msg: str, spinner: str = "bouncingBall", verbose: bool = False):
    """A context manager that does nothing is verbose is True. Otherwise it hides logs under a message.

    Args:
        msg: The message to log.
        spinner: The spinner to use.
        verbose: If True, print all logs, else hide them.
    """
    if verbose:
        return nullcontext()
    return CONSOLE.status(msg, spinner=spinner)


def get_progress(description: str, suffix: Optional[str] = None):
    """Helper function to return a rich Progress object."""
    progress_list = [TextColumn(description), BarColumn(), TaskProgressColumn(show_speed=True)]
    progress_list += [ItersPerSecColumn(suffix=suffix)] if suffix else []
    progress_list += [TimeRemainingColumn(elapsed_when_finished=True, compact=True)]
    progress = Progress(*progress_list)
    return progress


def run_command(cmd: str, verbose=False) -> Optional[str]:
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        CONSOLE.rule("[bold red] :skull: :skull: :skull: ERROR :skull: :skull: :skull: ", style="red")
        CONSOLE.print(f"[bold red]Error running command: {cmd}")
        CONSOLE.rule(style="red")
        CONSOLE.print(out.stderr.decode("utf-8"))
        sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out


def downscale_images(
    dir: Path,
    num_downscales: int,
    nearest_neighbor: bool = False,
    verbose: bool = False,
) -> str:

    if num_downscales == 0:
        return "No downscaling performed."

    with status(msg="[bold yellow]Downscaling images...", spinner="growVertical", verbose=verbose):
        downscale_factors = [2**i for i in range(num_downscales + 1)[1:]]
        for downscale_factor in downscale_factors:
            assert downscale_factor > 1
            assert isinstance(downscale_factor, int)
            downscale_dir = dir / f"{downscale_factor}x"
            downscale_dir.mkdir(parents=True, exist_ok=True)
            # Using %05d ffmpeg commands appears to be unreliable (skips images), so use scandir.
            files = os.scandir(dir / "1x")
            for f in files:
                if f.is_dir():
                    continue
                filename = f.name
                if not filename.endswith(".png"):
                    continue
                nn_flag = "" if not nearest_neighbor else ":flags=neighbor"
                ffmpeg_cmd = [
                    f'ffmpeg -y -noautorotate -i "{dir / "1x" / filename}" ',
                    f"-q:v 2 -vf scale=iw/{downscale_factor}:ih/{downscale_factor}{nn_flag} ",
                    f'"{downscale_dir / filename}"',
                ]
                ffmpeg_cmd = " ".join(ffmpeg_cmd)
                run_command(ffmpeg_cmd, verbose=verbose)

    CONSOLE.log("[bold green]:tada: Done downscaling images.")
    downscale_text = [f"[bold blue]{2**(i+1)}x[/bold blue]" for i in range(num_downscales)]
    downscale_text = ", ".join(downscale_text[:-1]) + " and " + downscale_text[-1]
    return f"We downsampled the images by {downscale_text}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing the images to downscale.",
    )
    parser.add_argument(
        "--num_downscales",
        type=int,
        default=2,
        help="Number of times to downscale the images.",
    )
    args = parser.parse_args()

    #

    downscale_images(args.dir, args.num_downscales)
