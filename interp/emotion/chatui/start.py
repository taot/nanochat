from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
FRONTEND_DIR = HERE / "frontend"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = "8001"
FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = "5173"
HEALTHCHECK_TIMEOUT_SECONDS = 60.0
HEALTHCHECK_INTERVAL_SECONDS = 0.5

logging.basicConfig(
    level=os.environ.get("CHATUI_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("chatui.start")


def _run_checked(command: list[str], cwd: Path) -> None:
    logger.info("running: %s", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _ensure_npm_ready() -> None:
    if shutil.which("npm") is None:
        logger.error("npm is required. Please install npm first.")
        raise SystemExit(1)
    logger.info("installing frontend dependencies.")
    _run_checked(["npm", "install"], FRONTEND_DIR)


def _terminate(processes: list[subprocess.Popen[object]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5
    for process in processes:
        if process.poll() is not None:
            continue
        remaining = max(0.1, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in processes:
        if process.poll() is None:
            process.wait()


def _backend_command(backend_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "interp.emotion.chatui.server",
        "--host",
        BACKEND_HOST,
        "--port",
        BACKEND_PORT,
        *backend_args,
    ]


def _frontend_command() -> list[str]:
    return [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        FRONTEND_HOST,
        "--port",
        FRONTEND_PORT,
        "--clearScreen",
        "false",
    ]


def _wait_for_backend_health(backend_process: subprocess.Popen[object]) -> None:
    url = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/health"
    deadline = time.monotonic() + HEALTHCHECK_TIMEOUT_SECONDS
    logger.info("waiting for backend healthcheck: %s", url)
    while time.monotonic() < deadline:
        return_code = backend_process.poll()
        if return_code is not None:
            raise RuntimeError(f"backend exited before healthcheck succeeded with code {return_code}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    logger.info("backend healthcheck succeeded.")
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(HEALTHCHECK_INTERVAL_SECONDS)
    raise TimeoutError(f"backend healthcheck did not succeed within {HEALTHCHECK_TIMEOUT_SECONDS:g}s")


def _start_processes(backend_args: list[str]) -> list[subprocess.Popen[object]]:
    backend_command = _backend_command(backend_args)
    frontend_command = _frontend_command()
    logger.info("backend: %s", " ".join(backend_command))
    backend_process = subprocess.Popen(backend_command, cwd=HERE.parents[2])
    try:
        _wait_for_backend_health(backend_process)
    except Exception:
        _terminate([backend_process])
        raise

    logger.info("frontend: %s", " ".join(frontend_command))
    frontend_process = subprocess.Popen(frontend_command, cwd=FRONTEND_DIR)
    return [backend_process, frontend_process]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start chatui backend and frontend. Pass backend args after --."
    )
    parser.add_argument("backend_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    backend_args = args.backend_args[1:] if args.backend_args[:1] == ["--"] else args.backend_args

    try:
        _ensure_npm_ready()
    except subprocess.CalledProcessError as error:
        logger.error("npm install failed with code %d.", error.returncode)
        raise SystemExit(error.returncode) from error

    try:
        processes = _start_processes(backend_args)
    except (RuntimeError, TimeoutError) as error:
        logger.error("%s", error)
        raise SystemExit(1) from error

    try:
        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    logger.info("process exited with code %d; stopping.", return_code)
                    _terminate(processes)
                    raise SystemExit(return_code)
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("stopping.")
        _terminate(processes)


if __name__ == "__main__":
    main()
