"""Docker sandbox for safely running untrusted code during debugging."""

import os
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.errors import ContainerError, ImageNotFound


SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "autodebug-sandbox:latest")
TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "60"))


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def output(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()


class Sandbox:
    """Runs code inside an isolated Docker container.

    Each call to run() spins up a fresh container, copies the repo and
    any extra files in, executes the command, and returns output.
    All containers are removed after execution regardless of outcome.
    """

    def __init__(self, repo_path: str, image: str = SANDBOX_IMAGE):
        self.repo_path = Path(repo_path)
        self.image = image
        self._client = docker.from_env()
        self._ensure_image()

    def _ensure_image(self) -> None:
        try:
            self._client.images.get(self.image)
        except ImageNotFound:
            raise RuntimeError(
                f"Sandbox image '{self.image}' not found. "
                "Run: docker build -t autodebug-sandbox:latest ./docker/sandbox"
            )

    def run(
        self,
        command: str,
        extra_files: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = TIMEOUT,
    ) -> RunResult:
        """Execute command inside a container with the repo mounted read-only.

        Args:
            command: Shell command to run inside the container.
            extra_files: Mapping of filename → content for files to inject
                         into /workspace before running (e.g. repro scripts).
            env: Additional environment variables.
            timeout: Max seconds before the container is killed.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            if extra_files:
                for filename, content in extra_files.items():
                    (tmp_path / filename).write_text(content, encoding="utf-8")

            volumes = {
                str(self.repo_path): {"bind": "/repo", "mode": "ro"},
                str(tmp_path): {"bind": "/workspace", "mode": "rw"},
            }

            full_command = textwrap.dedent(f"""
                set -e
                cp -r /repo/. /workspace/repo
                cd /workspace/repo
                {command}
            """).strip()

            try:
                container = self._client.containers.run(
                    image=self.image,
                    command=["bash", "-c", full_command],
                    volumes=volumes,
                    environment=env or {},
                    working_dir="/workspace/repo",
                    mem_limit="512m",
                    network_disabled=False,  # needed for pip installs
                    remove=False,
                    detach=True,
                )

                try:
                    result = container.wait(timeout=timeout)
                    stdout = container.logs(stdout=True, stderr=False).decode(errors="replace")
                    stderr = container.logs(stdout=False, stderr=True).decode(errors="replace")
                    exit_code = result["StatusCode"]
                finally:
                    container.remove(force=True)

            except ContainerError as e:
                return RunResult(exit_code=1, stdout="", stderr=str(e))

            return RunResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    def run_script(self, script: str, **kwargs) -> RunResult:
        """Inject script.py into workspace and run it."""
        return self.run(
            command="python /workspace/script.py",
            extra_files={"script.py": script},
            **kwargs,
        )
