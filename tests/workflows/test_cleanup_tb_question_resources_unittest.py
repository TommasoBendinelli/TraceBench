from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "workflows" / "rollout" / "cleanup_tb_question_resources.sh"
WRAPPER_SCRIPT = REPO_ROOT / "workflows" / "rollout" / "cleanup_tb_question_resources"


def _fake_docker_bin(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    containers = tmp_path / "containers.txt"
    networks = tmp_path / "networks.txt"
    log = tmp_path / "docker.log"
    docker.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
case "${1:-} ${2:-}" in
  "ps -a")
    cat "$FAKE_DOCKER_CONTAINERS"
    ;;
  "network ls")
    cat "$FAKE_DOCKER_NETWORKS"
    ;;
  "rm -f")
    echo "container $3" >> "$FAKE_DOCKER_LOG"
    ;;
  "network rm")
    echo "network $3" >> "$FAKE_DOCKER_LOG"
    ;;
  *)
    echo "unexpected docker args: $*" >&2
    exit 9
    ;;
esac
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    log.write_text("", encoding="utf-8")
    return bin_dir, containers, networks, log


def _run_cleanup(
    tmp_path: Path,
    args: list[str],
    *,
    containers: str,
    networks: str,
    script: Path = SCRIPT,
) -> subprocess.CompletedProcess[str]:
    bin_dir, containers_path, networks_path, log_path = _fake_docker_bin(tmp_path)
    containers_path.write_text(containers, encoding="utf-8")
    networks_path.write_text(networks, encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
            "FAKE_DOCKER_CONTAINERS": str(containers_path),
            "FAKE_DOCKER_NETWORKS": str(networks_path),
            "FAKE_DOCKER_LOG": str(log_path),
        }
    )
    proc = subprocess.run(
        [str(script), *args],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    proc.fake_docker_log = log_path.read_text(encoding="utf-8")  # type: ignore[attr-defined]
    return proc


def test_cleanup_dry_run_lists_matches_without_removing(tmp_path: Path) -> None:
    proc = _run_cleanup(
        tmp_path,
        ["--dry-run", "--run-id", "run_a"],
        containers="tb-run_a-client\nunrelated\n",
        networks="tb-run_a-network\nother\n",
    )

    assert proc.returncode == 0, proc.stderr
    assert "dry_run kind=container name=tb-run_a-client" in proc.stdout
    assert "dry_run kind=network name=tb-run_a-network" in proc.stdout
    assert proc.fake_docker_log == ""  # type: ignore[attr-defined]


def test_cleanup_run_id_accepts_multiple_ids(tmp_path: Path) -> None:
    proc = _run_cleanup(
        tmp_path,
        ["--run-id", "run_a", "run_b"],
        containers="tb-run_a-client\ntb-run_b-client\ntb-run_c-client\n",
        networks="tb-run_b-network\ntb-run_c-network\n",
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.fake_docker_log == (  # type: ignore[attr-defined]
        "container tb-run_a-client\n"
        "container tb-run_b-client\n"
        "network tb-run_b-network\n"
    )


def test_cleanup_without_run_id_targets_documented_question_prefix(tmp_path: Path) -> None:
    proc = _run_cleanup(
        tmp_path,
        [],
        containers="question_0-1-of-1-run-client\nquestion_0.1-of-1-run\nother\n",
        networks="tb-question_0-1-of-1-run-network\nother\n",
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.fake_docker_log == (  # type: ignore[attr-defined]
        "container question_0-1-of-1-run-client\n"
        "network tb-question_0-1-of-1-run-network\n"
    )


def test_extensionless_cleanup_wrapper_matches_documented_command(tmp_path: Path) -> None:
    proc = _run_cleanup(
        tmp_path,
        ["--dry-run", "--run-id", "run_a"],
        containers="tb-run_a-client\n",
        networks="tb-run_a-network\n",
        script=WRAPPER_SCRIPT,
    )

    assert proc.returncode == 0, proc.stderr
    assert "dry_run kind=container name=tb-run_a-client" in proc.stdout
    assert "dry_run kind=network name=tb-run_a-network" in proc.stdout
