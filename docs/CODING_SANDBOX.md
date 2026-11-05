# Coding-regression execution sandbox

`coding-v1` (and any later coding-regression suite) executes Python written by the model under
evaluation. That code is untrusted. This document describes the execution boundary, what it
guarantees, what it does not, and how to run security evaluation without it.

## What the previous implementation provided

Until runner version 2, generated code ran in a child `python -I` process with an empty environment,
a temporary working directory, `RLIMIT_CPU`/`RLIMIT_AS`/`RLIMIT_FSIZE` limits and discarded output.
That is resource containment only. The child ran as the evaluating user, so it could:

- read anything that user can read (`~/.aws/credentials`, `~/.cache/huggingface/token`, SSH keys,
  the repository, other runs' outputs);
- open network connections without restriction;
- write and persist files anywhere that user can write (only the size of each file was limited);
- start further processes (there was no process limit).

The empty environment only kept credentials held in environment variables away from the child.

## Current boundary (runner version 3)

Model-generated code only runs through an executor whose `isolated` flag is true, and the only such
executor is `sentinel.eval.sandbox.ContainerExecutor`. Each task gets a fresh container:

| Property | Setting (`configs/eval/coding_sandbox.yaml`) |
| --- | --- |
| Image | `python:3.12-slim@sha256:b18b261e…` pinned by digest, never pulled automatically (`--pull never`) |
| Network | `--network none` (loopback only) |
| Host file system | none, except the per-task directory (solution, hidden tests, runner) mounted **read-only** at `/work` |
| Root file system | `--read-only`; `/tmp` is a 16 MiB `noexec,nosuid,nodev` tmpfs |
| Identity | `--user 65534:65534` (nobody), `--cap-drop ALL`, `--security-opt no-new-privileges` |
| Credentials and environment | no `--env`/`--env-file`; the host environment is never passed in |
| Limits | 1 CPU, 1 GiB memory without swap, 64 processes, 1 MiB per file, 20 s wall clock per task |
| Output | stdout drained by the parent, only the last 4 KiB kept; stderr discarded |

**Fail closed.** Before the model is loaded, `run_model` calls the suite's preflight. It requires the
container runtime, a reachable daemon and the pinned image, then starts a probe container that must
observe exactly the required isolation: uid/gid 65534, only the `lo` interface, a read-only root
file system and task directory, no access to `/root`, and no mounts besides the runtime's own
pseudo file systems, `/tmp` and `/work`. Any failure raises `IsolationUnavailable` and the run stops
with exit code 3 and nothing written. A runtime error while starting a task container (exit status
125-127) also aborts the run instead of being counted as a failed task.

`LocalProcessExecutor` (the old child process) still exists for trusted, repository-authored code:
`python -m sentinel.foundation` uses it to check the reference solutions. `CodingV1.run` refuses it.

## What the boundary does not guarantee

- Containers share the host kernel. A kernel or container-runtime vulnerability could allow an
  escape. For untrusted models, run evaluations on a disposable host or VM that holds no valuable
  credentials, in addition to this sandbox.
- Whoever can talk to the Docker daemon is effectively root on the host. The sandbox protects the
  host from the generated code, not from the evaluating user.
- The completion marker (sent on stdin, printed after the hidden tests) stops accidental early exits
  such as `sys.exit(0)` from passing. It is not designed against code that deliberately searches the
  runner's memory for the marker; coding-v1 measures regressions, it is not an adversarial benchmark.
- CPU time inside the container is bounded by the wall-clock timeout and the CPU quota, not by
  `RLIMIT_CPU`.

## Running

Prepare an evaluation host once:

```bash
docker pull python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
```

Then run the bake-off or a direct evaluation with `--coding-suite coding-v1` as before. Podman works
with `runtime: podman` in the policy.

To evaluate security only (no container needed, nothing executed):

```bash
python -m sentinel.eval.run_model --bakeoff configs/model/bakeoff_qwen35.yaml --candidate qwen35-9b \
    --skip-coding-regression --output runs/qwen35-9b-security-only.json
```

The artifact records `coding_regression: {"status": "not_run", "reason": "coding-v1 skipped with
--skip-coding-regression ..."}`. `sentinel.eval.compare` refuses to compare a run with coding
regression against one without it, and runs under different sandbox policies (the policy is part of
`contract_sha256`).

## Tests

`tests/test_coding_regression.py` covers the policy validation, the fail-closed preflight (missing
runtime, daemon or image; a probe that sees a network interface, root, a writable file system or an
extra host mount), the refusal of non-isolated executors before any generation, runtime start
failures, and `--skip-coding-regression`. When a Docker daemon and the pinned image are available
(as in CI), the same tests also run against real containers, including a check that generated code
cannot open a network connection, see a host file or read the host environment, and does not run
as root.
