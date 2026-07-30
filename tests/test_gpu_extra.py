import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_gpu_extra_declares_cupy():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    gpu = pyproject["project"]["optional-dependencies"]["gpu"]
    assert any(dep.startswith("cupy-cuda12x") for dep in gpu)


def test_importing_sampling_does_not_import_cupy():
    for mod in [m for m in sys.modules if m.startswith("cupy")]:
        del sys.modules[mod]
    import causal_bench.sampling.smc  # noqa: F401
    import causal_bench.sampling.backend  # noqa: F401
    assert not any(m.startswith("cupy") for m in sys.modules), \
        "cupy imported at module load — must stay lazy"
