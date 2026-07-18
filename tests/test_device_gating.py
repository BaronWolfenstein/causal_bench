import sys
import types


def test_cuda_visible_devices_parsed_in_order(monkeypatch):
    from causal_bench.sampling.device import cuda_available_devices
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1,2")
    assert cuda_available_devices() == [3, 1, 2]


def test_empty_cuda_visible_devices_means_cpu_box(monkeypatch):
    from causal_bench.sampling.device import cuda_available_devices
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert cuda_available_devices() == []


def test_falls_back_to_torch_device_count(monkeypatch):
    from causal_bench.sampling.device import cuda_available_devices
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(device_count=lambda: 4)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert cuda_available_devices() == [0, 1, 2, 3]


def test_resolve_device_passthrough():
    from causal_bench.sampling.device import resolve_device
    assert resolve_device("cpu") == "cpu"
