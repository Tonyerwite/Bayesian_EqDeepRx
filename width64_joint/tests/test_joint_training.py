from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import math

import pytest
import torch

from bayesian_eqdeeprx import loss, model, training
from bayesian_eqdeeprx.config import BayesianEqDeepRxConfig
from bayesian_eqdeeprx.qam_signal import qam_modulate


MODULES = ("front_end.denoise.", "state_init.", "updates.", "demapper.", "log_sigma")
INDIVIDUAL_MODULES = ("front_end.denoise.", "state_init.", "updates.0.", "updates.1.",
                      "updates.2.", "updates.3.", "demapper.", "log_sigma")


def small_config():
    return replace(BayesianEqDeepRxConfig(), n_subcarriers=24, n_ofdm_symbols=4,
                   n_rx_antennas=4)


class SyntheticSystem:
    """Small physical linear channel for deterministic optimizer/resume tests."""

    def __init__(self, config):
        self.config = config

    def generate_batch(self, *, batch_size, n_layers, pilot_count, snr_db, seed,
                       add_interference, **kwargs):
        c = self.config
        g = torch.Generator().manual_seed(seed)
        n, k, f, s, r = batch_size, n_layers, c.n_subcarriers, c.n_ofdm_symbols, c.n_rx_antennas
        mask = torch.zeros(n, k, f, s)
        times = (1,) if pilot_count == 1 else (1, s - 1)
        for layer in range(k):
            mask[:, layer, layer::4, list(times)] = 1
        pilots = mask.to(torch.complex64) * complex(2**-0.5, 2**-0.5)
        data = torch.ones(n, k, f, s)
        data[..., list(times)] = 0
        bits = torch.randint(0, 2, (n, k, f, s, 6), generator=g).float()
        symbols = qam_modulate(bits, "64QAM") * data + pilots
        channel = (torch.randn(n, r, k, 1, 1, generator=g)
                   + 1j * torch.randn(n, r, k, 1, 1, generator=g)) / 2**0.5
        noise = 0.15 * (torch.randn(n, r, f, s, generator=g)
                        + 1j * torch.randn(n, r, f, s, generator=g))
        received = (channel * symbols[:, None]).sum(2) + noise
        if add_interference:
            received = received + 0.2 * torch.randn(n, 1, f, s, generator=g)
        return SimpleNamespace(received=received, pilot_symbols=pilots, pilot_mask=mask,
            data_mask=data, transmitted=symbols, target_bits=bits.permute(0, 1, 4, 2, 3),
            snr_db=snr_db, backend="synthetic_test_only")


def fresh(config=None):
    torch.manual_seed(13)
    return model.BayesianEqDeepRx(config or small_config(), message_chunk_size=64)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def test_random_initialization_and_optimizer_include_every_module():
    receiver = fresh()
    assert all(p.requires_grad for p in receiver.parameters()), "DenoiseNN must be trainable"
    optimizer = training.make_optimizer(receiver)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in receiver.parameters()}
    assert_tree_equal(receiver.state_dict(), fresh().state_dict())
    torch.manual_seed(14)
    other = model.BayesianEqDeepRx(small_config())
    assert any(not torch.equal(p, q) for p, q in zip(
        receiver.front_end.denoise.parameters(), other.front_end.denoise.parameters()))
    assert math.isclose(training.paper_learning_rate(34999, total_steps=70000,
        base_lr=0.0044), 0.0044 * (1 - 34999 / 69999))


@pytest.mark.parametrize("layers", (2, 3, 4))
@pytest.mark.parametrize("pilots", (1, 2))
@pytest.mark.parametrize("interference", (False, True))
def test_bit_symbol_loss_backpropagates_through_all_modules(layers, pilots, interference):
    receiver = fresh()
    c = receiver.config
    b = SyntheticSystem(c).generate_batch(batch_size=1, n_layers=layers,
        pilot_count=pilots, snr_db=torch.tensor([6.0]), seed=13,
        add_interference=interference)
    final, aux = receiver(b.received, b.pilot_symbols, b.pilot_mask, b.data_mask,
                          return_aux=True)
    objective = loss.receiver_loss(aux["readouts"], aux["soft_symbols"], b.target_bits,
                                   b.transmitted, b.data_mask, 4.0)
    objective.backward()
    assert len(aux["readouts"]) == len(aux["soft_symbols"]) == 4
    for name, p in receiver.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    for prefix in INDIVIDUAL_MODULES:
        assert any(p.grad.abs().sum() > 0 for name, p in receiver.named_parameters()
                   if name.startswith(prefix)), prefix
    assert torch.isfinite(final).all()


def test_loss_components_preserve_objective_and_each_frontend_gradient_path():
    assert hasattr(loss, "receiver_loss_components"), "Need inspectable bit/symbol components"
    receiver = fresh()
    b = SyntheticSystem(receiver.config).generate_batch(batch_size=1, n_layers=2,
        pilot_count=2, snr_db=torch.tensor([6.0]), seed=13, add_interference=False)
    _, aux = receiver(b.received, b.pilot_symbols, b.pilot_mask, b.data_mask, return_aux=True)
    args = (aux["readouts"], aux["soft_symbols"], b.target_bits, b.transmitted,
            b.data_mask, torch.tensor([4.0]))
    parts = loss.receiver_loss_components(*args)
    torch.testing.assert_close(parts["loss"], loss.receiver_loss(*args), rtol=0, atol=0)
    torch.testing.assert_close(parts["loss"], parts["bit_loss"] + parts["symbol_loss"])
    assert parts["round_bit_losses"].shape == (4,)
    parameters = list(receiver.front_end.denoise.parameters())
    for key in ("bit_loss", "symbol_loss"):
        gradients = torch.autograd.grad(parts[key], parameters, retain_graph=True)
        assert all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)


def test_real_receiver_checkpoint_resume_restores_trained_denoiser_and_all_state(tmp_path):
    c = small_config()
    system = SyntheticSystem(c)
    kwargs = dict(batch_size=2, generation_batch_size=1, model_microbatch_size=1,
                  n_layers=2, pilot_count=2, interference_present=False, device="cpu", seed=23)
    full, split = fresh(c), fresh(c)
    initial = copy.deepcopy(full.state_dict())
    full_path, split_path = tmp_path / "full.pt", tmp_path / "split.pt"
    training.train_steps(full, system, c, steps=2, output_path=full_path, **kwargs)
    training.train_steps(split, system, c, steps=1, output_path=split_path, **kwargs)
    resumed = fresh(c)
    training.train_steps(resumed, system, c, steps=2, output_path=split_path,
                         resume_path=split_path, **kwargs)
    a = torch.load(full_path, map_location="cpu", weights_only=True)
    b = torch.load(split_path, map_location="cpu", weights_only=True)
    assert a["checkpoint_schema_version"] == 2
    assert a["run_signature"]["training_mode"] == "full_joint"
    assert a["run_signature"]["initialization"] == "random"
    assert {x["step"] for x in a["optimizer_state_dict"]["state"].values()} == {2}
    assert_tree_equal(a, b)
    for prefix in INDIVIDUAL_MODULES:
        assert any(not torch.equal(initial[k], v) for k, v in a["model_state_dict"].items()
                   if k.startswith(prefix)), prefix
    for key in ("bit_losses", "symbol_losses", "round_bit_losses"):
        assert len(a["history"][key]) == 2


def test_protected_run_directories_are_rejected_before_writes(tmp_path):
    assert hasattr(training, "validate_run_directory"), "Need isolated output path validation"
    for path in ("D:/BayesianEqDeepRxRuns/width64_fast_batch112",
                 "D:/BayesianEqDeepRxRuns/width64_fast_batch112/nested",
                 "D:/BayesianEqDeepRxRuns/width64_batch112_match46776",
                 "D:/BayesianEqDeepRxRuns/batch8_70000"):
        with pytest.raises(ValueError, match="protected"):
            training.validate_run_directory(Path(path))
    training.validate_run_directory(tmp_path)


@pytest.mark.parametrize("corruption", ("frozen_schema", "wrong_initialization", "missing_optimizer", "nonfinite_history"))
def test_resume_rejects_incompatible_or_damaged_checkpoint(tmp_path, corruption):
    receiver = fresh()
    c = receiver.config
    path = tmp_path / "joint.pt"
    kwargs = dict(batch_size=1, generation_batch_size=1, n_layers=2, pilot_count=2,
                  interference_present=False, seed=23, device="cpu")
    training.train_steps(receiver, SyntheticSystem(c), c, steps=1, output_path=path, **kwargs)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if corruption == "frozen_schema":
        saved["checkpoint_schema_version"] = 1
    elif corruption == "wrong_initialization":
        saved["run_signature"]["initialization"] = "pretrained"
    elif corruption == "missing_optimizer":
        saved["optimizer_state_dict"]["state"].pop(next(iter(saved["optimizer_state_dict"]["state"])))
    else:
        saved["history"]["losses"][0] = float("nan")
    torch.save(saved, path)
    with pytest.raises((ValueError, RuntimeError)):
        training.train_steps(fresh(c), SyntheticSystem(c), c, steps=2,
                             resume_path=path, **kwargs)


def test_effective112_samples_are_one_synchronized_optimizer_update(tmp_path):
    class CountingSystem(SyntheticSystem):
        samples = 0

        def generate_batch(self, **kwargs):
            self.samples += kwargs["batch_size"]
            return super().generate_batch(**kwargs)

    receiver = fresh()
    system = CountingSystem(receiver.config)
    path = tmp_path / "batch112.pt"
    training.train_steps(receiver, system, receiver.config, steps=1, batch_size=112,
        generation_batch_size=2, model_microbatch_size=2, n_layers=2, pilot_count=2,
        interference_present=False, seed=23, output_path=path, device="cpu")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert system.samples == 112
    assert saved["next_step"] == 1
    assert {s["step"] for s in saved["optimizer_state_dict"]["state"].values()} == {1}


def test_initialization_does_not_load_any_pretrained_file(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Random joint initialization must not load a checkpoint")

    monkeypatch.setattr(torch, "load", forbidden)
    receiver = fresh()
    assert all(p.requires_grad for p in receiver.front_end.denoise.parameters())



