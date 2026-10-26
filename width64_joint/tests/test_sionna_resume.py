import torch

from bayesian_eqdeeprx.config import BayesianEqDeepRxConfig
from bayesian_eqdeeprx.model import BayesianEqDeepRx
from bayesian_eqdeeprx.sionna_system import SionnaTR38901System
from bayesian_eqdeeprx.training import train_steps
from test_joint_training import assert_tree_equal


def test_full_grid_sionna_cpu_restart_is_bit_exact_including_denoiser(tmp_path):
    """Actual time-domain UMa generation, not the small synthetic fixture."""
    config = BayesianEqDeepRxConfig()
    kwargs = dict(batch_size=1, generation_batch_size=1, model_microbatch_size=1,
                  n_layers=2, pilot_count=1, interference_present=False,
                  seed=2026, device="cpu")

    def receiver():
        torch.manual_seed(2026)
        return BayesianEqDeepRx(config, message_chunk_size=128)

    full_path, split_path = tmp_path / "physical_full.pt", tmp_path / "physical_split.pt"
    full_system = SionnaTR38901System(config, "cpu")
    train_steps(receiver(), full_system, config, steps=2, output_path=full_path, **kwargs)
    del full_system
    first_system = SionnaTR38901System(config, "cpu")
    train_steps(receiver(), first_system, config, steps=1, output_path=split_path, **kwargs)
    del first_system
    restarted_system = SionnaTR38901System(config, "cpu")
    train_steps(receiver(), restarted_system, config, steps=2, output_path=split_path,
                resume_path=split_path, **kwargs)
    full = torch.load(full_path, map_location="cpu", weights_only=True)
    restarted = torch.load(split_path, map_location="cpu", weights_only=True)
    assert_tree_equal(full, restarted)

