# Copyright (c) 2026 Adrian R. Minut
# Copyright (c) 2026 ABI Team
# SPDX-License-Identifier: GPL-3.0

import os
import torch
from scripts.eval_mcsamples import run_evaluation
from vlbench.train.trainutils import do_evalbatch, savecheckpoint, loadcheckpoint
from vloptimizers.ivon import IVON


class DummyModel(torch.nn.Module):
    def __init__(self, out_features=10):
        super().__init__()
        self.fc = torch.nn.Linear(5, out_features)

    def forward(self, x):
        return self.fc(x)


def test_do_evalbatch_with_optimizer():
    """Test that do_evalbatch can handle an optimizer with sampled_params."""
    model = DummyModel()
    optimizer = IVON(model.parameters(), lr=0.1, ess=100.0)

    batch_size = 4
    x = torch.randn(batch_size, 5)
    y = torch.randint(0, 10, (batch_size,))
    batchinput = (x, y)

    # Test with optimizer and repeat > 1 (should average)
    with torch.no_grad():
        prob, gt, loss = do_evalbatch(batchinput, model, optimizer=optimizer, repeat=4)

    assert prob.shape == (batch_size, 10)
    assert gt.shape == (batch_size,)
    assert isinstance(loss, float)


def test_run_evaluation_logic(tmp_path):
    """Test the iteration logic of run_evaluation."""
    model = DummyModel()
    optimizer = IVON(model.parameters(), lr=0.1, ess=100.0)

    # We need to mock do_epoch because our dataloader is simple
    save_dir = str(tmp_path / "results")

    # Run a simplified version of evaluation
    mc_samples_list = [1, 2]

    # Since run_evaluation calls do_epoch which expects a real dataloader and a coroutine
    # we just verify that it doesn't crash if we provide a minimal setup.
    # Actually, let's just test that it runs.

    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.randn(10, 5), torch.randint(0, 10, (10,)))
    loader = DataLoader(dataset, batch_size=5)

    import warnings
    from sklearn.exceptions import UndefinedMetricWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        run_evaluation(
            model,
            optimizer,
            loader,
            mc_samples_list,
            torch.device("cpu"),
            save_dir,
            prefix="test_eval",
        )

    assert os.path.exists(os.path.join(save_dir, "test_eval.csv"))


def test_load_checkpoint(tmp_path):
    """Test saving and loading a dummy checkpoint."""
    model = DummyModel()
    optimizer = IVON(model.parameters(), lr=0.1, ess=100.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    checkpoint_path = str(tmp_path / "checkpoint.pt")

    # Register DummyModel in vlbench.models._registry globals for loadmodel to find it
    import vlbench.models._registry as registry

    registry.dummy = lambda out: DummyModel(out)

    savecheckpoint(checkpoint_path, "dummy", (10,), {}, model, optimizer, scheduler)

    assert os.path.exists(checkpoint_path)

    # Load back
    startepoch, model_l, optimizer_l, scheduler_l, dic = loadcheckpoint(checkpoint_path)

    assert isinstance(model_l, DummyModel)
    assert isinstance(optimizer_l, IVON)
    assert startepoch == 0
