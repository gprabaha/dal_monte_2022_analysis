"""Regression tests for legacy behavioral pickle module remapping."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from dal_monte_2022_analysis.data.records.behavioral import (
        BehaviorRunContext,
        NeuralTimelineData,
    )
    from dal_monte_2022_analysis.utils.io import load_pickle

    _HAS_BEHAVIORAL_RECORDS = True
except ModuleNotFoundError:
    _HAS_BEHAVIORAL_RECORDS = False


@unittest.skipUnless(_HAS_BEHAVIORAL_RECORDS, "behavioral record modules are required for these tests")
class TestLegacyBehavioralPickleModuleRemap(unittest.TestCase):
    """Checks that legacy `behavioral_data` pickle module paths still load."""

    def test_load_pickle_remaps_behavioral_data_module_to_canonical_behavioral_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "timeline.pkl"

            context_module = BehaviorRunContext.__module__
            timeline_module = NeuralTimelineData.__module__
            try:
                BehaviorRunContext.__module__ = "dal_monte_2022_analysis.data.behavioral_data"
                NeuralTimelineData.__module__ = "dal_monte_2022_analysis.data.behavioral_data"
                obj = NeuralTimelineData(
                    context=BehaviorRunContext(
                        date="01012020",
                        session="1",
                        agent="m1",
                        monkey_name="test_monkey",
                    ),
                    t=np.asarray([0.0, 0.1, 0.2], dtype=float),
                )
                with path.open("wb") as f:
                    pickle.dump(obj, f)
            finally:
                BehaviorRunContext.__module__ = context_module
                NeuralTimelineData.__module__ = timeline_module

            loaded = load_pickle(path)
            self.assertIsInstance(loaded, NeuralTimelineData)
            self.assertIsInstance(loaded.context, BehaviorRunContext)
            self.assertTrue(np.allclose(np.asarray(loaded.t, dtype=float), np.asarray([0.0, 0.1, 0.2], dtype=float)))
            self.assertEqual(str(loaded.context.agent), "m1")


if __name__ == "__main__":
    unittest.main()
