import unittest
import numpy as np
import sklearn.metrics

from time_series_models.gap_runner import (
    _score,
    score_df_at_locs,
)


class TestGapRunnerFunctions(unittest.TestCase):
    def test_score(self):
        arr_t = np.ones(4) + np.ones(4) * 0.001
        arr_p = np.ones(4, dtype=float)
        scores_dict = _score(arr_t, arr_p)
        self.assertIsInstance(scores_dict, dict)
        [self.assertIsInstance(k, str) for k in scores_dict.keys()]
        [self.assertIsInstance(v, float) for v in scores_dict.values()]

        # TODO: eliminate hard-coding by making the metrics an enum
        #  and checking the list of keys in scores_dict with the list of enums
        self.assertEqual(len(scores_dict), 2)
        self.assertTrue("RMSE" in scores_dict)
        self.assertEqual(
            scores_dict["RMSE"],
            sklearn.metrics.mean_squared_error(arr_t, arr_p, squared=False),
        )
        self.assertTrue("median_absolute_error" in scores_dict)
        self.assertEqual(
            scores_dict["median_absolute_error"],
            sklearn.metrics.median_absolute_error(arr_t, arr_p),
        )

    def test_score_df_at_locs(self):
        pass  # TODO


if __name__ == "__main__":
    unittest.main()
