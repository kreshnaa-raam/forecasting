import unittest
import numpy as np

from unittest.mock import patch

from time_series_models.transformers_shortcuts import (
    append_business_day_transformer,
    append_day_of_week_transformer,
    append_harmonic_transformers,
)


class AppendTransformerTests(unittest.TestCase):
    @patch(
        "time_series_models.transformers_shortcuts.is_business_day_pipeline",
        return_value="the pipeline",
    )
    def test_append_business_day_transformer(self, mock_business_day_pipeline):
        transformers_list = []
        append_business_day_transformer(transformers_list)
        # expect one transformer, a tuple of ("name", pipeline_call(*args, **kwargs), [0])
        self.assertEqual(1, len(transformers_list))
        self.assertIsInstance(transformers_list[0], tuple)
        self.assertEqual(len(transformers_list[0]), 3)
        self.assertEqual(transformers_list[0][0], "is_")
        self.assertEqual(transformers_list[0][1], "the pipeline")
        self.assertListEqual(transformers_list[0][2], [0])
        mock_business_day_pipeline.assert_called_once()

    @patch(
        "time_series_models.transformers_shortcuts.one_hot_encode_day_of_week_pipeline",
        return_value="the pipeline",
    )
    def test_append_day_of_week_transformer(self, mock_day_of_week_pipeline):
        transformers_list = []
        append_day_of_week_transformer(transformers_list)
        # expect one transformer, a tuple of ("name", pipeline_call(*args, **kwargs), [0])
        self.assertEqual(1, len(transformers_list))
        self.assertIsInstance(transformers_list[0], tuple)
        self.assertEqual(len(transformers_list[0]), 3)
        self.assertEqual(transformers_list[0][0], "day_of_week")
        self.assertEqual(transformers_list[0][1], "the pipeline")
        self.assertListEqual(transformers_list[0][2], [0])
        mock_day_of_week_pipeline.assert_called_once()
        mock_day_of_week_pipeline.assert_called_with(handle_unknown="error")

    @patch(
        "time_series_models.transformers_shortcuts.harmonic_transform_pipeline",
        return_value="the pipeline",
    )
    def test_append_harmonic_transformers(self, mock_harmonic_pipeline):
        transformers_list = []
        harmonics = [np.timedelta64(24, "h"), np.timedelta64(168, "h")]
        append_harmonic_transformers(transformers_list, harmonics)
        # expect two transformers, each a tuple of ("name", pipeline_call(*args, **kwargs), [0])
        self.assertEqual(2, len(transformers_list))

        self.assertIsInstance(transformers_list[0], tuple)
        self.assertEqual(len(transformers_list[0]), 3)
        self.assertEqual(transformers_list[0][0], "harmonic_24_hr")
        self.assertEqual(transformers_list[0][1], "the pipeline")
        self.assertListEqual(transformers_list[0][2], [0])

        self.assertIsInstance(transformers_list[1], tuple)
        self.assertEqual(len(transformers_list[1]), 3)
        self.assertEqual(transformers_list[1][0], "harmonic_168_hr")
        self.assertEqual(transformers_list[1][1], "the pipeline")
        self.assertListEqual(transformers_list[1][2], [0])

        self.assertEqual(mock_harmonic_pipeline.call_count, 2)
        mock_harmonic_pipeline.assert_has_calls(
            calls=[
                unittest.mock.call(
                    np.timedelta64(24, "h"), np.datetime64("1970-01-01")
                ),
                unittest.mock.call(
                    np.timedelta64(168, "h"), np.datetime64("1970-01-01")
                ),
            ]
        )


if __name__ == "__main__":
    unittest.main()
