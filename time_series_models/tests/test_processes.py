import os
import unittest
from abc import ABC, abstractmethod

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class ProcessTest(ABC):
    class TestHarness:
        def __init__(self, time_step, *args, **kwargs):
            self.tstep = time_step
            self._args = args
            self.__kwargs = kwargs

    @abstractmethod
    def test_get_range(self):
        """
        Test should mock the remote fetch operation and assert correctness of the resulting data structure
        """

    @abstractmethod
    def test_make_preprocessor(self):
        """
        Test construction of the column transformer for all possible feature types
        """

    @abstractmethod
    def test_run_preprocessor(self):
        """
        Integration test for execution of the preprocessor column transformer.
        """

    @abstractmethod
    def test_feature_names(self):
        """
        Integration test for execution of the preprocessor column transformer.
        """


@unittest.skip("Not implemented yet!")
class AmiHourlyForecastTest(unittest.TestCase):
    pass


@unittest.skip("Not implemented yet!")
class BalancingAreaHourlyTest(unittest.TestCase):
    pass


@unittest.skip("Add test for model building with PV models")
class PVPySamForecastProcessTest(unittest.TestCase):
    def test_model_construction(self):
        self.fail("Not implemented")


if __name__ == "__main__":
    unittest.main()
