import unittest

from time import sleep
import multiprocessing as mp
from joblib.externals import loky
from joblib import Parallel, delayed, parallel_backend

from time_series_models.joblib import joblib_monkey_patch

# Store the initialization status in a global variable of a module.
loky._INITIALIZER_STATUS = "uninitialized"


class JobLibPatchTest(unittest.TestCase):
    @staticmethod
    def initializer(x):
        # Use print not logger for this test
        print(f"[{mp.current_process().name}] initializer")
        loky._INITIALIZER_STATUS = x

    @staticmethod
    def return_initializer_status(delay=0):
        sleep(delay)
        print(f"[{mp.current_process().name}] return_initializer_status")
        return getattr(loky, "_INITIALIZER_STATUS", "uninitialized")

    def setUp(self) -> None:
        joblib_monkey_patch.parallel_patch(
            initializer=self.initializer, initargs=("initialized",)
        )

    def test_parallel(self):
        with parallel_backend("loky", inner_max_num_threads=2):
            par = Parallel(n_jobs=6, verbose=10)
            results = par(delayed(self.return_initializer_status)() for x in range(8))

        for x in results:
            self.assertEqual(x, "initialized")


if __name__ == "__main__":
    unittest.main()
