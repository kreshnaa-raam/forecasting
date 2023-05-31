import unittest
import contextlib

from time_series_models import decorators


class EscapeContextProperyTestCase(unittest.TestCase):
    @decorators.escape_context_property("tprop")
    @contextlib.contextmanager
    def faux_context(self):
        yield dict(magic=True)

    def test_escape_context_property(self):
        # The normal way
        with self.faux_context() as faux:
            self.assertDictEqual(faux, dict(magic=True))

        # The magic way for use in jupyter notebook
        self.assertDictEqual(self.faux_context().tprop, dict(magic=True))


if __name__ == "__main__":
    unittest.main()
