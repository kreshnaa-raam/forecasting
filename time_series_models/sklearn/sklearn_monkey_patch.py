import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

logger = logging.getLogger(__name__)


def apply_patches():
    """
    Patch Sci-Kit-Learn objects to fix feature names!
    """

    def pipeline_gfn(self):
        name, trans = self.steps[-1]
        if not hasattr(trans, "get_feature_names"):
            raise AttributeError(
                "Last Transformer %s (type %s) of the %s pipeline does not "
                "provide get_feature_names." % (self, str(name), type(trans).__name__)
            )

        return trans.get_feature_names()

    Pipeline.get_feature_names = pipeline_gfn

    def function_transformer_gfn(self):
        if hasattr(self.func, "get_feature_names"):
            return self.func.get_feature_names()
        else:
            raise AttributeError(
                "Function Transformer %s or its function %s must implement get_feature_names"
                % (self, self.func)
            )

    FunctionTransformer.get_feature_names = function_transformer_gfn
