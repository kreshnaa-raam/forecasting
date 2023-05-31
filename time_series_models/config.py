import json
import re
import logging
import typing
import types

import numpy as np
from time_series_models import pv_physical_model
from time_series_models.data_fetchers import numba_groupby_functions

CAMUS_JSON_TYPE = "__camus_json_type__"
CAMUS_JSON_DATA = "__camus_json_data__"

logger = logging.getLogger(__name__)


class _ConfigEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        Encode obj to dict, structured in format:
        {
            "__camus_json_type__": type_of_object,
            "__camus_json_data__": serialized_object,
        }
        """
        logger.debug("encoding: %s", type(obj))
        match type(obj):
            case np.ndarray:
                return self._encode_ndarray(obj)
            case np.timedelta64:
                return self._encode_numpy_timedelta64(obj)
            case pv_physical_model.PvSamV1PhysicalModel | pv_physical_model.PvWattsV8PhysicalModel:
                return self._encode_pysam_physical_model(obj)
            case types.FunctionType:
                return self._encode_numba_groupby(obj)
            case _:
                return super().default(obj)

    @staticmethod
    def _encode_numpy_timedelta64(obj: np.timedelta64):
        rmatch = re.match(
            r"numpy.timedelta64\((?P<duration>\d+),'(?P<units>\w+)'\)", repr(obj)
        )
        if not rmatch:
            raise ValueError("Encoding parser failed for numpy time delta: %s", obj)
        return {
            CAMUS_JSON_TYPE: "numpy.timedelta64",
            CAMUS_JSON_DATA: {
                "duration": rmatch.group("duration"),
                "units": rmatch.group("units"),
            },
        }

    @staticmethod
    def _encode_ndarray(obj: np.ndarray):
        if type(obj[0]) is np.timedelta64:
            return {
                CAMUS_JSON_TYPE: "numpy.ndarray",
                CAMUS_JSON_DATA: [
                    _ConfigEncoder._encode_numpy_timedelta64(element) for element in obj
                ],
            }
        raise ValueError(
            f"ndarray encoding not configured for numpy ndarray with dtype {type(obj[0])}",
        )

    @staticmethod
    def _encode_pysam_physical_model(obj: pv_physical_model.PySamPhysicalModel):
        return {
            CAMUS_JSON_TYPE: "pv_physical_model.PySamPhysicalModel",
            CAMUS_JSON_DATA: {
                "class": obj.__class__.__name__,
            }
            | obj.as_json(),
        }

    @staticmethod
    def _encode_numba_groupby(obj):
        return obj.to_dict()


class ConfigHandler:
    """
    Helper to encode and decode time series models configurations
    """

    @classmethod
    def config_decoder(cls, dct):
        """
        Decode dict to obj, with dict structured in format:
        {
            "__camus_json_type__": type_of_object,
            "__camus_json_data__": serialized_object,
        }
        """
        if CAMUS_JSON_TYPE not in dct:
            return dct
        typ = dct[CAMUS_JSON_TYPE]

        if CAMUS_JSON_DATA not in dct:
            raise ValueError(
                f"Did not find field {CAMUS_JSON_DATA} in serialized object {dct}!"
            )
        data = dct[CAMUS_JSON_DATA]

        match typ:
            case "numpy.timedelta64":
                return cls._decode_numpy_timedelta64(data)
            case "numpy.ndarray":
                return cls._decode_ndarray(data)
            case "pv_physical_model.PySamPhysicalModel":
                return cls._decode_pysam_physical_model(data)
            case "numba_groupby_hour_diff":
                return numba_groupby_functions.hour_diff(*data)
            case "numba_groupby_hour_sum":
                return numba_groupby_functions.hour_sum(*data)
            case "numba_groupby_hour_pick":
                return numba_groupby_functions.hour_pick(data)
            case _:
                raise ValueError(
                    f"Did not find a decoder implemented for payload "
                    f"{dct} of type {typ} Did you create one!?"
                )

    @classmethod
    def _decode_numpy_timedelta64(cls, dct):
        return np.timedelta64(dct["duration"], dct["units"])

    @classmethod
    def _decode_ndarray(cls, lst):
        return np.array(lst)

    @classmethod
    def _decode_pysam_physical_model(cls, dct: dict):
        class_mapping = {
            "PvSamV1PhysicalModel": pv_physical_model.PvSamV1PhysicalModel.from_exported,
            "PvWattsV8PhysicalModel": pv_physical_model.PvWattsV8PhysicalModel.from_exported,
        }
        klass_name = dct.pop("class")
        klass = class_mapping[klass_name]
        return klass(**dct)

    @classmethod
    def decode(cls, encoded_configuration: str):
        return json.loads(encoded_configuration, object_hook=cls.config_decoder)

    @classmethod
    def encode(cls, configuration: typing.Any):
        # Not really Any... it will cry at you if it can't encode it
        return json.dumps(configuration, cls=_ConfigEncoder)
