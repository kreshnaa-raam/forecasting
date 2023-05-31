from enum import unique, Enum

import numpy as np


@unique
class TimeUnitEnum(Enum):
    """
    ENUM for calendar conversion of numpy datetime64 arrays
    """

    YEAR = "Y"
    MONTH = "M"
    DAY = "D"
    HOUR = "h"
    MINUTE = "m"
    SECOND = "s"
    MICROSECOND = "us"
    INT_DAY_OF_WEEK = "w"
    DAY_OF_WEEK = "W"

    @property
    def mapping(self):
        return {
            self.YEAR: self._as_year,
            self.MONTH: self._as_month,
            self.DAY: self._as_day,
            self.HOUR: self._as_hour,
            self.MINUTE: self._as_minute,
            self.SECOND: self._as_second,
            self.MICROSECOND: self._as_microsecond,
            self.INT_DAY_OF_WEEK: self._as_int_day_of_week,
            self.DAY_OF_WEEK: self._as_day_of_week,
        }

    def as_unit(self, array):
        """
        Convert the given numpy datetime64 array to the TimeUnitEnum unit.

        Example:
            test_array = np.arange(
                np.datetime64("2020-01"),
                np.datetime64("2022-02"),
                step=np.timedelta64(6, "M"),
            )
            result = TimeUnitEnum.YEAR.as_unit(test_array)
            np.testing.assert_array_equal(result, np.array((2020, 2020, 2021, 2021, 2022)))

        :param array: a numpy datetime64 array in the input unit
        :return: an array of datetime64 array in the output unit
        """
        return self.mapping[self](array)

    @classmethod
    def _as_year(cls, array):
        return (cls.array_as(array, TimeUnitEnum.YEAR)).astype("i4") + 1970

    @classmethod
    def _as_month(cls, array):
        return (
            (
                cls.array_as(array, TimeUnitEnum.MONTH)
                - cls.array_as(array, TimeUnitEnum.YEAR)
            )
        ).astype("i4") + 1

    @classmethod
    def _as_day(cls, array):
        return (
            (
                cls.array_as(array, TimeUnitEnum.DAY)
                - cls.array_as(array, TimeUnitEnum.MONTH)
            )
        ).astype("i4") + 1

    @classmethod
    def _as_hour(cls, array):
        return (
            (array - cls.array_as(array, TimeUnitEnum.DAY)).astype("m8[h]").astype("i4")
        )

    @classmethod
    def _as_minute(cls, array):
        return (
            (array - cls.array_as(array, TimeUnitEnum.HOUR))
            .astype("m8[m]")
            .astype("i4")
        )

    @classmethod
    def _as_second(cls, array):
        return (
            (array - cls.array_as(array, TimeUnitEnum.MINUTE))
            .astype("m8[s]")
            .astype("i4")
        )

    @classmethod
    def _as_microsecond(cls, array):
        return (
            (array - cls.array_as(array, TimeUnitEnum.SECOND))
            .astype("m8[us]")
            .astype("i4")
        )

    @classmethod
    def _as_int_day_of_week(cls, array):
        """
        Monday is zero, Sunday is six
        :param array:
        :return:
        """
        return (cls.array_as(array, cls.DAY).astype("i4") - 4) % 7

    @classmethod
    def _as_day_of_week(cls, array):
        """
        :param array: datetime64 array
        :return: day of week as strings
        """
        result = np.empty(array.shape, dtype="U9")
        dow = cls._as_int_day_of_week(array)
        result[dow == 0] = "Monday"
        result[dow == 1] = "Tuesday"
        result[dow == 2] = "Wednesday"
        result[dow == 3] = "Thursday"
        result[dow == 4] = "Friday"
        result[dow == 5] = "Saturday"
        result[dow == 6] = "Sunday"

        return result

    @classmethod
    def array_as(cls, array, unit):
        return array.astype("M8[{}]".format(unit.value))
