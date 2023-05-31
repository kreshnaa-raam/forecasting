import logging
from typing import Callable, Any

from joblib import Parallel

from time_series_models.decorators import debug


logger = logging.getLogger(__name__)


@debug(logger)
def parallel_patch(
    initializer: Callable, initargs: tuple = (), idle_worker_timeout: int = 100
):
    """
    Patch the JobLib Parallel class to set initializer arguments for the executor.
    This method gets the existing implementation of _initialize_backend, and updates
    the backend_args before calling the original method.
    :param initializer: a function
    :param initargs: a tuple of arguments for the function
    :param idle_worker_timeout: the idle worker timeout value in seconds (see loky get_reusable_executor)
    """

    base_method = getattr(Parallel, "_initialize_backend")

    def _initialize_backend(self):
        # Apply custom initializers to the backend args
        self._backend_args.update(
            dict(
                initializer=initializer,
                initargs=initargs,
                idle_worker_timeout=idle_worker_timeout,
                kill_workers=True,
            )
        )
        logger.debug("Intitializing Backend with %s", initargs)
        return base_method(self)

    logger.warning("Patched JobLib Parallel to inject initializer to the executor")

    Parallel._initialize_backend = _initialize_backend
