import time
import uuid
from contextlib import contextmanager


@contextmanager
def trace_request():

    request_id = str(uuid.uuid4())

    start = time.perf_counter()

    try:

        yield request_id

    finally:

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000