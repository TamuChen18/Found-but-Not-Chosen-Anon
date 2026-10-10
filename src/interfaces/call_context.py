"""Per-request provenance tags (task_id, operator, ...) carried to the API client's call log.

Context variables do not cross ThreadPoolExecutor boundaries on their own; submit work with
contextvars.copy_context().run to keep the tags.
"""
import contextlib
import contextvars

_CTX = contextvars.ContextVar("call_ctx", default={})


@contextlib.contextmanager
def call_context(**tags):
    token = _CTX.set({**_CTX.get(), **tags})
    try:
        yield
    finally:
        _CTX.reset(token)


def current_tags():
    return dict(_CTX.get())
