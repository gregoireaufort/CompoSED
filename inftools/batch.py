from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
import multiprocessing as mp
import traceback as traceback_module
from typing import Callable, Iterable, Literal


@dataclass(frozen=True)
class BatchFitFailure:
    """One failed independent catalog fit."""

    index: int
    error_type: str
    message: str
    traceback: str


@dataclass
class BatchFitResult:
    """Results from running the same fitting function over many objects."""

    results: list
    failures: list[BatchFitFailure] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def successful_indices(self) -> list[int]:
        failed = {failure.index for failure in self.failures}
        return [i for i, result in enumerate(self.results) if i not in failed and result is not None]


def fit_many(
    items: Iterable,
    fit_one: Callable,
    *,
    n_workers: int = 1,
    chunksize: int = 1,
    executor: Literal["process", "thread", "serial"] = "process",
    mp_context: str | None = None,
    fail_fast: bool = True,
) -> BatchFitResult:
    """Run one independent fitting function over a catalog.

    ``fit_one(item)`` should do the full single-object job: construct or reuse
    whatever backend/likelihood/sampler is appropriate and return the
    single-object result. For process-based execution, ``fit_one`` must be
    pickleable; in practice that means a top-level function rather than a
    notebook closure.
    """

    items = list(items)
    if int(n_workers) <= 0:
        raise ValueError("n_workers must be positive.")
    if int(chunksize) <= 0:
        raise ValueError("chunksize must be positive.")
    if executor not in {"process", "thread", "serial"}:
        raise ValueError("executor must be one of: process, thread, serial.")

    if executor == "serial" or int(n_workers) == 1:
        outputs = [_run_one_fit((i, item, fit_one)) for i, item in enumerate(items)]
    else:
        payloads = [(i, item, fit_one) for i, item in enumerate(items)]
        if executor == "thread":
            with ThreadPoolExecutor(max_workers=int(n_workers)) as pool:
                outputs = list(pool.map(_run_one_fit, payloads, chunksize=int(chunksize)))
        else:
            context = mp.get_context(mp_context) if mp_context is not None else None
            with ProcessPoolExecutor(max_workers=int(n_workers), mp_context=context) as pool:
                outputs = list(pool.map(_run_one_fit, payloads, chunksize=int(chunksize)))

    results = [None] * len(items)
    failures: list[BatchFitFailure] = []
    for index, result, failure in outputs:
        if failure is not None:
            if fail_fast:
                raise RuntimeError(
                    f"fit_many failed for object {failure.index}: "
                    f"{failure.error_type}: {failure.message}\n{failure.traceback}"
                )
            failures.append(failure)
            continue
        results[index] = result

    return BatchFitResult(
        results=results,
        failures=failures,
        meta={
            "n_items": len(items),
            "n_workers": int(n_workers),
            "chunksize": int(chunksize),
            "executor": executor,
            "mp_context": mp_context,
            "fail_fast": bool(fail_fast),
        },
    )


def _run_one_fit(payload):
    index, item, fit_one = payload
    try:
        return index, fit_one(item), None
    except Exception as exc:  # pragma: no cover - traceback content is platform dependent
        failure = BatchFitFailure(
            index=int(index),
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=traceback_module.format_exc(),
        )
        return index, None, failure
