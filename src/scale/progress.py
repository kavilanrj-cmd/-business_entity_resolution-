"""Progress and memory observability.

Two jobs:

1. report rows processed, elapsed time, candidate counts and RSS at a fixed
   interval, so a long build is observable rather than a black box;
2. enforce a hard RSS ceiling so a runaway allocation fails *loudly and early*
   instead of becoming an OOM kill with no traceback.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

LOGGER = logging.getLogger(__name__)

try:  # psutil is present in this environment; degrade gracefully if not.
    import psutil

    _PROCESS = psutil.Process(os.getpid())
except Exception:  # pragma: no cover - environment dependent
    psutil = None  # type: ignore[assignment]
    _PROCESS = None  # type: ignore[assignment]


def rss_bytes() -> int:
    """Resident set size of this process, in bytes (0 if unavailable)."""
    if _PROCESS is not None:
        try:
            return int(_PROCESS.memory_info().rss)
        except Exception:  # pragma: no cover
            return 0
    try:  # POSIX fallback
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:  # pragma: no cover
        return 0


def peak_rss_bytes() -> int:
    """High-water mark of RSS, in bytes (0 if unavailable)."""
    if _PROCESS is not None:
        try:
            return int(_PROCESS.memory_info().rss)
        except Exception:  # pragma: no cover
            return 0
    return rss_bytes()


def format_bytes(n: float) -> str:
    """Human-readable byte count, e.g. ``1.4 GB``."""
    step = 1024.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < step:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= step
    return f"{value:.1f} PB"


class MemoryLimitExceeded(RuntimeError):
    """Raised when RSS crosses the configured budget."""


@dataclass
class MemoryGuard:
    """Watches RSS and trips when the budget is exceeded.

    ``check()`` is cheap (one ``/proc``-style syscall via psutil) and is called
    once per batch, not per record.
    """

    budget_bytes: int
    warn_fraction: float = 0.80
    _warned: bool = field(default=False, init=False)
    _peak: int = field(default=0, init=False)

    def rss(self) -> int:
        return rss_bytes()

    def peak(self) -> int:
        return self._peak

    def observe(self) -> int:
        """Sample RSS, fold it into the peak, and return it.

        Use this (not :meth:`peak`) when you want a number to report: ``peak``
        only knows about samples that :meth:`check` already took, so a stage that
        never trips a check would report zero.
        """
        current = rss_bytes()
        if current > self._peak:
            self._peak = current
        return current

    def check(self, where: str = "") -> int:
        current = rss_bytes()
        if current > self._peak:
            self._peak = current
        if current > self.budget_bytes:
            raise MemoryLimitExceeded(
                f"RSS {format_bytes(current)} exceeds the budget "
                f"{format_bytes(self.budget_bytes)}"
                + (f" while {where}" if where else "")
                + ". Lower ScaleConfig.batch_rows or a posting budget."
            )
        if not self._warned and current > self.budget_bytes * self.warn_fraction:
            self._warned = True
            LOGGER.warning(
                "RSS %s has passed %d%% of the %s budget; consider a smaller batch",
                format_bytes(current), int(self.warn_fraction * 100), format_bytes(self.budget_bytes),
            )
        return current

    def report(self) -> str:
        return f"rss={format_bytes(self.rss())} peak={format_bytes(self._peak)}"


class _Rate:
    """Rows/second over a sliding window, for a stable log line."""

    __slots__ = ("_t0", "_n0", "_last_t", "_last_n", "_rate")

    def __init__(self) -> None:
        self._t0 = self._last_t = time.perf_counter()
        self._n0 = self._last_n = 0
        self._rate = 0.0

    def update(self, n: int) -> float:
        now = time.perf_counter()
        self._last_n += n
        dt = now - self._last_t
        if dt >= 1.0:
            self._rate = (self._last_n - self._n0) / dt if self._n0 else self._last_n / dt
            self._t0, self._n0 = now, self._last_n
            self._last_t = now
        return self._rate

    @property
    def rate(self) -> float:
        return self._rate


@dataclass
class Progress:
    """Interval-based progress logger.

    Usage::

        with Progress("normalise train_source2", total=5_034_616) as p:
            for chunk in chunks:
                p.add(len(chunk))
    """

    label: str
    total: int | None = None
    interval_s: float = 10.0
    guard: MemoryGuard | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    _processed: int = field(default=0, init=False)
    _start: float = field(default_factory=time.perf_counter, init=False)
    _last_log: float = field(default=0.0, init=False)
    _rate: _Rate = field(default_factory=_Rate, init=False)

    def add(self, n: int, **extra: Any) -> None:
        """Record ``n`` more processed units, logging if the interval elapsed."""
        self._processed += n
        if extra:
            self.extra.update(extra)
        self._rate.update(n)
        now = time.perf_counter()
        if now - self._last_log >= self.interval_s:
            self._last_log = now
            self.log()

    def set(self, **extra: Any) -> None:
        self.extra.update(extra)

    @property
    def processed(self) -> int:
        return self._processed

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def log(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_log < self.interval_s:
            return
        self._last_log = now
        self._emit(logging.INFO)

    def _emit(self, level: int) -> None:
        elapsed = self.elapsed
        bits = [f"{self.label}: {self._processed:,} rows", f"{elapsed:,.1f}s"]
        if self.total:
            pct = 100.0 * self._processed / self.total
            eta = (elapsed / self._processed * (self.total - self._processed)) if self._processed else float("nan")
            bits.append(f"{pct:5.1f}%")
            bits.append(f"eta {eta / 60:,.1f}m")
        if self._rate.rate:
            bits.append(f"{self._rate.rate:,.0f} rows/s")
        if self.guard is not None:
            bits.append(f"rss {format_bytes(self.guard.rss())}")
        for key, value in self.extra.items():
            bits.append(f"{key}={value:,}" if isinstance(value, int) else f"{key}={value}")
        LOGGER.log(level, " | ".join(bits))

    def __enter__(self) -> "Progress":
        self._start = time.perf_counter()
        self._last_log = 0.0
        LOGGER.info("START %s", self.label)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            LOGGER.error("FAILED %s after %.1fs (rss %s)", self.label, self.elapsed,
                         format_bytes(self.guard.rss()) if self.guard else "?")
            return
        self._emit(logging.INFO)
        if self.guard is not None:
            self.guard.check(self.label)
            LOGGER.info("DONE  %s | peak %s", self.label, format_bytes(self.guard.peak()))
        else:
            LOGGER.info("DONE  %s", self.label)


@contextmanager
def log_stage(label: str, guard: MemoryGuard | None = None) -> Iterator[None]:
    """Log the start, peak RSS and duration of a named stage."""
    t0 = time.perf_counter()
    LOGGER.info("=== %s ===", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        extra = f" | rss {format_bytes(rss_bytes())}" if rss_bytes() else ""
        LOGGER.info("=== %s finished in %.1fs%s ===", label, elapsed, extra)


def make_guard(config: Any) -> MemoryGuard:
    """Build a :class:`MemoryGuard` from a :class:`~src.scale.config.ScaleConfig`."""
    return MemoryGuard(
        budget_bytes=int(getattr(config, "memory_budget_bytes", 6 * 1024**3)),
        warn_fraction=float(getattr(config, "memory_warn_fraction", 0.8)),
    )
