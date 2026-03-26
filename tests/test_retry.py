import time
import pytest
from core.retry import CircuitBreaker, CircuitBreakerOpen, guarded_call, MaxRetriesExceeded

class TestCircuitBreaker:

    def test_starts_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state.name == "CLOSED"

    def test_trips_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        for _ in range(3):
            try: cb.call(lambda: (_ for _ in ()).throw(OSError("fail")))
            except OSError: pass
        assert cb.state.name == "OPEN"

    def test_open_blocks_calls(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        try: cb.call(lambda: (_ for _ in ()).throw(OSError()))
        except OSError: pass
        with pytest.raises(CircuitBreakerOpen):
            cb.call(lambda: "ok")

    def test_transitions_half_open_after_timeout(self, monkeypatch):
        cb = CircuitBreaker("test", failure_threshold=1, reset_timeout=0.01)
        try: cb.call(lambda: (_ for _ in ()).throw(OSError()))
        except OSError: pass
        time.sleep(0.02)
        assert cb.state.name == "HALF_OPEN"

    def test_half_open_probe_success_closes(self, monkeypatch):
        cb = CircuitBreaker("test", failure_threshold=1, reset_timeout=0.01)
        try: cb.call(lambda: (_ for _ in ()).throw(OSError()))
        except OSError: pass
        time.sleep(0.02)
        cb.call(lambda: "ok")
        assert cb.state.name == "CLOSED"

    def test_half_open_probe_failure_reopens(self):
        cb = CircuitBreaker("test", failure_threshold=1, reset_timeout=0.01)
        try: cb.call(lambda: (_ for _ in ()).throw(OSError()))
        except OSError: pass
        time.sleep(0.02)
        try: cb.call(lambda: (_ for _ in ()).throw(OSError()))
        except OSError: pass
        assert cb.state.name == "OPEN"

    def test_reset_clears_state(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        try: cb.call(lambda: (_ for _ in ()).throw(OSError()))
        except OSError: pass
        cb.reset()
        assert cb.state.name == "CLOSED"

    def test_guarded_call_retries_before_tripping(self):
        calls = []
        cb = CircuitBreaker("test", failure_threshold=5)
        def flaky():
            calls.append(1)
            raise OSError("transient")
        with pytest.raises(MaxRetriesExceeded):
            guarded_call(flaky, breaker=cb, max_attempts=3, base_delay=0.001)
        assert len(calls) == 3          # retried 3 times
        assert cb._failures == 1        # breaker only counted 1 persistent failure
