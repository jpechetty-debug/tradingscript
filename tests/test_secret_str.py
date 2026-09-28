from __future__ import annotations

import pytest
from core.config import SecretStr, get_secret_value


def test_secret_str_masking():
    s = SecretStr("my-super-secret-token")
    assert str(s) == "SecretStr('****')"
    assert repr(s) == "SecretStr('my-s...oken')"
    assert f"{s}" == "SecretStr('****')"
    assert "%s" % s == "SecretStr('****')"

    # Short secret
    short_s = SecretStr("short")
    assert repr(short_s) == "SecretStr('****')"
    assert str(short_s) == "SecretStr('****')"

    # Empty secret
    empty_s = SecretStr("")
    assert str(empty_s) == ""
    assert repr(empty_s) == "SecretStr('****')"
    assert not bool(empty_s)


def test_secret_str_retrieval():
    raw = "real_api_secret_12345"
    s = SecretStr(raw)
    assert s.get_secret_value() == raw
    assert get_secret_value(s) == raw
    assert get_secret_value("plain_str") == "plain_str"
    assert get_secret_value(None) == ""


def test_secret_str_prevents_concatenation_leaks():
    s = SecretStr("secret_key")

    with pytest.raises(TypeError, match="Cannot concatenate SecretStr"):
        _ = s + "_suffix"

    with pytest.raises(TypeError, match="Cannot concatenate SecretStr"):
        _ = "prefix_" + s


def test_secret_str_equality_and_hashing():
    s1 = SecretStr("secret")
    s2 = SecretStr("secret")
    s3 = SecretStr("other")

    assert s1 == s2
    assert s1 == "secret"
    assert s1 != s3
    assert s1 != "other"
    assert s1 != 123
    assert len(s1) == 6
    assert hash(s1) == hash(s2)
