from pathlib import Path

from tools.scripts.fyers_setup import persist_access_token


def test_persist_access_token_replaces_existing_value(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "FYERS_CLIENT_ID=client\nFYERS_ACCESS_TOKEN=old-token\nTELEGRAM_CHAT_ID=123\n",
        encoding="utf-8",
    )

    persist_access_token("new-token", env_path)

    assert env_path.read_text(encoding="utf-8") == (
        "FYERS_CLIENT_ID=client\nFYERS_ACCESS_TOKEN=new-token\nTELEGRAM_CHAT_ID=123\n"
    )


def test_persist_access_token_appends_missing_value(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("FYERS_CLIENT_ID=client\n", encoding="utf-8")

    target = persist_access_token("new-token", env_path)

    assert target == Path(env_path)
    assert env_path.read_text(encoding="utf-8") == (
        "FYERS_CLIENT_ID=client\nFYERS_ACCESS_TOKEN=new-token\n"
    )
