import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_env_path() -> Path:
    return REPO_ROOT / ".env"


def _upsert_env_var(env_path: Path, key: str, value: str) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    prefix = f"{key}="

    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f"{prefix}{value}"
            break
    else:
        lines.append(f"{prefix}{value}")

    env_path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def persist_access_token(access_token: str, env_path: Path | None = None) -> Path:
    target = env_path or _repo_env_path()
    _upsert_env_var(target, "FYERS_ACCESS_TOKEN", access_token)
    return target


def generate_access_token():
    from fyers_apiv3 import fyersModel

    load_dotenv(dotenv_path=_repo_env_path())

    client_id = os.getenv("FYERS_CLIENT_ID")
    secret_key = os.getenv("FYERS_SECRET_KEY")
    redirect_uri = os.getenv("FYERS_REDIRECT_URI")

    if not all([client_id, secret_key, redirect_uri]):
        print("? Error: FYERS_CLIENT_ID, FYERS_SECRET_KEY, or FYERS_REDIRECT_URI missing in .env")
        return

    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )

    response = session.generate_auth_code()
    print("\n1. Open this URL in your browser and log in:")
    print(f"? {response}")

    auth_code = input(
        "\n2. After login, you will be redirected. Paste the 'auth_code' from the URL here: "
    ).strip()

    if not auth_code:
        print("? Auth code cannot be empty.")
        return

    session.set_token(auth_code)
    try:
        response = session.generate_access_token()
        access_token = response.get("access_token")
        if access_token:
            from utils.messaging import mask_token

            env_path = persist_access_token(access_token)
            print("\n? Success! Your Access Token for today is:")
            print(f"\n{mask_token(access_token)}\n")
            print(f"Updated {env_path} with FYERS_ACCESS_TOKEN.")
        else:
            print(f"? Failed to generate access token: {response}")
    except Exception as e:
        print(f"? Error during token generation: {e}")


if __name__ == "__main__":
    generate_access_token()
