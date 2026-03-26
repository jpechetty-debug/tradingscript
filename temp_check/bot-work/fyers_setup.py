import os
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

load_dotenv()

def generate_access_token():
    client_id = os.getenv("FYERS_CLIENT_ID")
    secret_key = os.getenv("FYERS_SECRET_KEY")
    redirect_uri = os.getenv("FYERS_REDIRECT_URI")

    if not all([client_id, secret_key, redirect_uri]):
        print("❌ Error: FYERS_CLIENT_ID, FYERS_SECRET_KEY, or FYERS_REDIRECT_URI missing in .env")
        return

    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code"
    )

    response = session.generate_auth_code()
    print("\n1. Open this URL in your browser and log in:")
    print(f"🔗 {response}")
    
    auth_code = input("\n2. After login, you will be redirected. Paste the 'auth_code' from the URL here: ").strip()
    
    if not auth_code:
        print("❌ Auth code cannot be empty.")
        return

    session.set_token(auth_code)
    try:
        response = session.generate_access_token()
        access_token = response.get("access_token")
        if access_token:
            from utils.messaging import mask_token
            print("\n✅ Success! Your Access Token for today is:")
            print(f"\n{mask_token(access_token)}\n")
            print("Copy this into your .env as FYERS_ACCESS_TOKEN.")
        else:
            print(f"❌ Failed to generate access token: {response}")
    except Exception as e:
        print(f"❌ Error during token generation: {e}")

if __name__ == "__main__":
    generate_access_token()
