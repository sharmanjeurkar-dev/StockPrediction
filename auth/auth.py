import webbrowser
from urllib.parse import parse_qs, urlparse

from fyers_apiv3 import fyersModel

CLIENT_ID = input("Enter App id:")  # Looks like: ABCD1234EF-100
SECRET_KEY = input("Enter secred id:")  # Looks like: AB12CD34EF
REDIRECT_URI = "https://google.com"  # Must match exactly what you put in the dashboard!


def login_and_generate_token():
    print("🔄 Initializing Fyers Auth Flow...")

    # Initialize the Session Model
    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",
    )

    # Generate the login link
    auth_link = session.generate_authcode()
    print("\n Opening browser for Fyers Login...")
    print(f"If it doesn't open automatically, click here:\n{auth_link}\n")
    webbrowser.open(auth_link)

    # Wait for the user to log in and paste the URL
    print(" Log in on the webpage.")
    print(
        " It will redirect you to a google.com page. The page might look blank or broken—that is normal!"
    )
    print(" Copy the ENTIRE URL from the address bar.")

    redirected_url = input("\n📝 Paste the full URL here: ").strip()

    # Safely extract the auth_code from the URL
    try:
        parsed_url = urlparse(redirected_url)
        auth_code = parse_qs(parsed_url.query)["auth_code"][0]
        print("\n Extracted Auth Code successfully.")
    except Exception as e:
        print("\n Error: Could not find 'auth_code' in the URL you pasted.", e)
        print(
            "Make sure you copied the whole thing. It should look like: https://google.com/?auth_code=..."
        )
        return

    # Exchange the auth_code for the final Access Token
    session.set_token(auth_code)
    response = session.generate_token()

    if response.get("s") == "ok":
        access_token = response["access_token"]
        print("\n SUCCESS! Here is your Daily Access Token:")
        print("-" * 60)
        print(access_token)
        print("-" * 60)

        # Save it to a text file automatically
        with open("access_token.env", "w") as f:
            f.write(f'FYERS_CLIENT_ID="{CLIENT_ID}"\n')
            f.write(f'FYERS_ACCESS_TOKEN="{access_token}"\n')
        print(
            "\n Token automatically saved to 'access_token.txt'. You are ready to scrape!"
        )
    else:
        print("\n Failed to generate token. API Response:")
        print(response)


if __name__ == "__main__":
    login_and_generate_token()
