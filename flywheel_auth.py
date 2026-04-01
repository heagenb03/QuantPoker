import requests
import webbrowser
import os
import hashlib
import base64
import secrets
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from threading import Thread

REGISTER_URL = "https://flywheel.paradigma.inc/mcp-server/register"
AUTH_URL     = "https://flywheel.paradigma.inc/mcp-server/authorize"
TOKEN_URL    = "https://flywheel.paradigma.inc/mcp-server/token"
CALLBACK_PORT = 3333
CALLBACK_URL  = f"http://127.0.0.1:{CALLBACK_PORT}/callback"

# Shared state between the HTTP callback handler and main thread
_state = {"code": None}

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if "code" in qs:
            _state["code"] = qs["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"Authorization successful! Return to your terminal.")
        else:
            self.send_response(400)
            self.end_headers()
    def log_message(self, *args):
        pass

def generate_pkce():
    verifier  = secrets.token_urlsafe(64)
    digest    = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge

def update_env(key, value):
    lines = []
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            lines = f.readlines()
    with open(".env", "w") as f:
        found = False
        for line in lines:
            if line.startswith(f"{key}="):
                f.write(f"{key}={value}\n")
                found = True
            else:
                f.write(line)
        if not found:
            f.write(f"{key}={value}\n")

def main():
    print("Registering client with Flywheel...")
    res = requests.post(REGISTER_URL, json={
        "client_name": "Poker Bot Orchestrator",
        "redirect_uris": [CALLBACK_URL],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"]
    })
    if res.status_code not in (200, 201):
        print(f"Registration failed: {res.text}")
        return

    client_id = res.json()["client_id"]
    verifier, challenge = generate_pkce()

    auth_uri = (
        f"{AUTH_URL}?response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri={CALLBACK_URL}"
        f"&code_challenge={challenge}"
        f"&code_challenge_method=S256"
    )

    print("\n" + "=" * 80)
    print("OPEN THE FOLLOWING URL IN YOUR BROWSER TO AUTHORIZE:")
    print(auth_uri)
    print("=" * 80 + "\n")
    webbrowser.open(auth_uri)

    # Start HTTP server in background thread to catch the redirect
    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), CallbackHandler)
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()

    print("Waiting... (browser should redirect automatically)")
    print("If not, paste the full http://127.0.0.1:3333/callback?... URL here and press Enter:")

    while _state["code"] is None:
        try:
            user_input = input("> ").strip()
            if user_input:
                if "code=" in user_input:
                    _state["code"] = parse_qs(urlparse(user_input).query).get("code", [""])[0]
                else:
                    _state["code"] = user_input
        except EOFError:
            import time; time.sleep(0.5)

    server.shutdown()
    code = _state["code"]
    print(f"\nCode received! Exchanging for tokens...")

    token_res = requests.post(TOKEN_URL, data={
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  CALLBACK_URL,
        "client_id":     client_id,
        "code_verifier": verifier
    })

    if token_res.status_code == 200:
        refresh_token = token_res.json().get("refresh_token")
        if refresh_token:
            update_env("FLYWHEEL_API_KEY", refresh_token)
            print("\n✓ SUCCESS! Permanent token saved to .env — you never need to run this again.")
        else:
            print("No refresh_token in response:", token_res.json())
    else:
        print(f"Token exchange failed: {token_res.text}")

if __name__ == "__main__":
    main()
