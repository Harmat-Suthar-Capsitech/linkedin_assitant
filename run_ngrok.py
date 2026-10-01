"""
Streamlit Public Deployment Script via ngrok.
Starts Streamlit (app.py) and opens a secure public HTTPS tunnel using ngrok.
"""
import argparse
import os
import socket
import subprocess
import sys
import time
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

try:
    from pyngrok import ngrok, conf
except ImportError:
    print("❌ 'pyngrok' is not installed. Installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
    from pyngrok import ngrok, conf


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a local port is already occupied/listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def main():
    parser = argparse.ArgumentParser(description="Expose Streamlit UI to the web using ngrok.")
    parser.add_argument("--port", type=int, default=8501, help="Port where Streamlit runs (default: 8501)")
    parser.add_argument("--app", type=str, default="app.py", help="Streamlit script to run (default: app.py)")
    args = parser.parse_args()

    # 1. Retrieve and configure ngrok authtoken
    token = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if not token:
        print("❌ Error: NGROK_AUTHTOKEN is missing in your .env file!")
        print("Please add 'NGROK_AUTHTOKEN=your_token_here' to .env and retry.")
        sys.exit(1)

    print("🔐 Authenticating with ngrok...")
    ngrok.set_auth_token(token)

    # 2. Check if Streamlit is already running, or start it
    streamlit_proc = None
    if not is_port_in_use(args.port):
        print(f"🚀 Starting Streamlit on port {args.port}...")
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            args.app,
            "--server.port",
            str(args.port),
            "--server.headless",
            "true",
        ]
        streamlit_proc = subprocess.Popen(cmd)

        # Wait for Streamlit to start listening
        print("⏳ Waiting for Streamlit server to initialize...", end="", flush=True)
        max_wait = 30
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if is_port_in_use(args.port):
                print(" Ready!")
                break
            time.sleep(1)
            print(".", end="", flush=True)
        else:
            print("\n⚠️ Warning: Streamlit did not start within 30 seconds, attempting tunnel anyway...")
    else:
        print(f"ℹ️ Streamlit is already running on port {args.port}. Tunnelling to existing instance.")

    # 3. Create ngrok tunnel
    print(f"🌐 Creating secure public ngrok tunnel on port {args.port}...")
    try:
        # Kill any existing tunnels on this config to avoid conflicts
        ngrok.kill()
        tunnel = ngrok.connect(addr=args.port, proto="http")
        public_url = tunnel.public_url

        # Ensure HTTPS URL
        if public_url.startswith("http://"):
            public_url = "https://" + public_url[7:]

        print("\n" + "=" * 65)
        print("🎉 Streamlit Web App is LIVE & Publicly Accessible!")
        print("=" * 65)
        print(f"🔗 Public URL:  {public_url}")
        print(f"🏠 Local URL:   http://localhost:{args.port}")
        print("=" * 65)
        print("💡 Share the Public URL with anyone to access your Streamlit UI.")
        print("Press Ctrl+C anytime to stop the tunnel and server.\n")

        # Keep alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down ngrok tunnel...")
    except Exception as e:
        print(f"\n❌ Error establishing ngrok tunnel: {e}")
    finally:
        try:
            ngrok.kill()
            print("✅ ngrok tunnels closed.")
        except Exception:
            pass

        if streamlit_proc:
            print("🛑 Stopping Streamlit process...")
            streamlit_proc.terminate()
            try:
                streamlit_proc.wait(timeout=5)
            except Exception:
                streamlit_proc.kill()
            print("✅ Streamlit stopped.")


if __name__ == "__main__":
    main()
