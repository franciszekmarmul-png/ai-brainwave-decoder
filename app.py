import argparse
import threading
import webbrowser

from src.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the AI Brainwave Decoder and open it in your browser.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = serve(port=args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"AI Brainwave Decoder is running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
