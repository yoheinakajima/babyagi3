"""
Entry point for the agent.

Usage:
    python main.py          # Run CLI agent
    python main.py serve    # Run API server
    python main.py serve 8080  # Run on custom port
"""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        import uvicorn
        from server import app
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        from agent import main as agent_main
        agent_main()


if __name__ == "__main__":
    main()
