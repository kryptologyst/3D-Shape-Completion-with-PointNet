#!/usr/bin/env python3
"""Launch the interactive demo for 3D Shape Completion."""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch Streamlit demo."""
    demo_path = Path(__file__).parent / "demo" / "app.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path), "--server.port", "8501"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching demo: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
    except KeyboardInterrupt:
        print("\nDemo stopped.")

if __name__ == "__main__":
    main()
