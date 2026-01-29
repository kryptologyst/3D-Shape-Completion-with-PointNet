#!/usr/bin/env python3
"""Main evaluation script for 3D Shape Completion."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.eval.evaluate import main

if __name__ == "__main__":
    main()
