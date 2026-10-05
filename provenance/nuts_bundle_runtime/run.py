#!/usr/bin/env python3
"""Entry point for the self-contained, frozen six-reservoir NUTS experiment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from portable_runtime.cli import main
if __name__=='__main__':main()
