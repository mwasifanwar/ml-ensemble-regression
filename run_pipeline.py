#!/usr/bin/env python3
"""
Main script to run the advanced KNN regression pipeline
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import main

if __name__ == "__main__":
    main()