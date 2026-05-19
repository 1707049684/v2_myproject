# -*- coding: utf-8 -*-
"""
Generative Cognitive Diagnosis (G-NCDM)

A PyTorch implementation of Generative Cognitive Diagnosis.
"""

import sys
import os

# Ensure local imports work
if __name__ != '__main__':
    if not hasattr(sys.modules[__name__], '__path__'):
        pass
    else:
        # Make core modules importable
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

__version__ = "1.0.0"
