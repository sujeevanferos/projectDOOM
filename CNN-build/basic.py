#!/usr/bin/env python3
"""Compatibility entry point for cnn_research.exploration.basic."""


if __name__ == "__main__":
    import runpy
    runpy.run_module("cnn_research.exploration.basic", run_name="__main__")
else:
    from cnn_research.exploration.basic import (
        choice,
        sleep,
    )
