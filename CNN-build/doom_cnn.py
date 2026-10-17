#!/usr/bin/env python3
"""Compatibility entry point for cnn_research.exploration.doom_cnn."""


if __name__ == "__main__":
    import runpy
    runpy.run_module("cnn_research.exploration.doom_cnn", run_name="__main__")
else:
    from cnn_research.exploration.doom_cnn import (
        ACTIONS,
        DoomCNN,
        Transition,
        create_game,
        deque,
        namedtuple,
        nn,
        optimize,
        preprocess,
        run,
        tensorize,
    )
