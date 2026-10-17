#!/usr/bin/env python3
"""Compatibility entry point for cnn_research.exploration.labelled_buffer_cnn."""
from cnn_research.exploration.labelled_buffer_cnn import (
    ACTIONS,
    DoomCNN,
    Path,
    ReplayBuffer,
    choose_action,
    create_game,
    deque,
    load_checkpoint,
    nn,
    parse_args,
    play_episode,
    preprocess,
    run,
    save_checkpoint,
    train_step,
)


if __name__ == "__main__":
    run(parse_args())
