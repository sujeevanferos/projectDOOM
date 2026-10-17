#!/usr/bin/env python3
"""Compatibility entry point for cnn_research.training.train_imitation_cnn."""
from cnn_research.training.train_imitation_cnn import (
    ACTIONS,
    ACTION_NAMES,
    DataLoader,
    Dataset,
    DoomImitationCNN,
    F,
    FrameDataset,
    Path,
    TIC_CLASSES,
    create_game,
    epoch_pass,
    evaluate,
    load_demos,
    main,
    nn,
    parse_args,
    preprocess,
)


if __name__ == "__main__":
    main()
