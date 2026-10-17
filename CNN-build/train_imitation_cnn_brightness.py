#!/usr/bin/env python3
"""Compatibility entry point for cnn_research.training.train_imitation_cnn_brightness."""
from cnn_research.training.train_imitation_cnn_brightness import (
    BrightnessTrainingDataset,
    DataLoader,
    Dataset,
    DoomImitationCNN,
    FrameDataset,
    Path,
    TIC_CLASSES,
    datetime,
    epoch_pass,
    load_demos,
    main,
    sha256_file,
    timezone,
)


if __name__ == "__main__":
    main()
