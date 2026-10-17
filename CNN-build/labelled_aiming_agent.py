#!/usr/bin/env python3
"""Compatibility entry point for cnn_research.teacher.labelled_aiming_agent."""
from cnn_research.teacher.labelled_aiming_agent import (
    ACTIONS,
    ACTION_NAMES,
    Path,
    SCREEN_H,
    SCREEN_W,
    choose_action,
    create_game,
    episode,
    get_enemy,
    main,
    rgb_chw_uint8,
)


if __name__ == "__main__":
    main()
