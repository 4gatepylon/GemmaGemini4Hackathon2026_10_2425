# Shorthand for ios_kernels_launcher.py: `make <command> args="--flags"` runs
# `uv run ios_kernels_launcher.py <command> --flags`. All logic lives in the launcher.
#
#   make help                 list commands
#   make help args=serve      flags for one command
#   make serve args="--kernel 20261024 --port 9000"
#
# Works from anywhere: make -C ios_kernels <command>

HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
LAUNCHER := uv run --quiet --project $(HERE) $(HERE)ios_kernels_launcher.py

COMMANDS := setup build serve serve-fake serve-mlx serve-llamacpp chat ask smoke \
            test test-fast bench bench-ours reference kernels new-kernel clean

.DEFAULT_GOAL := help
.PHONY: help $(COMMANDS)

help:
	@$(LAUNCHER) help $(args)

$(COMMANDS):
	@$(LAUNCHER) $@ $(args)
