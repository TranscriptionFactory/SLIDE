#!/usr/bin/env bash
# Wrapper around cc that runs from /tmp to avoid GCC mistaking ./specs dir
# for a spec file (GCC implicitly reads ./specs if it exists).
exec env -C /tmp cc "$@"
