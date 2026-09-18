#!/usr/bin/env bash
# source scripts/env.sh — активирует окружение курса и выставляет переменные.
# uv-команды (uv sync / uv add) после этого работают с .venvs/base, а не с .venv.
_RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$_RL_ROOT/.venvs/base"
export MUJOCO_GL=egl            # рендер MuJoCo без экрана через NVIDIA EGL
export PYTHONUNBUFFERED=1       # логи в tmux/tee появляются сразу
[ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ] && source "$UV_PROJECT_ENVIRONMENT/bin/activate"
cd "$_RL_ROOT"
