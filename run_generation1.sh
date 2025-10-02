#!/usr/bin/env bash
set -euo pipefail

# --- (A) Activate env (optional but recommended)
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate king
fi

# --- (B) Point to ONE CARLA version (server + Python egg must MATCH)
# EDIT these 3 lines to your actual install/version:
export CARLA_ROOT="/home/vegetableclean/carla_packed_linux/CARLA_0.9.15"
export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export CARLA_EGG="${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"

# --- (C) Python path (KING + ScenarioRunner + CARLA)
# Put CARLA egg FIRST to avoid picking up the wrong API.
export PYTHONPATH="$CARLA_EGG:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$(pwd -P)/scenario_runner:$(pwd -P)/leaderboard:${PYTHONPATH:-}"

# --- (D) (Optional) Avoid GPU OOM for a first smoke test; comment this to use GPU
export CUDA_VISIBLE_DEVICES=""

# --- (E) Print sanity info
python - <<'PY'
import sys, importlib
print("Python:", sys.executable)
try:
    carla = importlib.import_module("carla")
    print("carla OK:", getattr(carla,"__version__", "unknown"))
except Exception as e:
    print("carla import failed ->", e); raise
PY

# --- (F) Start server if not already running (background)
# If you already run the server in another terminal, comment this block.
if ! nc -z localhost 2000; then
  SDL_VIDEODRIVER=offscreen "${CARLA_SERVER}" -RenderOffScreen -quality-level=Epic -world-port=2000 >/tmp/carla.log 2>&1 &
  sleep 10
fi

########### run KING generation (start small to avoid OOM) ############

# 1 agent, fewer iters first (smoke test)
python generate_scenarios.py \
  --num_agents 2 \
  --save_path ./generation_results/agents_1_smoke \
  --opt_iters 1 --beta1 0.8 --beta2 0.999 --w_adv_col 0.0 --w_adv_rd 20.0

# If the above finishes, you can scale up (uncomment next blocks):

# # 4 Agents
# python generate_scenarios.py \
#   --num_agents 2 --save_path ./generation_results/agents_4 \
#   --opt_iters 100 --beta1 0.8 --beta2 0.99 --w_adv_col 3.0 --w_adv_rd 20.0

# # 2 agents
# python generate_scenarios.py \
#   --num_agents 2 --save_path ./generation_results/agents_2 \
#   --opt_iters 120 --beta1 0.8 --beta2 0.99 --w_adv_col 5.0 --w_adv_rd 23.0

# # 1 agent (full)
# python generate_scenarios.py \
#   --num_agents 1 --save_path ./generation_results/agents_1 \
#   --opt_iters 150 --beta1 0.8 --beta2 0.999 --w_adv_col 0.0 --w_adv_rd 20.0

echo "Overall results"
echo "==============="
python3 tools/parse_generation_results.py --results_dir ./generation_results/
echo "1 agent (smoke)"
echo "==============="
python3 tools/parse_generation_results.py --results_dir ./generation_results/agents_1_smoke --num_agents 1 

