#!/bin/bash
# Extract best checkpoints for all three agents
# Run from: ~/scratch/soccertwos/soccer-twos-starter/

set -e

source /usr/local/pace-apps/manual/packages/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ~/scratch/envs/soccertwos

BASEDIR=~/scratch/soccertwos/soccer-twos-starter

echo "=== Finding best checkpoints ==="

for VARIANT in vanilla_tvr shaped_tvr curriculum_tvr; do
    echo ""
    echo "--- $VARIANT ---"

    TRIAL_DIR=$(ls -d $BASEDIR/ray_results/$VARIANT/PPO_* 2>/dev/null | head -1)
    if [ -z "$TRIAL_DIR" ]; then
        echo "  No trial found for $VARIANT, skipping"
        continue
    fi

    PROGRESS="$TRIAL_DIR/progress.csv"
    if [ ! -f "$PROGRESS" ]; then
        echo "  No progress.csv found, skipping"
        continue
    fi

    # Find the iteration with highest episode_reward_mean (column 3)
    # Also get training_iteration (column 11)
    BEST_LINE=$(tail -n +2 "$PROGRESS" | awk -F',' '{print NR+1, $3, $11}' | sort -k2 -n -r | head -1)
    BEST_REWARD=$(echo $BEST_LINE | awk '{print $2}')
    BEST_ITER=$(echo $BEST_LINE | awk '{print $3}')

    echo "  Best reward: $BEST_REWARD at iteration $BEST_ITER"

    # Find the closest checkpoint (checkpointed every 50 iters)
    CLOSEST_CKPT=$(ls -d $TRIAL_DIR/checkpoint_* 2>/dev/null | sort | tail -1)
    if [ -z "$CLOSEST_CKPT" ]; then
        echo "  No checkpoints found, skipping"
        continue
    fi

    CKPT_NUM=$(basename $CLOSEST_CKPT | sed 's/checkpoint_0*//')
    CKPT_FILE="$CLOSEST_CKPT/checkpoint-$CKPT_NUM"

    echo "  Using latest checkpoint: $CLOSEST_CKPT"

    # Determine output directory
    case $VARIANT in
        vanilla_tvr)    AGENT_DIR="$BASEDIR/agent_vanilla" ;;
        shaped_tvr)     AGENT_DIR="$BASEDIR/agent_shaped" ;;
        curriculum_tvr) AGENT_DIR="$BASEDIR/agent_curriculum" ;;
    esac

    mkdir -p "$AGENT_DIR"

    echo "  Extracting to $AGENT_DIR/checkpoint.pth"
    python3 $BASEDIR/extract_checkpoint.py "$CKPT_FILE" "$AGENT_DIR/checkpoint.pth"
done

echo ""
echo "=== Done! ==="
echo "Agent directories:"
for d in agent_vanilla agent_shaped agent_curriculum; do
    if [ -f "$BASEDIR/$d/checkpoint.pth" ]; then
        SIZE=$(ls -lh "$BASEDIR/$d/checkpoint.pth" | awk '{print $5}')
        echo "  $d/checkpoint.pth ($SIZE)"
    else
        echo "  $d/checkpoint.pth - MISSING"
    fi
done
