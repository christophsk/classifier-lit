#!/bin/bash
# Two arguments are required. The first is the .csv
# with the sentence data, and the second is the pytorch
# model directory.

# !!! UPDATE FOR YOUR ENVIRONMENT!!!
export PROJECT=$HOME/pycharm_projects/classifier-lit
export TRANSFORMERS_CACHE=/Users/chris/.cache/huggingface/transformers/

# CLI arguments
export PORT=5432
export BSZ=8
export MAX_LEN=128

#-------------------------------------------------
# You shouldn't need to change anything below here
#-------------------------------------------------
export LIT=$PROJECT/classifier_lit/clf_lit.py

if (( $# != 3 )); then
    >&2 echo "Usage: bash clf.sh <data file> <model file> <num_labels>"
    exit 1
fi

if ! [ -f $LIT ]; then
  >&2 echo "Cannot find clf_lit.py in $PROJECT"
  exit 1
fi

if ! [ -f $1 ]; then
  >&2 echo "Data file not found: $1"
  exit 1
fi

export SENT_CSV=$1
export CLF_MODEL=$2
export NUM_LABELS=$3

echo "starting classifier_lit"
echo ""

python $LIT \
  --model_path $CLF_MODEL \
  --data_path $SENT_CSV \
  --num_labels $NUM_LABELS \
  --batch_size $BSZ \
  --max_seq_len $MAX_LEN \
  --port $PORT
