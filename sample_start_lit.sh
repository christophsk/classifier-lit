#!/bin/bash
#
# Two arguments are required. The first is the .csv
# with the sentence data, and the second is the pytorch
# model directory.
#

# !!! Update for your environment or put in your PYTHONPATH !!!
export LIT=~/projects/classifier-lit/classifier_lit/clf_lit.py

if (( $# != 2 )); then
    >&2 echo "usage: bash start_lit.sh <data file> <model file>"
    >&2 echo ""
    exit 1
fi

if ! [ -f $1 ]; then
  >&2 echo "data file not found: $1"
  exit 1
fi

if ! [ -d $2 ]; then
  >&2  echo "model directory not found: $2"
  exit 1
fi

export SENT_CSV=$1
export CLF_MODEL=$2

echo "starting..."
echo ""

python $LIT \
  --model_path $CLF_MODEL \
  --data_path $SENT_CSV \
  --num_labels 3 \
  --batch_size 8 \
  --max_seq_len 128 \
  --port 5432
