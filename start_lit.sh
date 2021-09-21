#!/bin/bash
#------------------------------------------------------------------------------
# start_lit: Starts the LIT server from the content root
#
# Arguments:
#   data: .csv with label and text columns
#   cols: python-style list with column indexes of the label and text,
#         e.g., [1,2] (no spaces)
#   model_name_or_path: directory of the pytorch model or the name
#         of a HuggingFace SequenceClassification model
#------------------------------------------------------------------------------

# !!! UPDATE FOR YOUR ENVIRONMENT!!!
export PROJECT=$HOME/pycharm_projects/classifier-lit

if ! [ -d $PROJECT ]; then
  >&2 "Cannot find classifier_lit at $PROJECT"
  >&2 "Update PROJECT in this script for your environment"
  exit 1
fi

if (( $# != 3 )); then
    >&2 echo "Usage: start_lit <data_file.csv> <csv_cols> <model_file>"
    if (( $# > 3 )); then
      >&2 echo "Make sure there are no spaces in each of the arguments"
    fi
    >&2 echo ""
    exit 1
fi

# Defaults for inference
export PORT=5432
export BSZ=8
export MAX_LEN=128

#-------------------------------------------------
# You shouldn't need to change anything below here
#-------------------------------------------------
export DATA=$1
export COLS=$2
export MODEL=$3

export PYTHONPATH=.
export LIT=$PROJECT/classifier_lit/clf_server.py


if ! [ -f $LIT ]; then
  >&2 echo "Cannot find clf_server.py at $LIT"
  exit 1
fi

if ! [ -f $DATA ]; then
  >&2 echo "Data file not found at $DATA"
  exit 1
fi

echo "starting LIT..."
echo ""

python $LIT \
  --model_path $MODEL \
  --data_path $DATA \
  --label_text_cols $COLS \
  --batch_size $BSZ \
  --max_seq_len $MAX_LEN \
  --port $PORT \

# MIT License
# Copyright (c) 2021 Chris Skiscim
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.