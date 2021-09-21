#!/bin/bash

export PROJECT=../classifier-lit

export DATA=$PROJECT/classifier_lit/examples/data.csv

# [label_column, text_column]
export COLS=[1,0]
export MODEL=bhadresh-savani/distilbert-base-uncased-emotion

export LIT=$PROJECT/classifier_lit/clf_server.py
export PORT=5432
export BSZ=8
export MAX_LEN=128

echo "starting LIT..."
echo ""

export PYTHONPATH=.
python $LIT \
  --model_path $MODEL \
  --data_path $DATA \
  --label_text_cols $COLS \
  --batch_size $BSZ \
  --max_seq_len $MAX_LEN \
  --port $PORT