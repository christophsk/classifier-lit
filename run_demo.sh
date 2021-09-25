#!/bin/bash

export PROJECT=../classifier-lit

# Data
export DATA=$PROJECT/classifier_lit/static/data.csv
export COLS=[1,0]

# Server
export MODEL=bhadresh-savani/distilbert-base-uncased-emotion
export PORT=5432
export BSZ=8
export MAX_LEN=128

export LIT=$PROJECT/classifier_lit/clf_server.py

echo "starting LIT..."
export PYTHONPATH=.
python $LIT \
  --model_path $MODEL \
  --data_path $DATA \
  --label_text_cols $COLS \
  --batch_size $BSZ \
  --max_seq_len $MAX_LEN \
  --port $PORT