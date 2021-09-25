#!/bin/bash

export PROJECT=../classifier-lit

# data
export DATA=$PROJECT/classifier_lit/static/data.csv
export COLS=[1,0]

# model params
export MODEL=bhadresh-savani/distilbert-base-uncased-emotion
export PORT=5432
export BSZ=8
export MAX_LEN=128

export LIT=$PROJECT/classifier_lit/seq_server.py

echo ""
echo "starting LIT server, model $MODEL"
export PYTHONPATH=.
python $LIT \
  --model_path $MODEL \
  --data_path $DATA \
  --label_text_cols $COLS \
  --batch_size $BSZ \
  --max_seq_len $MAX_LEN \
  --port $PORT