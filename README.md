# `classifier-lit`
This package provides an implementation of the Language Interpretability Tool (LIT) for
a `pytorch` sentence classifier.  Additional information on LIT can be 
found at [pair-code](https://pair-code.github.io/lit/). This is released under the MIT License.

## Requirements
```
pip install -r requirements.txt
```

## Model and Data
The implementation assumes you have a trained PyTorch text classifier with labels 0,...,num_labels. 
The model directory should confirm to the usual layout of PyTorch models.

The data is a `.csv` with columns
```
"src", "label", "sentence"
```
where *src* is a user-specific identifier, `sentence` is the text to be classified. The `label` column
is the expected classification label and is used to calculate various metrics. If the expected label is not
known, a value of 0 can be used. The metrics will be useless, but everything else will work.

## Starting the LIT Server

```bash
python lit.py \
    --model_path ~/your_model_directory \
    --data_path ~/your_sentences.csv \
    --num_labels 3 \
    --batch_size 8 \
    --max_seq_len 128 \
    --port 5432
```
After a short wait, you should see
```
I0604 16:10:49.734775 139835092129600 classifier_lit.py:75] model loaded
I0604 16:10:49.742584 139835092129600 gc_dataset.py:22] rows : 371
I0604 16:10:49.805324 139835092129600 dev_server.py:88]
 (    (
 )\ ) )\ )  *   )
(()/((()/(` )  /(
 /(_))/(_))( )(_))
(_)) (_)) (_(_())
| |  |_ _||_   _|
| |__ | |   | |
|____|___|  |_|


I0604 16:10:49.805436 139835092129600 dev_server.py:89] Starting LIT server...
I0604 16:10:49.805539 139835092129600 caching.py:125] CachingModelWrapper 'distilbert': no cache path specified, not loading.
I0604 16:10:49.810200 139835092129600 gradient_maps.py:120] Skipping token_grad_sentence since embeddings field not found.
I0604 16:10:49.810353 139835092129600 gradient_maps.py:235] Skipping token_grad_sentence since embeddings field not found.
I0604 16:10:49.810859 139835092129600 wsgi_serving.py:43]

Starting Server on port 5432
You can navigate to 127.0.0.1:5432


I0604 16:10:49.811561 139835092129600 _internal.py:122]  * Running on http://127.0.0.1:5432/ (Press CTRL+C to quit)
``` 
Point your browser to `127.0.0.1:5432` to view the results.


### Using a GPU
Inference on a even a moderate set of sentences can be computationally intensive and benefits from
GPU assistance. Typically, this is a headless cloud instance. 

In this case, use SSH with port forwarding, e.g.,
```
ssh -i access-creds.pem -L 5432:localhost:5432 <your id>@<remote IP address>
```
Start the server on the remote and view the results in your local browser.

## Usage
```
usage: python clf_lit.py [-h] [--absl_flags ABSL_FLAGS] --model_path
                                MODEL_PATH --data_path DATA_PATH --num_labels
                                NUM_LABELS [--batch_size BATCH_SIZE]
                                [--max_seq_len MAX_SEQ_LEN] [--port PORT]

Start the LIT server

optional arguments:
  -h, --help            show this help message and exit
  --absl_flags ABSL_FLAGS
                        absl flags - defaults to []
  --model_path MODEL_PATH
                        directory of the pytorch model or pretrained name
  --data_path DATA_PATH
                        path + file.csv, for input data .csv
  --num_labels NUM_LABELS
                        number of labels in the classification model
  --batch_size BATCH_SIZE
                        batch size, default 8
  --max_seq_len MAX_SEQ_LEN
                        maximum sequence length up to 512, default 128
  --port PORT           LIT server port, default 5432
```

# License
MIT License Copyright (c)  2021 Chris Skiscim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE