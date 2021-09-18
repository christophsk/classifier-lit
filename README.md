# 🔥 `classifier-lit`
## The Language Interpretability Tool (LIT) for Text Classification

This package provides an implementation of the PAIR code 
[Language Interpretability Tool](https://pair-code.github.io/lit/) for a sentence
classification. It builds off the various LIT examples to provide a convenient
way to run the LIT server.

## Requirements
```
pip install -r requirements.txt
```

## Model and Data
### Model
You're required to have a local `pytorch` SequenceClassification model you've trained on labels in
`range(num_labels)`. The model's directory should conform to the usual `pytorch` layout.

### Data
The data is a `.csv` ideally consisting of validation data, with columns
```
"src", "label", "sentence"
```
`src` is a user-specific identifier, `label` is the validation label, and `sentence` is the text to be classified.

If the validation label is not known, any value in `range(num_labels)` can be used (e.g., 0). 
The metrics will be meaningless, but all the other LIT features are
available.

## Starting the Server

The easiest way is to use bash script `clf_lit.sh`. 
The script takes three arguments, the `.csv` and the model directory:

```bash
./clf_lit.sh sentences.csv model_directory
```

The default content root is `$HOME/classifier-lit` and can be changed in the script as can
the other defaults (shown below).

Or,

```bash
python clf_lit.py \
    --data_path sentences.csv \
    --model_path model_directory \
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


### Running with a GPU
Inference on a even a moderate set of sentences can be computationally intensive and benefits from
GPU assistance. Typically, this is a remote, headless cloud instance. 

You can use your local browser by forwarding the port from the remote using SSH:

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
MIT License Copyright (c) 2021 Chris Skiscim

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