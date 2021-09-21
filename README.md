# 🔥 `classifier-lit`
## The Language Interpretability Tool (LIT) for Text Classification

This app is an implementation of the
[Language Interpretability Tool](https://pair-code.github.io/lit/) for text classification.
This was assembled from examples in LIT to provide an easy-to-use way to
try various features of LIT.

Contributions are welcome.

## Requirements
```
pip install -r requirements.txt
```

## Model
A model name or path for a `transformers` SequenceClassification model. 
This can be a path to your (`pytorch`) trained model or the name an appropriate
model from [HuggingFace models](https://huggingface.co/models). You will need
to know the number of classification labels in your model.

### Data
The data is a `.csv`,  ideally consisting of validation data, with columns
```
"src", "label", "sentence"
```
`src` is a user-specific identifier, `label` is the validation label, and `sentence` is the text to be classified.
Modify `clf_dataset.py` as appropriate to conform to the expected format.

If the validation label is not known, use 0 (zero). 
The metrics will be meaningless, but all the other LIT features are
available.

## Starting the Server

The easiest way is to use bash script `clf_lit.sh`. 
The script takes two arguments, the data `.csv` and the model name or path:

```bash
./start_lit.sh text.csv model-name-or-path
```

The default content root is `$HOME/classifier-lit`. Update as required
for your environment.

```bash
python clf_lit.py \
    --data_path data.csv \
    --model_path model-name-or-path \
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

## Notebook Example

## Using a GPU
A GPU is automatically detected and used.
If your GPU instance is a remote, headless cloud instance, you can still
use your local browser by using port forwarding feature of SSH. For port 5432:

```
ssh -i access-creds.pem -L 5432:localhost:5432 <your id>@<remote IP address>
```

Start the server on the remote and view the results in your local browser.

## Usage
```
usage: clf_server.py [-h] --model_path MODEL_PATH --data_path DATA_PATH 
                     [--label_text_cols LABEL_TEXT_COLS] 
                     [--batch_size BATCH_SIZE] [--max_seq_len MAX_SEQ_LEN]
                     [--port PORT] [--notebook] [--height HEIGHT]

Start the LIT server

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        tar.gz, name, or directory of the pytorch model
  --data_path DATA_PATH
                        path + file.csv, for the data .csv
  --label_text_cols LABEL_TEXT_COLS
                        python-style list of the label index and text index in 
                        the .csv, default=[0,1]
  --batch_size BATCH_SIZE
                        batch size, default 8
  --max_seq_len MAX_SEQ_LEN
                        maximum sequence length up to 512, default=128
  --port PORT           LIT server port, default=5432
  --notebook            LIT widget for Jupyter notebooks
  --height HEIGHT       height for the rendered notebook widget
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

lit-nlp is licensed under the Apache License Version 2.0