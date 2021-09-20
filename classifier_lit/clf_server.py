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
#
# This was assembled from various examples in
# https://github.com/PAIR-code/lit
# lit-nlp is licensed under the Apache License Version 2.0
import os

import transformers as trf
from absl import app
from absl import flags
from absl import logging
from lit_nlp import dev_server
from lit_nlp import notebook
from lit_nlp import server_flags

from classifier_lit.clf_dataset import ClfDataset
from classifier_lit.clf_lit import TextClassifier

FLAGS = flags.FLAGS


def main(_):
    data_csv = FLAGS.data_path
    model_path = FLAGS.model_path

    # TODO test .tar.gz model files
    try:
        model_path = trf.file_utils.cached_path(
            model_path, extract_compressed_file=True
        )
    except (OSError, EnvironmentError):
        pass

    if not os.path.isfile(data_csv):
        raise FileNotFoundError(data_csv)

    lit_class = TextClassifier(model_path)
    models = {"classifier": lit_class}
    datasets = {"data": ClfDataset(data_csv, lit_class.LABELS)}

    # direct LIT to a notebook or start the server
    if FLAGS.notebook:
        return notebook.LitWidget(models, datasets, height=FLAGS.height)
    else:
        lit_server = dev_server.Server(
            models, datasets, **server_flags.get_flags()
        )
        lit_server.serve()


if __name__ == "__main__":
    from argparse import ArgumentParser

    log_fmt = "[%(asctime)s%(levelname)8s], [%(filename)s:%(lineno)s "
    log_fmt += "- %(funcName)s()], %(message)s"
    logger = logging.get_absl_logger()
    logger.setLevel(logging.INFO)

    parser = ArgumentParser(
        prog=os.path.split(__file__)[-1], description="Start the LIT server"
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        required=True,
        help="tar.gz, name, or directory of the pytorch model",
    )
    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=str,
        required=True,
        help="path + file.csv, for the data .csv",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=str,
        required=False,
        default=8,
        help="batch size, default 8",
    )
    parser.add_argument(
        "--max_seq_len",
        dest="max_seq_len",
        type=int,
        required=False,
        default=128,
        help="maximum sequence length up to 512, default 128",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=5432,
        help="LIT server port, default 5432",
    )
    parser.add_argument(
        "--notebook",
        dest="notebook",
        action="store_true",
        help="LIT widget for Jupyter notebooks",
    )
    parser.add_argument(
        "--height",
        dest="height",
        type=int,
        default=800,
        required=False,
        help="height for the notebook widget",
    )

    parser.add_help = True
    args_in = parser.parse_args()
    flags.DEFINE_string("model_path", args_in.model_path, "saved model")
    flags.DEFINE_string("data_path", args_in.data_path, "validation data")
    flags.DEFINE_integer("batch_size", args_in.batch_size, "batch size")
    flags.DEFINE_integer("max_seq_len", args_in.max_seq_len, "max seq length")
    flags.DEFINE_bool("notebook", args_in.notebook, "notebook widget")
    flags.DEFINE_integer("height", args_in.height, "height if in a notebook")
    flags.port = args_in.port
    flags.absl_flags = []

    app.run(main)
