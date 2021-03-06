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
# lit-nlp is licensed under the Apache License Version 2.0
import logging
import os

import pandas as pd
import pandas.errors as pd_err
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types
from typing import List

logger = logging.getLogger(__name__)


class SeqDataset(lit_dataset.Dataset):
    def __init__(
        self, data_path: str, labels: List[str], lbl_txt_cols: List[int]
    ):
        super().__init__()
        if not os.path.isfile(data_path):
            raise FileNotFoundError(data_path)

        self.LABELS = labels
        self._examples = list()

        df = self.read_data(data_path, lbl_txt_cols)

        logger.info("         rows: {:>4,d}".format(len(df)))
        logger.info("unique labels: {:>4,d}".format(len(df.label.unique())))

        self._examples = [
            {
                "text": row["text"],
                "label": row["label"],
            }
            for _, row in df.iterrows()
        ]

    def spec(self):
        return {
            "text": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(vocab=self.LABELS),
        }

    @staticmethod
    def read_data(data_path, lbl_txt_cols):
        try:
            if lbl_txt_cols[0] > lbl_txt_cols[1]:
                columns = ["text", "label"]
            else:
                columns = ["label", "text"]

            df = pd.read_csv(
                data_path,
                delimiter=",",
                header=None,
                usecols=lbl_txt_cols,
                names=columns,
            )
            return df
        except (pd_err.ParserError, pd_err.EmptyDataError) as e:
            raise e
