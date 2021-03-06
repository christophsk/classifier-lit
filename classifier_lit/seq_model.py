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
import re

import torch
import torch.nn.functional as F
import transformers as trf
from absl import flags
from absl import logging
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
from transformers.modeling_outputs import SequenceClassifierOutput

FLAGS = flags.FLAGS
logger = logging.get_absl_logger()


def _from_pretrained(cls, *args, **kw):
    try:
        return cls.from_pretrained(*args, **kw)
    except (OSError, EnvironmentError) as e:
        raise e


class SeqModel(lit_model.Model):
    compute_grads: bool = True

    def __init__(
        self,
        model_name_or_path=None,
    ):
        """
        Retrieve the model and data in preparation for inference.

        Args:
            model_name_or_path (str): directory containing the `pytorch`
                model or the name of a HuggingFace model. If the model
                name is not in the local cache (~/.cache), it will be
                downloaded, if available.

        Raises:
            OSError, EnvironmentError: If the model_name_or_path cannot
                be loaded

        """
        # TODO add support for label names in place of integers
        try:
            self.tokenizer = trf.AutoTokenizer.from_pretrained(
                model_name_or_path
            )
            model_config = trf.AutoConfig.from_pretrained(
                model_name_or_path,
                output_hidden_states=True,
                output_attentions=True,
            )
            self.model = _from_pretrained(
                trf.AutoModelForSequenceClassification,
                model_name_or_path,
                config=model_config,
            )
            if torch.cuda.is_available():
                self.model.cuda()

            self.id2label = model_config.id2label
            self.LABELS = list(map(str, self.id2label.keys()))
            self.architectures = ",".join(model_config.architectures)
            logging.info(
                "{} loaded for {} labels".format(
                    self.architectures, len(self.LABELS)
                )
            )
        except (OSError, EnvironmentError) as e:
            logger.exception(
                "failed to load the model - {}: {}".format(type(e), str(e))
            )
            raise e
        self.model.eval()

    # LIT API implementation
    def max_minibatch_size(self):
        return FLAGS.batch_size

    def predict_minibatch(self, inputs):
        encoded_input = self.tokenizer.batch_encode_plus(
            [ex["text"] for ex in inputs],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=FLAGS.max_seq_len,
            padding="longest",
            truncation="longest_first",
        )

        if torch.cuda.is_available():
            self.model.cuda()
            for tensor in encoded_input:
                encoded_input[tensor] = encoded_input[tensor].cuda()

        # Run a forward pass with gradient.
        with torch.set_grad_enabled(self.compute_grads):
            out: SequenceClassifierOutput = self.model(**encoded_input)

        # Post-process outputs.
        batched_outputs = {
            "probas": F.softmax(out.logits, dim=-1),
            "input_ids": encoded_input["input_ids"],
            "ntok": torch.sum(encoded_input["attention_mask"], dim=1),
            "cls_emb": out.hidden_states[-1][:, 0],  # last layer, first token
        }
        # Add attention layers to batched_outputs
        assert len(out.attentions) == self.model.config.num_hidden_layers
        for i, layer_attention in enumerate(out.attentions):
            batched_outputs[f"layer_{i}/attention"] = layer_attention

        # Request gradients after the forward pass. Note: hidden_states[0]
        # includes position and segment encodings, as well as sub-word
        # embeddings.
        if self.compute_grads:
            scalar_pred_for_gradients = torch.max(
                batched_outputs["probas"], dim=1, keepdim=False, out=None
            )[0]

            batched_outputs["input_emb_grad"] = torch.autograd.grad(
                scalar_pred_for_gradients,
                out.hidden_states[0],
                grad_outputs=torch.ones_like(scalar_pred_for_gradients),
            )[0]

        # Return as numpy for further processing. NB: cannot use
        # v.cpu().numpy() when gradients are computed.
        detached_outputs = {
            k: v.cpu().detach().numpy() for k, v in batched_outputs.items()
        }

        # un-batch outputs so we get one record per input example.
        for output in utils.unbatch_preds(detached_outputs):
            ntok = output.pop("ntok")
            output["tokens"] = self.tokenizer.convert_ids_to_tokens(
                output.pop("input_ids")[1 : ntok - 1]
            )
            # set token gradients - exclude special tokens
            if self.compute_grads:
                output["token_grad_sentence"] = output["input_emb_grad"][
                    1 : ntok - 1
                ]

            # Process attention.
            for key in output:
                if not re.match(r"layer_(\d+)/attention", key):
                    continue
                # Select only real tokens, since most of this matrix is padding
                output[key] = output[key][:, :ntok, :ntok].transpose((0, 2, 1))
                # Make a copy of this array to avoid memory leaks, since NumPy
                # otherwise keeps a pointer around that prevents the source
                # array from being GC'd.
                output[key] = output[key].copy()
            yield output

    def input_spec(self) -> lit_types.Spec:
        return {
            "text": lit_types.TextSegment(),
            "label": lit_types.CategoryLabel(
                vocab=self.LABELS, required=False
            ),
        }

    def output_spec(self) -> lit_types.Spec:
        ret = {
            "tokens": lit_types.Tokens(),
            "probas": lit_types.MulticlassPreds(
                parent="label", vocab=self.LABELS
            ),
            "cls_emb": lit_types.Embeddings(),
        }
        # Gradients, if requested.
        if self.compute_grads:
            ret["token_grad_sentence"] = lit_types.TokenGradients(
                align="tokens"
            )

        # Attention heads, one field for each layer.
        for i in range(self.model.config.num_hidden_layers):
            ret[f"layer_{i}/attention"] = lit_types.AttentionHeads(
                align_in="tokens", align_out="tokens"
            )
        return ret

    def fit_transform_with_metadata(self, indexed_inputs):
        raise NotImplementedError

    def get_embedding_table(self):
        raise NotImplementedError
