# !/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

from training_arguments import ModelArguments, DataTrainingArguments
import training_util
from training_util import init_tokenizers
from multitask_model import MultitaskModel
from data_util import detect_last_checkpoint, postprocess_text, DataPreprocessing

import logging
import os
import sys
from pathlib import Path
import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import pandas as pd
import transformers
from filelock import FileLock
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser,
                          MBart50Tokenizer, MBart50TokenizerFast,
                          MBartTokenizer, MBartTokenizerFast, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, set_seed, EarlyStoppingCallback)
from transformers.utils import (check_min_version, is_offline_mode,
                                send_example_telemetry)
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)

    training_util.setup_logging(logger=logger, training_args=training_args)

    # Detecting last checkpoint.
    last_checkpoint = detect_last_checkpoint(training_args)
    
    # Set seed before initializing model.
    set_seed(training_args.seed)

    sum_tokenizer, sim_tokenizer = init_tokenizers(model_args, data_args)

    multitask_model = MultitaskModel.create(
        model_name=model_args.model_name_or_path,
        main_target_lang_id=sum_tokenizer.lang_code_to_id[data_args.main_tgt_lang], 
        aux_target_lang_id=sim_tokenizer.lang_code_to_id[data_args.aux_tgt_lang],
        max_length=data_args.max_source_length,
        lambda_=data_args.lambda_
    )
    
    # Load datasets
    preprocessor = DataPreprocessing(sum_tokenizer, sim_tokenizer, model_args, data_args, training_args)
    raw_datasets = preprocessor.get_raw_datasets
    
    # Resizes input token embeddings matrix of the model if new_num_tokens != config.vocab_size
    multitask_model.resize_token_embeddings(len(sum_tokenizer))

    training_util.resize_model_position_embeddings(multitask_model.model_list, model_args, data_args)
    training_util.check_label_smoothing(multitask_model.model_list, training_args)
    
    # Run the preprocessing functions
    train_dataset, eval_dataset, predict_dataset = preprocessor.run_preprocessing()
    
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else sum_tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        sum_tokenizer,
        model=multitask_model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Metric
    metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # TODO: Fix Model predicting -100 which leads to errors during decoding, this is a workaround
        preds = np.where(preds != -100, preds, sum_tokenizer.pad_token_id)
        decoded_preds = sum_tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, sum_tokenizer.pad_token_id)
        decoded_labels = sum_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != sum_tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=multitask_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=sum_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[EarlyStoppingCallback(3)]
    )
    

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        training_util.save_model(model_args.model_name_or_path, multitask_model, training_args.output_dir)
        
    results = {}

    max_length = training_util.get_max_length_for_generation(training_args, data_args)
    num_beams = training_util.get_num_beams_for_generation(training_args, data_args)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_new_tokens=max_length, num_beams=num_beams, metric_key_prefix="eval", 
                                   decoder_start_token_id=sum_tokenizer.lang_code_to_id["de_DE"], 
                                   forced_bos_token_id=sum_tokenizer.lang_code_to_id["de_DE"])
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_new_tokens=max_length, num_beams=num_beams, 
                                   decoder_start_token_id=sum_tokenizer.lang_code_to_id["de_DE"], 
                                   forced_bos_token_id=sum_tokenizer.lang_code_to_id["de_DE"])
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = sum_tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                max_predict_samples = min(len(raw_datasets["test"]), data_args.max_predict_samples)
                raw_datasets["test"] = raw_datasets["test"].select(range(max_predict_samples))
                references = preprocessor.get_references(raw_datasets["test"])
                data = {"system": predictions,
                        "reference": references}
                df = pd.DataFrame(data)
                filepath = Path(data_args.test_output_path)  
                filepath.parent.mkdir(parents=True, exist_ok=True)  
                df.to_csv(filepath)

    return results

if __name__ == "__main__":
    main()
