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
from training_util import save_model, setup_logging
from multitask_model import MultitaskModel
from data_util import detect_last_checkpoint, load_datasets

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

import wandb


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
    
    wandb.init(mode="disabled")

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
    
    setup_logging(logger=logger, training_args=training_args)

    # Detecting last checkpoint.
    last_checkpoint = detect_last_checkpoint(training_args)
    
    raw_datasets = load_datasets(model_args, data_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    sum_tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    sim_tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    multitask_model = MultitaskModel.create(
        model_name=model_args.model_name_or_path,
        target_lang_id=sum_tokenizer.lang_code_to_id[data_args.tgt_lang], 
        source_lang_id=sim_tokenizer.lang_code_to_id[data_args.src_lang],
        lambda_=data_args.lambda_
    )
    
    # Resizes input token embeddings matrix of the model if new_num_tokens != config.vocab_size
    multitask_model.resize_token_embeddings(len(sum_tokenizer))

    for model in multitask_model.model_list:
        if model.config.decoder_start_token_id is None and isinstance(sum_tokenizer, (MBartTokenizer, MBartTokenizerFast)):
            # set decoder_start_token_id to generate the correct language during inference 
            # only for the summarization model because the simplification model is not used during inference at this moment
            if isinstance(sum_tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = sum_tokenizer.lang_code_to_id[data_args.tgt_lang]
            else:
                model.config.decoder_start_token_id = sum_tokenizer.convert_tokens_to_ids(data_args.tgt_lang)


        if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
        ):
            if model_args.resize_position_embeddings is None:
                logger.warning(
                    "Increasing the model's number of position embedding vectors from"
                    f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
                )
                model.resize_position_embeddings(data_args.max_source_length)
            elif model_args.resize_position_embeddings:
                model.resize_position_embeddings(data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                    f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                    f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                    " model's position encodings by passing `--resize_position_embeddings`."
                )


    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(sum_tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.src_lang is not None
        ), f"{sum_tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        sum_tokenizer.src_lang = data_args.src_lang
        sum_tokenizer.tgt_lang = data_args.tgt_lang # for summarization the target language should different from the input language
        
        sim_tokenizer.src_lang = data_args.src_lang
        sim_tokenizer.tgt_lang = data_args.src_lang # for simplification the target language should be same as input language

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        german_forced_bos_token_id = sum_tokenizer.lang_code_to_id["de_DE"]
        multitask_model.sum_model.config.forced_bos_token_id = german_forced_bos_token_id
        multitask_model.config.forced_bos_token_id = german_forced_bos_token_id
    # Get the column names for input/target.
    if data_args.text_column is None:
        text_column = "text"
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = "summary"
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.simple_column is None:
        simple_column = "simple"
    else:
        simple_column = data_args.simple_column
        if simple_column not in column_names:
            raise ValueError(
                f"--simple_column' value '{data_args.simple_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    for model in multitask_model.model_list:
        if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

    def preprocess_function_train(examples):
        """The preprocess function for the train dataset, where simplification targets are included.    

        Args:
            examples (Dataset): a batch of examples containing the data

        Returns:
            model_inputs (Dict): a dict containing the input for the models forward function, containing:
            {
                "input_ids": ...,
                "attention_mask": ...,
                "labels": ...,
                "sim_labels: ...
                
            }
        """
        # remove pairs where at least one record is None
        inputs, sum_targets, sim_targets = [], [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i] and examples[simple_column][i]:
                inputs.append(examples[text_column][i])
                sum_targets.append(examples[summary_column][i])
                sim_targets.append(examples[simple_column][i])

        model_inputs = sum_tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = sum_tokenizer(text_target=sum_targets, max_length=max_target_length, padding=padding, truncation=True)
        sim_labels = sim_tokenizer(text_target=sim_targets, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        for lab in [labels, sim_labels]:
            if padding == "max_length" and data_args.ignore_pad_token_for_loss:
                lab["input_ids"] = [
                    [(l if l != sum_tokenizer.pad_token_id else -100) for l in label] for label in lab["input_ids"]
                ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["sim_labels"] = sim_labels["input_ids"]
        return model_inputs
    
    def preprocess_function_eval(examples):
        """The preprocess function for the test dataset, where only summarization targets are included

        Args:
            examples (Dataset): a batch of examples containing the data

        Returns:
            model_inputs (Dict): a dict containing the input for the models forward function, containing:
            {
                "input_ids": ...,
                "attention_mask": ...,
                "labels": ...,
            }
        """
        # remove pairs where at least one record is None
        inputs, sum_targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i] and examples[simple_column][i]:
                inputs.append(examples[text_column][i])
                sum_targets.append(examples[summary_column][i])
        model_inputs = sum_tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        labels = sum_tokenizer(text_target=sum_targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != sum_tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

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

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

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
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
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
        
        save_model(model_args.model_name_or_path, multitask_model, training_args.output_dir)
        

    def get_references(dataset):
        references = []
        for i in range(len(dataset[summary_column])):
            if dataset[summary_column][i]:
                references.append(dataset[summary_column][i])
        return references
    
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
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
                references = get_references(raw_datasets["test"])
                data = {"system": predictions,
                        "reference": references}
                df = pd.DataFrame(data)
                filepath = Path(data_args.test_output_path)  
                filepath.parent.mkdir(parents=True, exist_ok=True)  
                df.to_csv(filepath)

    return results

if __name__ == "__main__":
    main()
