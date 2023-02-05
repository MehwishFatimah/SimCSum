import os
import transformers
import torch
from transformers import AutoModelForSeq2SeqLM
from multitask_model import MultitaskModel
import logging
import sys
import datasets
from transformers import AutoTokenizer

training_util_logger = logging.getLogger(__name__)



def setup_logging(training_args, logger):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
def init_tokenizers(model_args, data_args):
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
    sum_tokenizer.src_lang = data_args.src_lang
    sum_tokenizer.tgt_lang = data_args.main_tgt_lang # for summarization the target language should different from the input language
        
    sim_tokenizer.src_lang = data_args.src_lang
    sim_tokenizer.tgt_lang = data_args.aux_tgt_lang # for simplification the target language should be same as input language
    return sum_tokenizer, sim_tokenizer

def resize_model_position_embeddings(model_list, model_args, data_args):
    for model in model_list:
        if (
                hasattr(model.config, "max_position_embeddings")
                and model.config.max_position_embeddings < data_args.max_source_length
            ):
                if model_args.resize_position_embeddings is None:
                    training_util_logger.warning(
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
                    
def check_label_smoothing(model_list, training_args):
    for model in model_list:
        if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            training_util_logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

def save_model(model_name, multitask_model, output_dir):
    """Save the multitask model by saving both submodels via pytorch in the output_dir.

    Args:
        model_name (String): The name of the model to instantiate a tokenizer 
        multitask_model (MultitaskModel): The instance of the multitask_model we want to save
        output_dir (String): The output directory of the multitask_model we want to save
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    for task_name, model in zip(["sum", "sim"], [multitask_model.sum_model, multitask_model.sim_model]):
        model.config.to_json_file(
            f"{output_dir}/{task_name}_model/config.json"
        )
        torch.save(
            model.state_dict(),
            f"{output_dir}/{task_name}_model/pytorch_model.bin",
        )
        tokenizer.save_pretrained(f"./{task_name}_model/")
        
        
def load_multitask_model(model_path):
    """Supposed to load a trained multitask model. Not yet implemented in run_summarization.py

    Args:
        model_path (String): The path to the trained multitask model

    Returns:
        MultitaskModel: a multitask model with the weights loaded from the pytorch state_dict
    """
    try:
        sum_model = load_task_model(model_path, "sum")
        sim_model = load_task_model(model_path, "sim")
        encoder = sum_model.get_encoder()
        return MultitaskModel(encoder=encoder, model_1=sum_model, model_2=sim_model)
    except FileNotFoundError:
        print("Path to load models from does not exist.")
        
        
def load_task_model(model_path, model_name):
    """Load a single task model to use in the multitask model.

    Args:
        model_path (String): The path to the trained multitask model
        model_name (String): The name of the task. In our case "sum" or "sim". The model_path should contain directories with the name {model_name}_model.

    Returns:
        Model: the single task model to use with the multitask model
    """
    config_path = os.path.join(model_path, f"/{model_name}_model/config.json")
    state_dict_path = os.path.join(model_path, f"/{model_name}_model/pytorch_model.bin")
    config = torch.load(config_path)
    state_dict = torch.load(state_dict_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(config=config, state_dict=state_dict)
    return model

def get_max_length_for_generation(training_args, data_args):
    """
    Get the max length for generation, specified in the training args or the data args.
    """
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    return max_length

def get_num_beams_form_generation(training_args, data_args):
    """
    Get the number of beams for generation, specified in the training args or the data args.
    """
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    return num_beams