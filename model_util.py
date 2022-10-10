import transformers
import torch


def save_model(model_name, multitask_model, output_dir):
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