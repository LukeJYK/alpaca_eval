import logging
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import fire
import pandas as pd
import json
from . import analyze, annotators, constants, decoders, metrics, utils
from .types import AnyData, AnyLoadableDF, AnyPath

CUR_DIR = Path(__file__).parent

__all__ = ["evaluate"]


def evaluate(
    input_path: str=None,
    annotators_config: AnyPath = constants.DEFAULT_ANNOTATOR_CONFIG,
    output_path: str = None,
    max_instances: Optional[int] = None,
    annotation_kwargs: Optional[dict[str, Any]] = None,
    Annotator=annotators.PairwiseAnnotator,
    disable_shuffling: bool = False,
    summary_path: str = "summary.json",
    **annotator_kwargs,
):
    """Evaluate a model based on its outputs. This is the default entrypoint if no command is specified.

    Parameters
    ----------
    model_outputs : path or data or dict
        The outputs of the model to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
        (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction`
        and `output` with optional `input`. If None, we just print the leaderboard.

    reference_outputs : path or data, optional
        The outputs of the reference model. Same format as `model_outputs`. If None, the reference outputs are a
        specific set of Davinci 003 outputs on the AlpacaEval set:
        https://huggingface.co/datasets/tatsu-lab/alpaca_eval.

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file. For details see the docstring of
        `PairwiseAnnotator`.

    output_path : path, optional
        Path to the directory where the new leaderboard and the annotations should be stored. If None we don't save.
        If `auto` we use `model_outputs` if it is a path, and otherwise use the directory from which we call the script.

    max_instances : int, optional
        The maximum number of instances to annotate. Useful for testing.

    annotation_kwargs : dict, optional
        Additional arguments to pass to `PairwiseAnnotator.annotate_head2head`.

    Annotator : class, optional
        The annotator class to use.

    annotator_kwargs :
        Additional arguments to pass to `PairwiseAnnotator`.
    """
    annotation_kwargs = annotation_kwargs or dict()

    annotations = None
    if input_path is not None:
        inputs = utils.load_or_convert_to_dataframe(input_path)
        names = utils.get_generator_name(inputs)
        print("evaluating the following generators:",names)
        #need to fix max_instance function
        if max_instances is not None:
            # first we shuffle both outputs with a fix seed => more representative
            if utils.check_length(inputs)==False:
                logging.warning(
                    "model_outputs and reference_outputs have different lengths, so we cannot shuffle before taking the first max_instances."
                )
            else:
                seed = 123
                model_outputs = model_outputs.sample(frac=1, random_state=seed)
                reference_outputs = reference_outputs.sample(frac=1, random_state=seed)

            model_outputs = model_outputs[:max_instances]
            reference_outputs = reference_outputs[:max_instances]
        keys =["instruction",]
        for i in range(len(names)):
            keys.append("output_{}".format(i+1))
        annotator = Annotator(annotators_config=annotators_config,primary_keys=keys, **annotator_kwargs)
        annotations = annotator.annotate_head2head(
            outputs=inputs, 
            disable_shuffling=disable_shuffling,
            **annotation_kwargs
            )
        
        if annotations is not None:
            if output_path == None:
                utils.convert_to_dataframe(annotations).to_json(
                "test.json", orient="records", indent=2
                )
            else:
                utils.convert_to_dataframe(annotations).to_json(
                output_path, orient="records", indent=2
                )
            summary = {}
            for item in annotations:
                winner = item["raw_completion"][8]
                dataset = item["dataset_1"]
                winner_key = f"generator_{winner}"
                if "brief_" in item[winner_key]:
                    winner_prompt = 1
                elif "brief+_" in item[winner_key]:
                    winner_prompt = 2
                else:
                    winner_prompt = 0
                if dataset not in summary:
                    summary[dataset] = []
                summary[dataset].append({
                    "instruction": item["instruction"],
                    "assistance_lvl": winner_prompt,
                })
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=4)
    #output_path = utils.get_output_path(output_path, arg_model_outputs, name)


    # if output_path is not None:
    #     if isinstance(annotators_config, str) and "/" not in annotators_config:
    #         output_path = Path(output_path) / annotators_config
    #         output_path.mkdir(exist_ok=True, parents=True)
    #     logging.info(f"Saving all results to {output_path}")
    #     if annotations is not None:
    #         utils.convert_to_dataframe(annotations).to_json(
    #             output_path / "annotations.json", orient="records", indent=2
    #         )
ALL_FUNCTIONS = {
    "evaluate": evaluate
}


def main():
    is_fn_name = len(sys.argv) > 1 and "--" not in sys.argv[1]
    is_help = any(a == "--help" for a in sys.argv)

    if is_fn_name or is_help:
        fire.Fire(ALL_FUNCTIONS)
    else:
        # default behavior if no function is specified
        fire.Fire(evaluate)


if __name__ == "__main__":
    fire.Fire(ALL_FUNCTIONS)
