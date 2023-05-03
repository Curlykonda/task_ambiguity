"""
Generate a set of AmbiBench-style examples based on the given configuration.
Instead of running the inference pipeline, these examples are stored in a JSON file.

"""

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List

from src.structures.api_access import OpenAI_APIAccess
from src.structures.category import GenerationCategories
from src.structures.construction_types import ConstructionType
from src.structures.prompt import Prompt

logger = logging.getLogger("GenerateDataset")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.INFO,
)


@dataclass
class AmbiBenchConfig:

    construction_format: str
    n_shots: int
    n_queries: int
    n_multiple_choices: int
    prob_of_ambiguous: float

    needs_instruction: bool = False
    needs_informative: bool = False
    no_salient_task: bool = False
    include_ambiguous_examples: bool = False
    construction_types: List[str] = field(
        default_factory=list,
        metadata={"help": "List of tasks or categories for which to generate examples"},
    )

    # model: str

    @classmethod
    def from_dict(cls, params):
        class_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in class_fields})


@dataclass
class AmbiBenchDataset:

    date: str
    config: AmbiBenchConfig
    examples: List[Dict[str, str]] = field(
        default_factory=list, metadata={"help": "List of query-completion tuple"}
    )

    candidate_categories: List[str] = field(
        default_factory=list,
        metadata={"help": "List of possible categories that could generate examples."},
    )

    assistance_prompts: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Additional prompts for COT, clarification, verbalisation"},
    )

    @classmethod
    def from_dict(cls, params):
        class_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in class_fields})

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = AmbiBenchConfig(**self.config)


class DatasetGenerator:
    def __init__(self, config: AmbiBenchConfig) -> None:
        self.config = config

        self.dataset = AmbiBenchDataset(
            date=datetime.datetime.now().strftime("%Y%m%d_%H-%M"), config=config
        )

    def generate_examples(self, n_queries):
        # adapted from `Tester.run_two_feature_tests_with_two_set` in original repo
        # TODO: change the following depending on test cases
        # for now assume values for two-feature tests
        for_finetuning = True
        finetuning_control = False

        for construct_type in self.config.construction_types:

            if construct_type in ConstructionType.list():
                construction_type = ConstructionType(construct_type)
            else:
                logger.warning(
                    f"Construction type string '{construct_type}' does not have valid mapping to construction type -> Skipped!"
                )
                continue

            if self.config.no_salient_task:
                salient_task = None
            else:
                salient_task = construction_type

            for i in range(n_queries):

                prompt = Prompt(
                    shots=self.config.n_shots,
                    construction_type=construction_type,
                    format_type=self.config.construction_format,
                    needs_instruction=self.config.needs_instruction,
                    needs_informative=self.config.needs_informative,
                    include_ambiguous_examples=self.config.include_ambiguous_examples,
                    salient_type=salient_task,
                    prob_of_ambiguous=self.config.prob_of_ambiguous,
                    for_finetuning=for_finetuning,
                    finetuning_control=finetuning_control,
                )

                api_access = OpenAI_APIAccess(prompt)  # this object formats the prompt
                formatted_pair = api_access.generate_data_for_openai_finetuning(
                    format=self.config.construction_format,
                    needs_instruction=self.config.needs_instruction,
                )  # { "prompt": prompt, "completion": completion, "salient_category": salient_category }

                formatted_pair["salient_category"] = prompt.salient_category.label

                if self.config.n_multiple_choices > 0:
                    formatted_pair[
                        "multiple_choice_category"
                    ] = prompt.generate_multiple_choice_categories(
                        self.config.n_multiple_choices
                    )

                self.dataset.examples.append(formatted_pair)

        # Note: currently assuming that assitance prompts are general across examples and don't depend on construction type
        self.dataset.assistance_prompts[
            "clarify"
        ] = prompt.generate_clarifying_assertion()
        self.dataset.assistance_prompts[
            "category_prediction"
        ] = prompt.generate_category_prediction_prompt()

        # add all possible categories
        self.dataset.candidate_categories = GenerationCategories.label_list()

    def save_examples_as_json(self, output_dir: str, jsonl=False):
        os.makedirs(output_dir, exist_ok=True)

        if jsonl:
            file_path = os.path.join(
                output_dir, f"{self.dataset.date}_ambibench_examples.jsonl"
            )
            with open(file_path, "w") as f_out:
                json.dump(asdict(self.dataset), f_out)
        else:
            file_path = os.path.join(
                output_dir, f"{self.dataset.date}_ambibench_examples.json"
            )

            with open(file_path, "w", encoding="utf-8") as f_out:
                json.dump(asdict(self.dataset), f_out, indent=4)

        logger.info(f"Dataset saved to: {file_path}")
