import argparse
import random

from src.generate_dataset import AmbiBenchConfig, DatasetGenerator
from src.structures.construction_types import ConstructionType


def set_random_seed(seed: int) -> None:
    random.seed(seed)


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--construction_types",
        nargs="+",
        required=False,
        default="location",
        help="Provide 1 or more tasks/categories for which to generate examples",
        choices=ConstructionType.list(),
    )
    parser.add_argument(
        "--construction_format",
        choices=["arrow", "qa"],
        type=str,
        required=False,
        default="qa",
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        required=False,
        default=3,
        help="Number of shots per query",
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        required=False,
        default=10,
        help="Number of queries/examples to generate",
    )
    parser.add_argument(
        "--n_multiple_choices",
        type=int,
        required=False,
        default=4,
        help="Number of multiple-choice categories",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=None,
        help="Directory where to store dataset as JSON",
    )
    parser.add_argument("--needs_instruction", action="store_true")
    parser.add_argument("--needs_informative", action="store_true")
    parser.add_argument("--needs_multiple_choice", action="store_true")
    parser.add_argument("--include_ambiguous_examples", action="store_true")
    parser.add_argument(
        "--no_salient_task",
        action="store_true",
        help="Do not explicitly use construction type as salient task.",
    )
    parser.add_argument("--verbose", type=bool, required=False, default=True)
    parser.add_argument("--prob_of_ambiguous", type=float, required=False, default=50)

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=1337,
        help="Random seed",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()
    set_random_seed(args.seed)

    config = AmbiBenchConfig.from_dict(vars(args))

    data_generator = DatasetGenerator(config)
    data_generator.generate_examples(config.n_queries)

    if args.output_dir is not None:
        data_generator.save_examples_as_json(output_dir=args.output_dir)
