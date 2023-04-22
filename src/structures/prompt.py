import logging
import random
from typing import List, Optional, Tuple

from src.example_generation import (
    ExampleCategory,
    ExampleGenerator,
    GenerationCategories,
    get_generator_from_construction_type,
)
from src.structures.example import Example
from src.structures.instruction import (
    Instruction,
    get_instruction_from_construction_type,
)

logger = logging.getLogger("PromptConstructionLog")


class Prompt:
    """
    Creates a prompt for the OpenAI API using by generating examples
    A Prompt consists of three Examples (the last one being called the query), metadata on each of those Examples, and in some cases,
    an instruction
    For example, a Prompt may look like:
        Instruction
        Example 1 {metadata}
        Example 2 {metadata}
        Query {metadata}

    Attributes:
        shots (int): the number of examples to generate for each class (True / False)
        construction_type (str): the type of examples to generate: one of {subject_location, religious_pronoun, propn_negation}
        format_type (str): the type of format to generate: ['qa', 'arrow']
        needs_instruction (bool): True if wish to generate an instruction and False otherwise
        needs_informative (bool): True if the instruction is informative and False otherwise
        include_ambiguous_examples (bool): True if wish to include ambiguous examples and False otherwise
        prob_of_ambiguous (float): Number from 0.0 to 1.0 indicating the probability of each example generated being an ambigous example
        for_finetuning (bool): True if generating examples with withheld salient tasks for finetuning
        finetuning_control (bool): True if generating examples for finetuning control tests
        salient_task (str): salient task for which to make examples (not required to generate examples)
    """

    def __init__(
        self,
        shots,
        construction_type,
        format_type,
        needs_instruction,
        needs_informative,
        include_ambiguous_examples,
        prob_of_ambiguous,
        for_finetuning,
        finetuning_control,
        salient_task: Optional[str] = None,
    ):

        self.shots = shots
        self.construction_type = construction_type
        self.instruction = get_instruction_from_construction_type(
            self.construction_type
        )
        self.format_type = format_type

        self.examples = []
        self.clarifying_assertion = ""
        self.salient_category: GenerationCategories  # underlying ground truth category that determines the labels

        # makes examples based on type of test being run: either with an explicit salient task or without
        if salient_task is not None:
            self.make_given_distribution_examples(
                prob_of_ambiguous=prob_of_ambiguous,
                needs_instruction=needs_instruction,
                needs_informative=needs_informative,
                salient_task=salient_task,
                for_finetuning=for_finetuning,
                finetuning_control=finetuning_control,
            )
        else:
            self.make_examples(
                needs_instruction, needs_informative, include_ambiguous_examples
            )

    def _get_generator_for_construction_type(self):
        """
        Checks what type of object to make based upon the specific contruction type

        Args:
            None
        Returns:
            construction_obj (ExampleGenerator): the object corresponding to the specific construction type

        """

        return get_generator_from_construction_type(
            self.construction_type, self.format_type
        )

    def get_examples(self):
        return self.examples

    def make_examples(
        self, needs_instruction, needs_informative, include_ambiguous_examples
    ):
        """
        Generates a specific number (shots) of examples of the specific contruction type

        Args:
            needs_instruction (bool): True if instruction needed and False otherwise
            needs_informative (bool): True if instruction is informative and False otherwise
            include_ambiguous_examples (bool): True if wish to include ambiguous examples and False otherwise
        Returns:
            None
        """
        current_examples = []

        """
        Randomizes the order of the labels for the examples -- such that ~50% of the time the first label is X and the second is Y and
        the other 50% of the time the first label is Y and the second is X
        For example, if examples_label_randomizer == True:
            Label 1: X
            Label 2: Y
        But if examples_label_ randomizer == False:
            Label 1: Y
            Label 2: X
        """
        examples_label_randomizer = random.choice([True, False])

        """
        Randomizes the order of the examples -- such that ~50% of the time the first example has one set of features
        and the second has the other and vice versa for the other 50%
        For example, if construction_type == 'subject_location' && if examples_order_randomizer == True:
            Example 1: The {human} is in the {indoor_location}
            Example 2: The {animal} is in the {outdoor_location}
        But if examples_order_randomizer == False:
            Example 1: The {animal} is in the {outdoor_location}
            Example 2: The {human} is in the {indoor_location}
        """
        examples_order_randomizer = random.choice([True, False])

        # generates the first two examples using the randomizers explained above
        # selected the correct ExampleGenerator object based on the construction type
        construction_obj = self._get_generator_for_construction_type()

        if include_ambiguous_examples:
            for i in range(2):
                label = (
                    examples_label_randomizer
                    if i % 2 == 0
                    else not examples_label_randomizer
                )
                if examples_order_randomizer:
                    example = construction_obj.generate_example(True, True, label)
                else:
                    example = construction_obj.generate_example(False, False, label)

                self.examples.append(example)
                current_examples.append(example)

                # ensures the next example is the opposite kind as the previous one as explained above
                examples_order_randomizer = not examples_order_randomizer

        """
        Randomzies the query (which disambiguates the previous two examples)
        For example, if construction_type == 'subject_location' && if query_randomizer == True:
            Query: The {human} is in the {outdoor_location}
        But if query_randomzier == False:
            Query: The {animal} is in the {indoor_location}
        """
        query_randomizer = random.choice([True, False])

        # Randomizes the label of the query -- such that ~50% of the time the query label is X (if query_label_randomzier = True)
        # and the other 50% it is Y (if query_label_randomzier = False)
        query_label_randomizer = random.choice([True, False])

        # Generates the query
        query = construction_obj.generate_example(
            query_randomizer, not query_randomizer, query_label_randomizer
        )

        self.examples.append(query)
        current_examples.append(query)

        if needs_instruction:
            self.instruction = self.generate_instruction(
                current_examples, needs_informative, include_ambiguous_examples
            )

        # Sets the salient_task as the same task for all Examples for the current Prompt
        # (used for visualizations and data wrangling further down the pipeline)
        self.set_salient_task(
            current_examples=current_examples,
            include_ambiguous_examples=include_ambiguous_examples,
        )

        for _ in range(self.shots - 1):
            construction_obj = self._get_generator_for_construction_type()
            example = construction_obj.generate_example_given_salient(
                current_examples[-1]
            )

            self.examples.append(example)
            current_examples.append(example)

    def make_given_distribution_examples(
        self,
        prob_of_ambiguous,
        needs_instruction,
        needs_informative,
        for_finetuning,
        finetuning_control,
        salient_task: str,
    ):
        """
        Generates examples given a salient task

        Args:
            needs_instruction (bool): True if instruction needed and False otherwise
            needs_informative (bool): True if instruction is informative and False otherwise
            needs_informative (bool): True if wish to include informative instructions and False otherwise
            for_finetuning (bool): True if wish to generate examples for finetuning and False otherwise
            finetuning_control (bool): True if running control tests for finetuning and False otherwise
            salient_task (str): The salient task for the set of examples
        Returns:
            None
        """
        current_examples = []
        examples_distribution = ["ambiguous"] * prob_of_ambiguous + [
            "disambiguating"
        ] * (100 - prob_of_ambiguous)

        salient_task_label = random.choice([True, False])
        active_task_label = random.choice([True, False])

        possible_task_a = ["subject", "religious", "propn"]
        possible_task_b = ["location", "pronoun", "negation"]

        construction_obj = self._get_generator_for_construction_type()

        if salient_task in possible_task_a:
            salient = "task_a"
        elif salient_task in possible_task_b:
            salient = "task_b"
        else:
            raise Exception("invalid salient task")

        if for_finetuning and finetuning_control:
            randomize_tasks = random.choice([True, False])

        # generated specified number of examples
        for _ in range(self.shots):
            if not for_finetuning or not finetuning_control:
                randomize_tasks = random.choice([True, False])

            example_type = random.choice(examples_distribution)

            # Randomzies the example generated which maintaining the specified salient task for the set of examples
            if example_type == "disambiguating":
                if randomize_tasks and salient == "task_a":
                    example = construction_obj.generate_example(
                        salient_task_label,
                        not salient_task_label,
                        active_task_label,
                        salient_task,
                    )  # original: use `salient_task=salient_task`
                elif not randomize_tasks and salient == "task_a":
                    example = construction_obj.generate_example(
                        not salient_task_label,
                        salient_task_label,
                        not active_task_label,
                        salient_task,
                    )
                elif randomize_tasks and salient == "task_b":
                    example = construction_obj.generate_example(
                        not salient_task_label,
                        salient_task_label,
                        active_task_label,
                        salient_task,
                    )
                else:
                    example = construction_obj.generate_example(
                        salient_task_label,
                        not salient_task_label,
                        not active_task_label,
                        salient_task,
                    )
            else:
                if randomize_tasks:
                    example = construction_obj.generate_example(
                        salient_task_label,
                        salient_task_label,
                        active_task_label,
                        salient_task,
                    )
                else:
                    example = construction_obj.generate_example(
                        not salient_task_label,
                        not salient_task_label,
                        not active_task_label,
                        salient_task,
                    )

            current_examples.append(example)
            self.examples.append(example)

        # get salient category that determines labels
        self.salient_category = self.get_salient_category_from_example_set(
            current_examples,
            construction_obj,
            include_ambiguous_examples=True,
            salient_task_a_or_b=salient,
        )

        # adds instruction if needed
        if needs_instruction:
            self.instruction_text = self.generate_instruction(
                self.salient_category, needs_informative
            )

    def obtain_salient_task_key(self, current_examples) -> Tuple[str, bool]:
        """
        Obtains the correct salient task from the examples + query in the prompt

        This is necessary because Examples are generated randomly,
        so we need to infer the salient task that would have produced the Example

        For example: if in current examples,
        Example 1: {task_a_label = True, task_b_label = True, active_task_label = True}
        Example 2: {task_a_label = False, task_b_label = False, active_task_label = False}
        Query: {task_a_label = True, task_b_label = False, active_task_label = True}

        query.active_task_label == first_example.active_task_label,
        so salient_example = first_example
        this means that the example which we will need to focus on to determine
        the salient task is Example 1

        because query.active_task_label = True,
        and query.task_a_label == salient_example.task_a_label,
        and query_task_a_label,
        we return ('task_a', True), meaning that the first task from the first example
        is the task determining the instruction

        let's see a Prompt for which this would occur (and the corresponding task labels
        remain the same as previously discussed):
        The critic is in the theatre.
        > X
        The hound is in the prairie.
        > Y
        The surveyor is in the canyon.
        > X

        Following the above logic, we can conclude that the salient_example must be the first example, and the salient_task
        must be 'task_a' and the instruction should assert 'A' in the case that task_a_label is True.

        The resulting instruction would be "Output 'X' if the sentence contains a reference to a human and 'Y' otherwise."
        which can be verified as the correct disambiguating instruction.

        Args:
            current_examples (Example): a list of all the examples (and the query) in the current prompt
        Returns:
            tuple: (name of salient task, bool of that tasks' label)
        """

        assert len(current_examples) >= 3

        first_example = current_examples[0]
        second_example = current_examples[1]
        query = current_examples[2]

        if query.active_task_label == first_example.active_task_label:
            salient_example = first_example
        else:
            salient_example = second_example

        if query.active_task_label:
            if query.task_a_label == salient_example.task_a_label:
                if query.task_a_label:
                    return ("task_a", True)

                else:
                    return ("task_a", False)

            else:
                if query.task_b_label:
                    return ("task_b", True)
                else:
                    return ("task_b", False)
        else:
            if query.task_a_label == salient_example.task_a_label:
                if query.task_a_label:
                    return ("task_a", False)
                else:
                    return ("task_a", True)
            else:
                if query.task_b_label:
                    return ("task_b", False)
                else:
                    return ("task_b", True)

    def create_salient_task_key(
        self, current_examples, salient_task: str
    ) -> Tuple[str, bool]:
        """
        Creates a tuple of the salient task and its label for the current prompt

        Args:
            current_examples (list(Example)): a list of all the examples (and the query) in the current prompt
            salient_task_a_or_b (str): the salient task for the current prompt, one of {'task_a', 'task_b'}
        Returns:
            tuple(str, bool): (name of salient task, bool of that tasks' label for set of examples)
        """
        query = current_examples[-1]

        if salient_task is None:
            print("WARNING: salient task is None but should be either [task_a, task_b]")

        if salient_task == "task_a":
            key_task_label = query.task_a_label
        else:
            key_task_label = query.task_b_label

        if query.active_task_label:
            return (salient_task, key_task_label)
        else:
            return (salient_task, not key_task_label)

    def get_salient_category_from_example_set(
        self,
        examples: List[Example],
        example_generator: ExampleGenerator,
        include_ambiguous_examples: bool,
        salient_task_a_or_b,
    ) -> ExampleCategory:
        """Determines the underlying category that generates the label for the example set"""

        if include_ambiguous_examples:
            salient_task_key = self.obtain_salient_task_key(examples)
        else:
            salient_task_key = self.create_salient_task_key(
                examples, salient_task_a_or_b
            )

        # using task_key, determine salient category
        salient_category = example_generator.get_salient_category_from_task_key(
            salient_task_key
        )

        return salient_category

    def set_salient_task(
        self, current_examples, include_ambiguous_examples, salient_task_a_or_b=None
    ):
        """
        Generates the correct instruction for the given salient task and set of examples for two-feature tests

        Args:
            current_examples (list): The current set of examples
            include_ambiguous_examples (bool): True if ambiguous examples are included and False otherwise
            salient_task_a_or_b (str): 'task_a' if the salient task is task_a and 'task_b' if the salient task is task_b
        Returns:
            Instruction (str): The correct instruction for the given salient task and set of examples
        """
        return Instruction(construction_type=self.construction_type).set_salient_task(
            current_examples=current_examples,
            include_ambiguous_examples=include_ambiguous_examples,
            salient_task_a_or_b=salient_task_a_or_b,
        )

    def generate_instruction(
        self,
        salient_category: ExampleCategory,
        needs_informative: bool,
    ) -> str:
        """
        Generates the correct instruction for the given salient category

        Args:
            current_examples (list): The current set of examples
            include_ambiguous_examples (bool): True if ambiguous examples are included and False otherwise
            salient_task_a_or_b (str): 'task_a' if the salient task is task_a and 'task_b' if the salient task is task_b
        Returns:
            Instruction (str): The correct instruction for the given salient task and set of examples
        """

        if needs_informative:
            instruction = self.instruction.make_instruction(salient_category)
        else:
            instruction = self.instruction.make_uninformative_instruction()

        return instruction

    def generate_clarifying_assertion(self):
        return self.instruction.make_clarifying_assertion()

    def get_instruction_text(self) -> str:
        return self.instruction_text

    def get_clarifying_assertion(self):
        return self.clarifying_assertion

    def print(self):
        if self.generate_instruction:
            print(str(self.instruction_text))
        else:
            print(
                "Output 'X' if the sentence contains a [cateogry withheld] 'Y' otherwise."
            )
        for e in self.examples:
            print("<br>" + str(e.construction))

            if e.active_task_label:
                print("<br>&gt;X")
            else:
                print("<br>&gt;Y")

        print("###")
