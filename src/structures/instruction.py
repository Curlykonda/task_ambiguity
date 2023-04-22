from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from src.example_generation import ExampleCategory, HumanSubjects
from src.structures.construction_types import ConstructionType


class Instruction(ABC):
    """
    Creates an instruction for each prompt

    An instruction is a string which informs the model how to interpret the provided examples,
    which task to attend to in labeling each construction, and in turn how to label them.
    For example: 'Output 'X' if the sentence contains a reference to an outdoor setting and 'Y' otherwise'

    Attributes:
        construction_type (str): the type of examples to generate: one of {subject_location, religious_pronoun, propn_negation}
        current_examples (Example): a list of all the examples and the query in the current prompt
    """

    def __init__(self, construction_type: ConstructionType):
        self.construction_type = construction_type
        self.tasks = [None, None]
        self.salient_task = None

    def make_uninformative_instruction(self) -> str:
        return "Output 'X' if the sentence contains a [category withheld] and 'Y' otherwise."

    def make_clarifying_assertion(self):
        return "Ask a clarifying question if you are unsure what to output in the following task.\nTask:\n"

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

    def set_salient_task(
        self,
        current_examples,
        include_ambiguous_examples: bool,
        salient_task_a_or_b: Optional[str] = None,
    ):
        """
        Sets the salient_task for the current example.
        The salient_task corresponds to the task which determines the label of the query
        All of the examples in a Prompt will have the same salient_task value and as such this value is set once the Prompt has
        already been made

        For example, in lengthy example provided in the docstring for obtain_salient_task_key(), all 3 of the Examples would have
        their salient_task variable be set to 'subject' as that is the task that determined the instruction.
        This is needed when visualizing the results to provide each task with its own bar
        (as opposed to one bar for each construction_type)

        Args:
            current_examples (Example): a list of all the examples (and the query) in the current prompt
            include_ambiguous_examples (bool): whether or not to include ambiguous examples in the prompt
            salient_task_a_or_b (str): the salient task for the current prompt, one of {'task_a', 'task_b', None}
        Returns:
            None
        """
        if include_ambiguous_examples:
            salient_task_key = self.obtain_salient_task_key(current_examples)
        else:
            salient_task_key = self.create_salient_task_key(
                current_examples, salient_task_a_or_b
            )

        construction_type_map = {
            "subject_location": SubjectLocationInstruction(self.construction_type),
            "religious_pronoun": ReligiousPronounInstruction(self.construction_type),
            "propn_negation": PropNNegationInstruction(self.construction_type),
        }
        construction_type_map[self.construction_type].set_custom_salient_task(
            salient_task_key, current_examples
        )

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

    @abstractmethod
    def make_instruction(self, salient_category: ExampleCategory) -> str:
        """
        Given the salient category, creates an instruction string.

        Returns:
            instruction (str): a string consisting of the correct instruction for given the Prompt
        """
        raise NotImplementedError()

    def set_custom_salient_task(self, salient_task_key, current_examples):
        if salient_task_key[0] == "task_a":
            self.salient_task = self.tasks[0]
        else:
            self.salient_task = self.tasks[1]

        for e in current_examples:
            e.salient_task = self.salient_task

        return current_examples


class SubjectLocationInstruction(Instruction):
    """
    Generates subject-location-type instructions

    For example: 'Output 'X' if the sentence contains a reference to an outdoor setting and 'Y' otherwise'
    """

    def __init__(self, construction_type: ConstructionType):
        self.construction_type = construction_type
        self.tasks = ["subject", "location"]

    def make_instruction(self, salient_category: ExampleCategory) -> str:
        return f"Output 'X' if the sentence contains a reference to {salient_category.instruction} and 'Y' otherwise."


class ReligiousPronounInstruction(Instruction):
    """
    Generates religious-pronoun-type instructions

    For example: 'Output 'X' if the sentence contains a female pronoun and 'Y' otherwise'
    """

    def __init__(self, construction_type):
        self.construction_type = construction_type
        self.tasks = ["religious", "pronoun"]

    def make_instruction(self, salient_category: ExampleCategory) -> str:
        return f"Output 'X' if the sentence {salient_category.instruction} and 'Y' otherwise."


class PropNNegationInstruction(Instruction):
    """
    Generates propn-negation-type instructions

    For example: 'Output 'X' if the sentence contains a proper noun and 'Y' otherwise'
    """

    def __init__(self, construction_type):
        self.construction_type = construction_type
        self.tasks = ["propn", "negation"]

    def make_instruction(self, salient_category: ExampleCategory) -> str:

        if isinstance(salient_category, HumanSubjects):
            instruct = "does not contain a proper noun"
        else:
            instruct = salient_category.instruction

        return f"Output 'X' if the sentence {instruct} and 'Y' otherwise."


def get_instruction_from_construction_type(
    construction_type: Union[str, ConstructionType]
) -> Instruction:

    if isinstance(construction_type, str):
        construction_type = ConstructionType(construction_type)

    if construction_type in [
        ConstructionType.LOCATION,
        ConstructionType.SUBJECT,
        ConstructionType.SUBJECT_LOCATION,
    ]:
        return SubjectLocationInstruction(construction_type)
    elif construction_type in [
        ConstructionType.PROPN,
        ConstructionType.NEGATION,
        ConstructionType.PROPN_NEGATION,
    ]:
        return PropNNegationInstruction(construction_type)
    elif construction_type in [
        ConstructionType.RELIGIOUS,
        ConstructionType.PRONOUN,
        ConstructionType.RELIGIOUS_PRONOUN,
    ]:
        return ReligiousPronounInstruction(construction_type)

    else:
        raise ValueError(
            f"Undefined mapping to instruction for: {repr(construction_type)}"
        )
