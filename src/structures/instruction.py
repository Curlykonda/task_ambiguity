from abc import ABC, abstractmethod
from typing import Optional, Union

from src.example_generation import ExampleCategory, HumanSubjects
from src.structures.construction_types import ConstructionType

CATEGORY_WITHHELD = "[category withheld]"


class Instruction(ABC):
    """
    Creates an instruction for each prompt

    An instruction is a string which informs the model how to interpret the provided examples,
    which task to attend to in labeling each construction, and in turn how to label them.
    For example: 'Output 'X' if the sentence contains a reference to an outdoor setting and 'Y' otherwise'

    Attributes:
        construction_type (ConstructionType): the type of examples to generate
    """

    def __init__(self, construction_type: ConstructionType):
        self.construction_type = construction_type
        self.tasks = [None, None]
        self.salient_task = None

    def make_uninformative_instruction(self) -> str:
        return f"Output 'X' if the sentence contains a {CATEGORY_WITHHELD} and 'Y' otherwise."

    def make_clarifying_assertion(self) -> str:
        return "Ask a clarifying question if you are unsure what to output in the following task.\nTask:\n"

    def make_verbalize_category(self, davinci_version="003") -> str:
        if "002" == davinci_version:
            return f"What is the {CATEGORY_WITHHELD}?"
        else:
            return f"What is your best guess for the {CATEGORY_WITHHELD} above?"

    def make_mc_category_prediction(self, n_options: Optional[int] = None) -> str:
        # create prompt to predict category from multiple choice
        mc_stem = "Select one from the folliwng {} options:"
        if n_options is not None:
            mc_prompt = mc_stem.format(n_options)
        else:
            mc_prompt = mc_stem.format("")

        return (
            f"What is your best guess for the {CATEGORY_WITHHELD} above? " + mc_prompt
        )

    @abstractmethod
    def make_instruction(self, salient_category: ExampleCategory) -> str:
        """
        Given the salient category, creates an instruction string.

        Returns:
            instruction (str): a string consisting of the correct instruction for given the Prompt
        """
        raise NotImplementedError()


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
