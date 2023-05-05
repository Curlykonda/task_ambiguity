import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from src.structures.category import (
    Affirmations,
    AnimalSubjects,
    ExampleCategory,
    FemalePronouns,
    HumanSubjects,
    MalePronouns,
    NaturalLocations,
    Negations,
    ProperNouns,
    RegligiousLeaders,
    SecularLeaders,
    UrbanLocations,
)
from src.structures.construction_types import ConstructionType
from src.structures.example import Example


class ExampleGenerator(ABC):
    """
    Generate examples which are used to generate prompts for a language model

    Attributes:
        construction_type (str): specificies the type of example to generate: one of {subject_location, religious_pronoun, propn_negation}
        format_type (str): specifies the format needed to generate the example: one of {qa, arrow}
    """

    def __init__(self, construction_type: ConstructionType, format_type: str):
        self.construction_type = construction_type
        self.format_type = format_type

    def get_locations(self) -> List[str]:
        return NaturalLocations.values + UrbanLocations.values

    @abstractmethod
    def generate_example(
        self,
        task_a_label: bool,
        task_b_label: bool,
        active_task_label: bool,
        salient_task: Optional[str] = None,
    ) -> Example:
        raise NotImplementedError

    @abstractmethod
    def get_salient_category_from_task_key(
        cls, task_key: Tuple[str, bool]
    ) -> ExampleCategory:
        raise NotImplementedError

    def generate_example_given_salient(self, test_example: Example) -> Example:
        """
        Generates an example for a specificied salient task mirroring the test_example.
        When given a test example, it generates another example with the same salient task
        but randomizes the other task features.

        Args:
            text_example (Example): an example with the desired salient task set to True

        Returns:
            example (Example): an example mirroring the inputted test_example
        """
        mirror_example = random.choice([True, False])
        if mirror_example:
            task_a_label = test_example.task_a_label
            task_b_label = test_example.task_b_label
            active_task_label = test_example.active_task_label
        else:
            task_a_label = not test_example.task_a_label
            task_b_label = not test_example.task_b_label
            active_task_label = not test_example.active_task_label

        return self.generate_example(
            task_a_label, task_b_label, active_task_label, test_example.salient_task
        )


class SubjectLocationGenerator(ExampleGenerator):
    """
    Generates subject-location-type constructions
    An example construction: The {horse} is in the {lagoon}.
    """

    def __init__(self, construction_type, format_type):
        super().__init__(construction_type, format_type)
        self.salient_cat_map = {
            ("task_a", True): HumanSubjects,
            ("task_a", False): AnimalSubjects,
            ("task_b", True): UrbanLocations,
            ("task_b", False): NaturalLocations,
        }

    def generate_example(
        self,
        task_a_label: bool,
        task_b_label: bool,
        active_task_label: bool,
        salient_task: Optional[str] = None,
    ) -> Example:
        """
        Generates an construction of the above format.

        Args:
            task_a_label (str): True to randomly select a human subject, False to randomly select an animal subject
            task_b_label (str): True to randomly select an urban location, False to randomly select a natural location
            active_task_label (bool): the ouput label for the example: either True or False
            salient_task (str): the task which is salient for the example: one of {'task_a', 'task_b', None}
        Returns:
            Example (Example): Example object with relevant metadata
        """

        if task_a_label:
            choice_a = random.choice(HumanSubjects.values)
        else:
            choice_a = random.choice(AnimalSubjects.values)
        if task_b_label:
            choice_b = random.choice(UrbanLocations.values)
        else:
            choice_b = random.choice(NaturalLocations.values)

        construction = f"The {choice_a} is in the {choice_b}."

        return Example(
            construction_type=self.construction_type,
            format_type=self.format_type,
            construction=construction,
            task_a_label=task_a_label,
            task_b_label=task_b_label,
            active_task_label=active_task_label,
            salient_task=salient_task,
        )

    def get_salient_category_from_task_key(
        self, task_key: Tuple[str, bool]
    ) -> ExampleCategory:
        if task_key in self.salient_cat_map:
            return self.salient_cat_map[task_key]
        else:
            raise KeyError(f"Invalid key for salient task map: {repr(task_key)}")


class ReligiousPronounGenerator(ExampleGenerator):
    """
    Generates religious-pronoun-type constructions
    An example construction: {She} is in the laboratory with the {rabbi}.
    """

    def __init__(self, construction_type, format_type):
        super().__init__(construction_type, format_type)
        self.salient_cat_map = {
            ("task_a", True): RegligiousLeaders,
            ("task_a", False): SecularLeaders,
            ("task_b", True): MalePronouns,
            ("task_b", False): FemalePronouns,
        }

    def generate_example(
        self,
        task_a_label: bool,
        task_b_label: bool,
        active_task_label: bool,
        salient_task: Optional[str] = None,
    ):
        """
        Generates an construction of the above format.

        Args:
            task_a_feature (bool): True to randomly select a relgious leader, False to randomly select a secular leader
            task_b_feature (bool): True to select the pronoun 'he', False to select the pronoun 'she'
            active_task_label (bool): the ouput label for the example: either True or False
            salient_task (str): the task which is salient for the example: one of {task_a, task_b, None}
        Returns:
            Example (Example): Example object with relevant metadata
        """

        if task_a_label:
            choice_a = random.choice(RegligiousLeaders.values)
        else:
            choice_a = random.choice(SecularLeaders.values)

        if task_b_label:
            choice_b = random.choice(MalePronouns.values)
        else:
            choice_b = random.choice(FemalePronouns.values)

        urban_location = random.choice(UrbanLocations.values)

        construction = f"{choice_b} is in the {urban_location} with the {choice_a}."

        return Example(
            construction_type=self.construction_type,
            format_type=self.format_type,
            construction=construction,
            task_a_label=task_a_label,
            task_b_label=task_b_label,
            active_task_label=active_task_label,
            salient_task=salient_task,
        )

    def get_salient_category_from_task_key(
        self, task_key: Tuple[str, bool]
    ) -> ExampleCategory:
        if task_key in self.salient_cat_map:
            return self.salient_cat_map[task_key]
        else:
            raise KeyError(f"Invalid key for salient task map: {repr(task_key)}")


class ProperNounNegationGenerator(ExampleGenerator):
    """
    Generates propn-negation-type constructions
    An example construction: {Noam Chomsky} {was not} in the theatre.
    """

    def __init__(self, construction_type, format_type):
        super().__init__(construction_type, format_type)
        self.salient_cat_map = {
            ("task_a", True): ProperNouns,
            ("task_a", False): HumanSubjects,
            ("task_b", True): Affirmations,
            ("task_b", False): Negations,
        }

    def generate_example(
        self,
        task_a_label: bool,
        task_b_label: bool,
        active_task_label: bool,
        salient_task: Optional[str] = None,
    ):
        """
        Generates an construction of the above format.

        Args:
            task_a_feature (bool): True to randomly select a proper noun, False to select a common noun
            task_b_feature (bool): True to not contain a negation, False to contain a negation
            active_task_label (bool): the ouput label for the example: either True or False
            salient_task (str): the task which is salient for the example: one of {task_a, task_b, None}
        Returns:
            Example (Example): Example object with relevant metadata
        """

        if task_a_label:
            choice_a = random.choice(ProperNouns.values)
        else:
            choice_a = random.choice(HumanSubjects.values)
            choice_a = "The " + choice_a

        if task_b_label:
            choice_b = random.choice(Affirmations.values)
        else:
            choice_b = random.choice(Negations.values)

        urban_location = random.choice(UrbanLocations.values)

        construction = f"{choice_a} {choice_b} in the {urban_location}."

        return Example(
            construction_type=self.construction_type,
            format_type=self.format_type,
            construction=construction,
            task_a_label=task_a_label,
            task_b_label=task_b_label,
            active_task_label=active_task_label,
            salient_task=salient_task,
        )

    def get_salient_category_from_task_key(
        self, task_key: Tuple[str, bool]
    ) -> ExampleCategory:
        if task_key in self.salient_cat_map:
            return self.salient_cat_map[task_key]
        else:
            raise KeyError(f"Invalid key for salient task map: {repr(task_key)}")


def get_generator_from_construction_type(
    construction_type: Union[str, ConstructionType], format_type: str
) -> ExampleGenerator:

    if isinstance(construction_type, str):
        construction_type = ConstructionType(construction_type)

    if construction_type in [
        ConstructionType.LOCATION,
        ConstructionType.SUBJECT,
        ConstructionType.SUBJECT_LOCATION,
    ]:
        return SubjectLocationGenerator(construction_type, format_type)
    elif construction_type in [
        ConstructionType.PROPER_NOUN,
        ConstructionType.NEGATION,
        ConstructionType.PROPN_NEGATION,
    ]:
        return ProperNounNegationGenerator(construction_type, format_type)
    elif construction_type in [
        ConstructionType.RELIGIOUS,
        ConstructionType.PRONOUN,
        ConstructionType.RELIGIOUS_PRONOUN,
    ]:
        return ReligiousPronounGenerator(construction_type, format_type)

    else:
        raise ValueError(
            f"Undefined mapping to generator for: {repr(construction_type)}"
        )
