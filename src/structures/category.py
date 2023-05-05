from dataclasses import dataclass, field
from typing import List

from src.structures.common import ExtendedEnum
from src.structures.construction_types import ConstructionType


@dataclass
class ExampleCategory:
    label: str
    parent: ConstructionType
    instruction: str
    values: List[str] = field(default_factory=list)

    def __str__(self):
        return self.label


class UrbanLocations(ExampleCategory):

    label = "urban_location"
    values = [
        "laboratory",
        "theatre",
        "museum",
        "courtroom",
        "apartment building",
        "restaurant",
        "house",
        "film studio",
        "hotel lobby",
        "grocery store",
    ]
    parent = ConstructionType.LOCATION
    instruction = "an indoor setting"


class NaturalLocations(ExampleCategory):

    label = "natural_location"
    values = [
        "river",
        "pond",
        "woodlands",
        "cave",
        "canyon",
        "prairie",
        "jungle",
        "marsh",
        "lagoon",
        "meadow",
    ]
    parent = ConstructionType.LOCATION
    instruction = "an outdoor setting"


class HumanSubjects(ExampleCategory):

    label = "human_subject"
    values = [
        "student",
        "reporter",
        "hiker",
        "researcher",
        "firefighter",
        "fugitive",
        "critic",
        "photographer",
        "director",
        "surveyor",
    ]
    instruction = "a human"
    parent = ConstructionType.SUBJECT


class AnimalSubjects(ExampleCategory):

    label = "animal_subject"
    values = [
        "boar",
        "worm",
        "hawk",
        "hound",
        "butterfly",
        "snake",
        "duck",
        "bear",
        "mountain lion",
        "horse",
    ]
    parent = ConstructionType.SUBJECT
    instruction = "an animal"


class RegligiousLeaders(ExampleCategory):

    label = "religious_leader"
    values = [
        "pope",
        "reverend",
        "bishop",
        "Dalai Lama",
        "rabbi",
        "cardinal",
        "pastor",
        "deacon",
        "imam",
        "ayatollah",
    ]
    parent = ConstructionType.RELIGIOUS
    instruction = "contains a reference to a religious leader"


class SecularLeaders(ExampleCategory):

    label = "secular_leader"
    values = [
        "president",
        "CEO",
        "principal",
        "sheriff",
        "judge",
        "ambassador",
        "officer",
        "prime minister",
        "colonel",
        "professor",
    ]
    parent = (
        ConstructionType.RELIGIOUS
    )  # TODO: check whether to use different construction type
    instruction = "does not contain a reference to a religious leader"


class FemalePronouns(ExampleCategory):
    label = "female_pronoun"
    values = ["She"]
    parent = ConstructionType.PRONOUN
    instruction = "contains a female pronoun"


class MalePronouns(ExampleCategory):
    label = "male_pronoun"
    values = ["He"]
    parent = ConstructionType.PRONOUN
    instruction = "contains a male pronoun"


class ProperNouns(ExampleCategory):

    label = "proper_noun"
    values = [
        "Lebron James",
        "Bernie Sanders",
        "Christopher Nolan",
        "Paul Atreides",
        "Noam Chomsky",
        "Serena Williams",
        "Margot Robbie",
        "Alexandria Ocasio-Cortez",
        "Hermione Granger",
        "Jane Goodall",
    ]
    parent = ConstructionType.PROPER_NOUN
    instruction = "contains a proper noun"


class Negations(ExampleCategory):
    label = "negation"
    values = ["is not", "was not", "has not been", "may not be", "could not be"]

    parent = ConstructionType.NEGATION
    instruction = "contains a negation"


class Affirmations(ExampleCategory):
    label = "affirmation"
    values = ["is", "was", "has been", "may be", "could be"]

    parent = ConstructionType.NEGATION
    instruction = "does not contain a negation"


class GenerationCategories(ExtendedEnum):
    """Selection of possible categories that can underlie the example generation"""

    URBAN_LOCATIONS = UrbanLocations
    NATURAL_LOCATIONS = NaturalLocations
    HUMAN_SUBJECTS = HumanSubjects
    ANIMAL_SUBJECTS = AnimalSubjects
    RELIGIGOUS_LEADERS = RegligiousLeaders
    SECULAR_LEADERS = SecularLeaders
    PROPER_NOUN = ProperNouns
    NEGATATIONS = Negations
    AFFIRMATIONS = Affirmations
    FEMALE_PRONOUNS = FemalePronouns
    MALE_PRONOUNS = MalePronouns

    @classmethod
    def label_list(cls) -> List[str]:
        # get list of all category labels
        return list(map(lambda c: c.value.__str__(c.value), cls))
