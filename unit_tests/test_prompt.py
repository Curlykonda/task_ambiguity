"""Test correct behaviour of generating Prompt and determining salient Category """
import unittest
from unittest.mock import patch

from src.example_generation import HumanSubjects
from src.structures.prompt import Prompt


class TestPromptMultipleChoiceCategory(unittest.TestCase):
    @patch.object(Prompt, "__init__", return_value=None)
    def setUp(self, init_patch) -> None:

        self.prompt = Prompt()
        self.prompt.salient_category = HumanSubjects

    def test_multiple_choice_generation_has_right_content(self, num_options=4):

        mc_string = self.prompt.generate_multiple_choice_categories(num_options)

        self.assertTrue(mc_string.__contains__(self.prompt.salient_category.label))
        self.assertTrue(mc_string.startswith("1."))
        self.assertTrue(mc_string.__contains__(f"{num_options}."))


if __name__ == "__main__":
    unittest.main()
