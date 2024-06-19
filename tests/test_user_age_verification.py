import unittest
from unittest.mock import patch, MagicMock

# Mocking Streamlit components
mock_st = MagicMock()

modules = {
    'streamlit': mock_st,
    'streamlit_chat': mock_st,
}

patcher = patch.dict('sys.modules', modules)
patcher.start()

from BotFolio_Func import user_age_verification

class TestUserAgeVerification(unittest.TestCase):
    def test_valid_age(self):
        self.assertEqual(user_age_verification("25"), "You are over 18 years old! Enjoy the use of our investment portfolio generator!")

    def test_invalid_age_string(self):
        self.assertEqual(user_age_verification("abc"), "I'm sorry, but it looks like you entered an invalid number. Please enter a valid whole number!")

    def test_too_young(self):
        self.assertEqual(user_age_verification("17"), "This application requires you to be at least 18 years old!")

    def test_too_old(self):
        self.assertEqual(user_age_verification("111"), "I'm sorry, but it looks like you are too old to use this application. Please enter an age less than 110!")

if __name__ == '__main__':
    unittest.main()
