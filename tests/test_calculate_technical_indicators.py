import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

# Mocking Streamlit components
mock_st = MagicMock()

modules = {
    'streamlit': mock_st,
    'streamlit_chat': mock_st,
}

patcher = patch.dict('sys.modules', modules)
patcher.start()

from BotFolio_Func import calculate_technical_indicators

class TestCalculateTechnicalIndicators(unittest.TestCase):
    def test_calculate_technical_indicators(self):
        data = {
            'Date': pd.date_range(start='1/1/2020', periods=20),
            'Close': [i for i in range(20)]
        }
        df = pd.DataFrame(data)
        df = calculate_technical_indicators(df)

        self.assertIn('SMA_10', df.columns)
        self.assertIn('RSI_14', df.columns)
        self.assertIn('Upper_Band', df.columns)
        self.assertIn('Lower_Band', df.columns)
        self.assertIn('Close_1', df.columns)
        self.assertIn('Close_2', df.columns)

if __name__ == '__main__':
    unittest.main()
