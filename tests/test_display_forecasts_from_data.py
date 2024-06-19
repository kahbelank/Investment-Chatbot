import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Mocking Streamlit components
mock_st = MagicMock()

modules = {
    'streamlit': mock_st,
    'streamlit_chat': mock_st,
    'joblib': MagicMock(),
    'prophet': MagicMock(),
}

patcher = patch.dict('sys.modules', modules)
patcher.start()

from BotFolio_Func import display_forecasts_from_data

class TestDisplayForecastsFromData(unittest.TestCase):
    @patch('joblib.load')
    @patch('prophet.Prophet.predict')
    def test_display_forecasts_from_data(self, mock_predict, mock_load):
        data = {
            'Date': pd.date_range(start='1/1/2020', periods=10),
            'Close': [i for i in range(10)]
        }
        df = pd.DataFrame(data)
        df['SMA_10'] = df['Close']
        df['RSI_14'] = df['Close']
        df['Upper_Band'] = df['Close']
        df['Lower_Band'] = df['Close']
        df['Close_1'] = df['Close']
        df['Close_2'] = df['Close']

        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_predict.return_value = pd.DataFrame({
            'ds': pd.date_range(start='1/11/2020', periods=30),
            'yhat': [i for i in range(30)],
            'yhat_upper': [i + 1 for i in range(30)],
            'yhat_lower': [i - 1 for i in range(30)]
        })

        fig = display_forecasts_from_data(df, 'AAPL', periods=30)
        self.assertIsNotNone(fig)

if __name__ == '__main__':
    unittest.main()
