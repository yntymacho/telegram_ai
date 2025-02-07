from pathlib import Path
from typing import Dict, List
import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from config import config

class GoogleSheetsLoader:
    def __init__(self):
        """Initialize Google Sheets loader with credentials"""
        try:
            self.creds = Credentials.from_service_account_file(
                str(config.GOOGLE_CREDENTIALS_PATH),
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
            self.service = build('sheets', 'v4', credentials=self.creds)
            logger.info("Google Sheets API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets API: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def load_from_sheet(self) -> pd.DataFrame:
        """Load QA pairs from Google Sheet with retry logic"""
        try:
            result = self.service.spreadsheets().values().get(
                spreadsheetId=config.GOOGLE_SHEET_ID,
                range='questionsa.csv!A:B'  # Specify the sheet name
            ).execute()
            
            values = result.get('values', [])
            if not values:
                logger.warning("No data found in Google Sheet")
                return pd.DataFrame(columns=['question', 'answer'])
            
            df = pd.DataFrame(values[1:], columns=values[0])
            
            # Validate required columns
            required_columns = {'question', 'answer'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Missing required columns: {missing}")
            
            logger.info(f"Loaded {len(df)} QA pairs from Google Sheet")
            return df[['question', 'answer']]
            
        except HttpError as e:
            logger.error(f"HTTP error occurred: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading sheet data: {str(e)}")
            raise
