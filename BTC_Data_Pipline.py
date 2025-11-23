import os
import zipfile
import tempfile
import logging
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BTCDataLoader:
    """
    Class to download (optional), load, clean, resample and save BTC minute-level data.
    
    Usage:
        loader = BTCDataLoader(
            kaggle_dataset="mczielinski/bitcoin-historical-data",
            csv_filename="btcusd_1-min_data.csv"
        )
        df_hourly, df_daily = loader.load_and_clean(save_dir="Datasets")
    
    Parameters:
    - kaggle_dataset: optional Kaggle dataset identifier (string passed to kagglehub.dataset_download)
                       If None, you must provide local_path pointing to a CSV or ZIP.
    - local_path: optional path to an existing CSV or ZIP (if you already downloaded).
    - csv_filename: expected CSV file name inside the dataset/zip (default "btcusd_1-min_data.csv").
    - timestamp_col: name of the unix-timestamp column (default "Timestamp").
    """
    def __init__(
        self,
        kaggle_dataset: Optional[str] = None,
        local_path: Optional[str] = None,
        csv_filename: str = "btcusd_1-min_data.csv",
        timestamp_col: str = "Timestamp",
    ):
        self.kaggle_dataset = kaggle_dataset
        self.local_path = local_path
        self.csv_filename = csv_filename
        self.timestamp_col = timestamp_col
        self._temp_dir = None
        self._csv_path = None

    def _download_from_kaggle(self) -> str:
        """
        Download using kagglehub.dataset_download(...) like in your original script.
        Returns path to downloaded file or directory (may be a zip).
        """
        try:
            import kagglehub  # keep import inside function so it's optional
        except Exception as e:
            raise RuntimeError(
                "kagglehub is not available. Either install it or provide local_path pointing to the CSV/zip."
            ) from e

        if not self.kaggle_dataset:
            raise ValueError("kaggle_dataset must be set to download from Kaggle.")

        logger.info("Downloading dataset from Kaggle: %s", self.kaggle_dataset)
        path = kagglehub.dataset_download(self.kaggle_dataset)
        logger.info("Kaggle download returned path: %s", path)
        return path

    def _locate_csv(self, path: str) -> str:
        """
        Given a path (file, zip, or directory), find or extract csv_filename and return absolute path to CSV.
        """
        path = os.path.abspath(path)
        # If it's a directory, search for CSV
        if os.path.isdir(path):
            candidate = os.path.join(path, self.csv_filename)
            if os.path.exists(candidate):
                return candidate
            # search recursively
            for root, _, files in os.walk(path):
                if self.csv_filename in files:
                    return os.path.join(root, self.csv_filename)
            raise FileNotFoundError(f"{self.csv_filename} not found inside directory {path}")

        # If it's a zip file, extract
        if zipfile.is_zipfile(path):
            self._temp_dir = tempfile.mkdtemp(prefix="btcdata_")
            logger.info("Extracting zip to temp dir: %s", self._temp_dir)
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(self._temp_dir)
            candidate = os.path.join(self._temp_dir, self.csv_filename)
            if os.path.exists(candidate):
                return candidate
            # search extracted
            for root, _, files in os.walk(self._temp_dir):
                if self.csv_filename in files:
                    return os.path.join(root, self.csv_filename)
            raise FileNotFoundError(f"{self.csv_filename} not found inside zip {path}")

        # If it's a plain file path (CSV)
        if os.path.isfile(path) and path.endswith(".csv"):
            return path

        raise FileNotFoundError(f"Could not locate CSV. Provided path: {path}")

    def _read_csv(self, csv_path: str) -> pd.DataFrame:
        logger.info("Reading CSV: %s", csv_path)
        df = pd.read_csv(csv_path)
        return df

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform the cleaning steps you specified:
        - convert unix timestamp to datetime (unit='s')
        - rename column to Date and set index
        - drop duplicate timestamps (keep first)
        - forward/backfill missing values
        - drop rows with Volume <= 0
        """
        if self.timestamp_col not in df.columns:
            raise KeyError(f"Timestamp column '{self.timestamp_col}' not found in CSV.")

        logger.info("Converting timestamp column '%s' to datetime", self.timestamp_col)
        df["Date"] = pd.to_datetime(df[self.timestamp_col], unit="s", errors="coerce")
        df = df.drop(columns=[self.timestamp_col])

        df = df.set_index("Date")
        logger.info("Initial rows: %d", len(df))

        # drop duplicate indices
        dup_count = df.index.duplicated().sum()
        if dup_count:
            logger.info("Dropping %d duplicate timestamp rows (keeping first).", dup_count)
            df = df[~df.index.duplicated(keep="first")]

        # ffill then bfill
        df = df.ffill().bfill()

        # remove zero or negative volume rows
        if "Volume" in df.columns:
            before_vol = len(df)
            df = df[df["Volume"] > 0]
            logger.info("Removed %d rows with non-positive Volume.", before_vol - len(df))
        else:
            logger.warning("'Volume' column not found; skipping volume filtering.")

        logger.info("Rows after cleaning: %d", len(df))
        return df

    def _resample(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Resample with OHLCV aggregation.
        rule examples: "1H", "1D", "1T", etc.
        """
        logger.info("Resampling to rule: %s", rule)
        agg = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
        df_resampled = df.resample(rule).agg(agg).dropna()
        logger.info("Rows after resampling (%s): %d", rule, len(df_resampled))
        return df_resampled

    def save(self, df_hourly: pd.DataFrame, df_daily: pd.DataFrame, save_dir: str = "Datasets") -> Tuple[str, str]:
        os.makedirs(save_dir, exist_ok=True)
        output_hourly = os.path.join(save_dir, "clean_btcusd_hourly.csv")
        output_daily = os.path.join(save_dir, "clean_btcusd_daily.csv")
        df_hourly.to_csv(output_hourly)
        df_daily.to_csv(output_daily)
        logger.info("Saved hourly cleaned dataset: %s", output_hourly)
        logger.info("Saved daily cleaned dataset: %s", output_daily)
        return output_hourly, output_daily

    def load_and_clean(
        self,
        resample_hours: int = 1,
        resample_days: int = 1,
        save_dir: Optional[str] = None,
        force_local_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline convenience method.

        Parameters:
        - resample_hours: integer e.g. 1 for "1H"
        - resample_days: integer e.g. 1 for "1D"
        - save_dir: optional directory to save cleaned CSVs (if None, won't save)
        - force_local_path: optional path to a local CSV/zip to use instead of downloading

        Returns: (df_hourly, df_daily)
        """
        # Decide source path
        source_path = None
        if force_local_path:
            source_path = force_local_path
        elif self.local_path:
            source_path = self.local_path
        elif self.kaggle_dataset:
            source_path = self._download_from_kaggle()
        else:
            raise ValueError("No data source: set kaggle_dataset or local_path, or provide force_local_path.")

        # find csv
        self._csv_path = self._locate_csv(source_path)

        # load
        df_raw = self._read_csv(self._csv_path)

        # clean
        df_clean = self._clean_dataframe(df_raw)

        # resample
        hourly_rule = f"{resample_hours}H"
        daily_rule = f"{resample_days}D"
        df_hourly = self._resample(df_clean, hourly_rule)
        df_daily = self._resample(df_clean, daily_rule)

        # optionally save
        if save_dir:
            self.save(df_hourly, df_daily, save_dir=save_dir)

        return df_hourly, df_daily

    # convenience getters if you already have a cleaned df in memory
    @staticmethod
    def resample_from_df(df_clean: pd.DataFrame, hours: int = 1, days: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        loader = BTCDataLoader()
        hourly = loader._resample(df_clean, f"{hours}H")
        daily = loader._resample(df_clean, f"{days}D")
        return hourly, daily
