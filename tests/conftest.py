from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def toy_catalog() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "toy_catalog.csv")
