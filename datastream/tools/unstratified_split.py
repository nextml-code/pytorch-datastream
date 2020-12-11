from typing import Dict, Optional
from pathlib import Path
from datastream import tools


def unstratified_split(
    dataset,
    key_column: str,
    proportions: Dict[str, float],
    filepath: Optional[Path] = None,
    seed: Optional[int] = None,
    frozen: Optional[bool] = False,
):
    split_dataframes = tools.numpy_seed(seed)(tools.split_dataframes)
    return {
        split_name: dataset.replace(
            dataframe=dataframe,
            length=len(dataframe),
        )
        for split_name, dataframe in split_dataframes(
            dataset.dataframe,
            key_column,
            proportions,
            filepath=filepath,
            frozen=frozen,
        ).items()
    }
