import json
from pathlib import Path
from pydantic import validate_arguments


@validate_arguments
def verify_split(old_path: Path, new_path: Path):
    '''
    Verify that no keys from an old split are present in a different new split.

    .. highlight:: python
    .. code-block:: python

        verify_split(
            "path/to/old/split.json",
            "path/to/new/split.json",
        )

    '''
    for old_split_name, old_split in json.loads(old_path.read_text()).items():
        for new_split_name, new_split in json.loads(new_path.read_text()).items():
            if (
                old_split_name != new_split_name
                and len(set(old_split).intersection(set(new_split))) > 0
            ):
                raise ValueError(
                    f'Some keys from old split "{old_split_name}"'
                    f' are present in new split "{new_split_name}":\n'
                    + str("\n".join(
                        [str(old_split[index]) for index in range(min(10, len(old_split)))]
                        + (["..."] if len(old_split) > 10 else [])
                    ))
                )
