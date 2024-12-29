from pathlib import Path
from robot.libraries.BuiltIn import BuiltIn  # type: ignore
from robot.libraries.BuiltIn import RobotNotRunningError
from RPA.RobotLogListener import RobotLogListener
from RPA.utils import Directory
from RPA.utils import File
from typing import Any
from typing import Dict
from typing import Hashable
from typing import List
from typing import Optional
from typing import Union
import json


JSONValue = Optional[Union[str, int, float, bool, List[Any], Dict[Any, Any]]]
JSONType = Union[Dict[Hashable, JSONValue], List[JSONValue], JSONValue]


def json_dumps(payload: JSONType, **kwargs: Any) -> str:
    """Create JSON string in UTF-8 encoding."""
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(payload, **kwargs)


def is_json_equal(left: JSONType, right: JSONType) -> bool:
    """Deep-compare two output JSONs."""
    return json_dumps(left, sort_keys=True) == json_dumps(right, sort_keys=True)


def truncate(text: str, size: int) -> str:
    """Truncate a string from the middle."""
    if len(text) <= size:
        return text

    ellipsis = " ... "
    segment = (size - len(ellipsis)) // 2
    return text[:segment] + ellipsis + text[-segment:]


def resolve_path(path: str) -> Path:
    """Resolve a string-based path, and replace variables."""
    try:
        safe = str(path).replace("\\", "\\\\")
        path = BuiltIn().replace_variables(safe)
    except RobotNotRunningError:
        pass

    return Path(path).expanduser().resolve()


def get_dot_value(source: Dict[Any, Any], key: str) -> Any:
    """Returns the end value from `source` dictionary given `key` traversal."""
    keys = key.split(".")
    value: Any = source
    for _key in keys:
        assert value is not None
        value = value.get(_key)
    return value


def set_dot_value(source: Dict[Any, Any], key: str, *, value: Any) -> None:
    """Sets the end `value` into `source` dictionary given `key` destination."""
    keys = key.rsplit(".", 1)  # one or at most two parts
    if len(keys) == 2:
        source = get_dot_value(source, keys[0])
    source[keys[-1]] = value


def protect_keywords(base: str, keywords: List[str]) -> None:
    """Protects from logging a list of `keywords` relative to `base`."""
    to_protect = [f"{base}.{keyword}" for keyword in keywords]
    listener = RobotLogListener()
    listener.register_protected_keywords(to_protect)


def find_files(
    pattern: Union[str, Path],
    include_dirs: bool = True,
    include_files: bool = True,
) -> List[Union[File, Directory]]:
    """Find files recursively according to a pattern.

    :param pattern:         search path in glob format pattern,
                            e.g. *.xls or **/orders.txt
    :param include_dirs:    include directories in results (defaults to True)
    :param include_files:   include files in results (defaults to True)
    :return:                list of paths that match the pattern

    Example:
    .. code-block:: robotframework

        *** Tasks  ***
        Finding files recursively
            ${files}=    Find files    **/*.log
            FOR    ${file}    IN    @{files}
                Read file    ${file}
            END

    """
    pattern = Path(pattern)

    if pattern.is_absolute():
        root = Path(pattern.anchor)
        parts = pattern.parts[1:]
    else:
        root = Path.cwd()
        parts = pattern.parts

    pattern = str(Path(*parts))
    matches: List[Union[File, Directory]] = []
    for path in root.glob(pattern):
        if path == root:
            continue

        if path.is_dir() and include_dirs:
            matches.append(Directory.from_path(path))
        elif path.is_file() and include_files:
            matches.append(File.from_path(path))

    return sorted(matches)
