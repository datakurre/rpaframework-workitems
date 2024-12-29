from collections.abc import Iterable
from pathlib import Path
from typing import Any
from typing import NamedTuple


def is_dict_like(obj: Any) -> bool:
    """Check if `obj` behaves like a dictionary."""
    return all(
        hasattr(obj, attr) for attr in ("__getitem__", "keys", "__contains__")
    ) and not isinstance(obj, type)


def is_list_like(obj: Any) -> bool:
    """Check if `obj` behaves like a list."""
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


def is_namedtuple(obj: Any) -> bool:
    """Check if `obj` is a namedtuple."""
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


class File(NamedTuple):
    """Robot Framework -friendly container for files."""

    path: str
    name: str
    size: int
    mtime: float

    def __str__(self) -> str:
        return self.path

    def __fspath__(self) -> str:
        # os.PathLike interface
        return self.path

    @classmethod
    def from_path(cls, path: Path) -> "File":
        """Create a File object from pathlib.Path or a path string."""
        path = Path(path)
        stat = path.stat()
        return cls(
            path=str(path.resolve()),
            name=path.name,
            size=stat.st_size,
            mtime=stat.st_mtime,
        )


class Directory(NamedTuple):
    """Robot Framework -friendly container for directories."""

    path: str
    name: str

    def __str__(self) -> str:
        return self.path

    def __fspath__(self) -> str:
        # os.PathLike interface
        return self.path

    @classmethod
    def from_path(cls, path: Path) -> "Directory":
        """Create a directory object from pathlib.Path or a path string."""
        path = Path(path)
        return cls(str(path.resolve()), path.name)
