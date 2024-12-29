from contextlib import contextmanager
from pathlib import Path
from pytest import FixtureRequest
from pytest import MonkeyPatch
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import copy
import json
import os
import tempfile


try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import suppress as nullcontext  # type: ignore

from . import RESOURCES_DIR
from . import RESULTS_DIR
from RPA.Robocorp.WorkItems import BaseAdapter
from RPA.Robocorp.WorkItems import EmptyQueue
from RPA.Robocorp.WorkItems import ENCODING
from RPA.Robocorp.WorkItems import Error
from RPA.Robocorp.WorkItems import FileAdapter
from RPA.Robocorp.WorkItems import State
from RPA.Robocorp.WorkItems import WorkItem
from RPA.Robocorp.WorkItems import WorkItems
import pytest


VARIABLES_FIRST = {"username": "testguy", "address": "guy@company.com"}
VARIABLES_SECOND = {"username": "another", "address": "dude@company.com"}
IN_OUT_ID = "workitem-id-out"
VALID_DATA = {
    "workitem-id-first": VARIABLES_FIRST,
    "workitem-id-second": VARIABLES_SECOND,
    IN_OUT_ID: [1, 2, 3],
}
VALID_FILES = {
    "workitem-id-first": {
        "file1.txt": b"data1",
        "file2.txt": b"data2",
        "file3.png": b"data3",
    },
    "workitem-id-second": {},
    IN_OUT_ID: {},
}
ITEMS_JSON = [{"payload": {"a-key": "a-value"}, "files": {"a-file": "file.txt"}}]
FAILURE_ATTRIBUTES = {"status": "FAIL", "message": "The task/suite has failed"}

OUTPUT_DIR = RESULTS_DIR / "output_dir"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def temp_filename(
    content: Optional[bytes] = None, **kwargs: Any
) -> Generator[str, None, None]:
    """Create temporary file and yield file relative path, then delete it afterwards.
    Needs to close file handle, since Windows won't allow multiple
    open handles to the same file.
    """
    with tempfile.NamedTemporaryFile(delete=False, **kwargs) as fd:
        path = fd.name
        if content:
            fd.write(content)

    try:
        yield path
    finally:
        os.unlink(path)


def is_equal_files(lhs: Union[Path, str], rhs: Union[Path, str]) -> bool:
    lhs = Path(lhs).resolve()  # pyright: ignore
    rhs = Path(rhs).resolve()  # pyright: ignore
    return lhs == rhs


class MockAdapter(BaseAdapter):
    DATA: Dict[str, Any] = {}
    FILES: Dict[str, Any] = {}
    INDEX = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._data_keys: List[str] = []
        self.releases: List[Any] = []

    @classmethod
    def validate(cls, item: WorkItem, key: str, val: Any) -> None:
        assert item.id
        data = cls.DATA.get(item.id)
        assert data is not None
        assert data[key] == val

    @property
    def data_keys(self) -> List[str]:
        if not self._data_keys:
            self._data_keys = list(self.DATA.keys())
        return self._data_keys

    def reserve_input(self) -> Any:
        if self.INDEX >= len(self.data_keys):
            raise EmptyQueue("No work items in the input queue")

        try:
            return self.data_keys[self.INDEX]
        finally:
            self.INDEX += 1

    def release_input(
        self, item_id: str, state: State, exception: Optional[Any] = None
    ) -> None:
        self.releases.append((item_id, state, exception))  # purely for testing purposes

    def create_output(self, parent_id: str, payload: Any = None) -> str:
        self.save_payload(IN_OUT_ID, payload)
        return IN_OUT_ID

    def load_payload(self, item_id: str) -> Any:
        return self.DATA[item_id]

    def save_payload(self, item_id: str, payload: Any) -> None:
        self.DATA[item_id] = payload

    def list_files(self, item_id: str) -> Any:
        return self.FILES[item_id]

    def get_file(self, item_id: str, name: str) -> Any:
        return self.FILES[item_id][name]

    def add_file(
        self, item_id: str, name: str, *, original_name: str, content: bytes
    ) -> None:
        self.FILES[item_id][name] = content

    def remove_file(self, item_id: str, name: str) -> None:
        del self.FILES[item_id][name]


class TestLibrary:
    """Tests the library itself as a whole."""

    @staticmethod
    @pytest.fixture
    def adapter() -> Generator[type[MockAdapter], None, None]:
        MockAdapter.DATA = copy.deepcopy(VALID_DATA)
        MockAdapter.FILES = copy.deepcopy(VALID_FILES)
        try:
            yield MockAdapter
        finally:
            MockAdapter.DATA = {}
            MockAdapter.FILES = {}
            MockAdapter.INDEX = 0

    @staticmethod
    @pytest.fixture
    def library(adapter: type[BaseAdapter]) -> Generator[WorkItems, None, None]:
        yield WorkItems(default_adapter=adapter)

    @staticmethod
    def _get_resource_data(name: str, binary: bool = False) -> Union[str, bytes]:
        path = RESOURCES_DIR / "work-items" / name
        if binary:
            return path.read_bytes()

        return path.read_text(encoding=ENCODING)

    @classmethod
    @pytest.fixture(
        params=[
            ("mail-text.txt", "A message from e-mail"),
            ("mail-json.txt", {"message": "from email"}),
            ("mail-yaml.txt", {"message": "from email", "extra": {"value": 1}}),
        ]
    )
    def raw_email_data(cls, request: FixtureRequest) -> Tuple[Union[bytes, str], Any]:
        raw_email = cls._get_resource_data(request.param[0])
        expected_body = request.param[1]
        return raw_email, expected_body

    @classmethod
    @pytest.fixture(
        params=[
            ("email.text", False, "A message from e-mail"),
            ("__mail.html", True, "from email"),
        ]
    )
    def parsed_email_data(
        cls, request: FixtureRequest
    ) -> Tuple[str, Union[bytes, str], str]:
        email_var = request.param[0]
        parsed_email = None
        expected_body = request.param[2]
        if request.param[1]:
            parsed_email = cls._get_resource_data(email_var, binary=True)
        assert parsed_email is not None
        return email_var, parsed_email, expected_body

    def test_autoload(self, library: WorkItems) -> None:
        # Called by Robot Framework listener
        library._start_suite(None, None)

        # Work item loaded using env variables
        env = library.current
        assert env is not None
        assert env.payload == VARIABLES_FIRST

    def test_autoload_disable(self, adapter: type[BaseAdapter]) -> None:
        library = WorkItems(default_adapter=adapter, autoload=False)

        # Called by Robot Framework listener
        library._start_suite(None, None)
        assert library._current is None

    @pytest.mark.parametrize("end_hook", ["_end_test", "_end_suite"])
    def test_autorelease(self, library: WorkItems, end_hook: str) -> None:
        library.get_input_work_item()
        end_method = getattr(library, end_hook)
        end_method("My Failure Name", FAILURE_ATTRIBUTES)
        releases = library.adapter.releases  # type: ignore
        assert len(releases) == 1
        assert releases[0][2] == {
            "type": "APPLICATION",
            "code": None,
            "message": "The task/suite has failed",
        }

    def test_keyword_get_input_work_item(self, library: WorkItems) -> None:
        first = library.get_input_work_item()
        assert first.payload == VARIABLES_FIRST
        assert first == library.current

        second = library.get_input_work_item()
        assert second.payload == VARIABLES_SECOND
        assert second == library.current

    def test_keyword_save_work_item(self, library: WorkItems) -> None:
        item = library.get_input_work_item()
        for key, value in VARIABLES_FIRST.items():
            MockAdapter.validate(item, key, value)

        modified = {"username": "changed", "address": "dude@company.com"}
        item.payload = modified

        library.save_work_item()
        for key, value in modified.items():
            MockAdapter.validate(item, key, value)

    def test_no_active_item(self) -> None:
        library = WorkItems(default_adapter=MockAdapter)
        with pytest.raises(RuntimeError) as err:
            library.save_work_item()

        assert str(err.value) == "No active work item"

    def test_list_variables(self, library: WorkItems) -> None:
        library.get_input_work_item()

        names = library.list_work_item_variables()

        assert len(names) == 2
        assert "username" in names
        assert "address" in names

    def test_get_variables(self, library: WorkItems) -> None:
        library.get_input_work_item()

        value = library.get_work_item_variable("username")
        assert value == "testguy"

        with pytest.raises(KeyError):
            library.get_work_item_variable("notexist")

    def test_get_variables_default(self, library: WorkItems) -> None:
        library.get_input_work_item()

        value = library.get_work_item_variable("username", default="doesntmatter")
        assert value == "testguy"

        value = library.get_work_item_variable("notexist", default="doesmatter")
        assert value == "doesmatter"

    def test_delete_variables(self, library: WorkItems) -> None:
        library.get_input_work_item()
        assert "username" in library.list_work_item_variables()

        library.delete_work_item_variables("username")
        assert "username" not in library.list_work_item_variables()

        library.delete_work_item_variables("doesntexist")

        with pytest.raises(KeyError):
            library.delete_work_item_variables("doesntexist", force=False)

    def test_delete_variables_single(self, library: WorkItems) -> None:
        library.get_input_work_item()

        assert "username" in library.list_work_item_variables()
        assert len(library.current.payload) == 2  # type: ignore

        library.delete_work_item_variables("username")

        assert "username" not in library.list_work_item_variables()
        assert len(library.current.payload) == 1  # type: ignore

    def test_delete_variables_multiple(self, library: WorkItems) -> None:
        library.get_input_work_item()

        names = library.list_work_item_variables()
        assert "username" in names
        assert "address" in names
        assert len(names) == 2

        library.delete_work_item_variables("username", "address")

        names = library.list_work_item_variables()
        assert "username" not in names
        assert "username" not in names
        assert len(names) == 0

    def test_delete_variables_unknown(self, library: WorkItems) -> None:
        library.get_input_work_item()
        assert len(library.list_work_item_variables()) == 2

        library.delete_work_item_variables("unknown-variable")
        assert len(library.list_work_item_variables()) == 2

        with pytest.raises(KeyError):
            library.delete_work_item_variables("unknown-variable", force=False)
        assert len(library.list_work_item_variables()) == 2

    def test_raw_payload(self, library: WorkItems) -> None:
        _ = library.get_input_work_item()
        _ = library.get_input_work_item()
        item = library.get_input_work_item()

        payload = library.get_work_item_payload()
        assert payload == [1, 2, 3]

        library.set_work_item_payload({"output": 0xBEEF})
        library.save_work_item()
        MockAdapter.validate(item, "output", 0xBEEF)

    def test_list_files(self, library: WorkItems) -> None:
        library.get_input_work_item()

        files = library.list_work_item_files()
        assert files == ["file1.txt", "file2.txt", "file3.png"]

    def test_get_file(self, library: WorkItems) -> None:
        library.get_input_work_item()

        with temp_filename() as path:
            result = library.get_work_item_file("file2.txt", path)
            with open(result) as fd:
                data = fd.read()

            assert is_equal_files(result, path)
            assert data == "data2"

    def test_get_file_notexist(self, library: WorkItems) -> None:
        library.get_input_work_item()

        with pytest.raises(FileNotFoundError):
            library.get_work_item_file("file5.txt")

    def test_add_file(self, library: WorkItems) -> None:
        item = library.get_input_work_item()
        assert item.id

        with temp_filename(b"some-input-content") as path:
            library.add_work_item_file(path, "file4.txt")

            files = library.list_work_item_files()
            assert files == ["file1.txt", "file2.txt", "file3.png", "file4.txt"]
            assert "file4.txt" not in MockAdapter.FILES[item.id]

            library.save_work_item()
            assert MockAdapter.FILES[item.id]["file4.txt"] == b"some-input-content"

    def test_add_file_duplicate(self, library: WorkItems) -> None:
        item = library.get_input_work_item()
        assert item.id

        def verify_files() -> None:
            files = library.list_work_item_files()
            assert files == ["file1.txt", "file2.txt", "file3.png", "file4.txt"]

        with temp_filename(b"some-input-content") as path:
            library.add_work_item_file(path, "file4.txt")
            assert "file4.txt" not in MockAdapter.FILES[item.id]
            verify_files()

            # Add duplicate for unsaved item
            library.add_work_item_file(path, "file4.txt")
            assert "file4.txt" not in MockAdapter.FILES[item.id]
            verify_files()

            library.save_work_item()
            assert MockAdapter.FILES[item.id]["file4.txt"] == b"some-input-content"
            verify_files()

            # Add duplicate for saved item
            library.add_work_item_file(path, "file4.txt")
            verify_files()

            library.save_work_item()
            verify_files()

    def test_add_file_notexist(self, library: WorkItems) -> None:
        library.get_input_work_item()

        with pytest.raises(FileNotFoundError):
            library.add_work_item_file("file5.txt", "doesnt-matter")

    def test_remove_file(self, library: WorkItems) -> None:
        item = library.get_input_work_item()
        assert item.id

        library.remove_work_item_file("file2.txt")

        files = library.list_work_item_files()
        assert files == ["file1.txt", "file3.png"]
        assert "file2.txt" in MockAdapter.FILES[item.id]

        library.save_work_item()
        assert "file2.txt" not in MockAdapter.FILES[item.id]

    def test_remove_file_notexist(self, library: WorkItems) -> None:
        library.get_input_work_item()

        library.remove_work_item_file("file5.txt")

        with pytest.raises(FileNotFoundError):
            library.remove_work_item_file("file5.txt", missing_ok=False)

    def test_get_file_pattern(self, library: WorkItems) -> None:
        library.get_input_work_item()

        with tempfile.TemporaryDirectory() as outdir:
            file1 = os.path.join(outdir, "file1.txt")
            file2 = os.path.join(outdir, "file2.txt")

            paths = library.get_work_item_files("*.txt", outdir)
            assert is_equal_files(paths[0], file1)
            assert is_equal_files(paths[1], file2)
            assert os.path.exists(file1)
            assert os.path.exists(file2)

    def test_remove_file_pattern(self, library: WorkItems) -> None:
        item = library.get_input_work_item()
        assert item.id

        library.remove_work_item_files("*.txt")

        files = library.list_work_item_files()
        assert files == ["file3.png"]
        assert list(MockAdapter.FILES[item.id]) == [
            "file1.txt",
            "file2.txt",
            "file3.png",
        ]

        library.save_work_item()

        files = library.list_work_item_files()
        assert files == ["file3.png"]
        assert list(MockAdapter.FILES[item.id]) == ["file3.png"]

    def test_clear_work_item(self, library: WorkItems) -> None:
        library.get_input_work_item()

        library.clear_work_item()
        library.save_work_item()

        assert library.get_work_item_payload() == {}
        assert library.list_work_item_files() == []

    def test_get_file_unsaved(self, library: WorkItems) -> None:
        library.get_input_work_item()

        with temp_filename(b"some-input-content") as path:
            library.add_work_item_file(path, "file4.txt")

            files = library.list_work_item_files()
            assert files == ["file1.txt", "file2.txt", "file3.png", "file4.txt"]
            assert "file4.txt" not in MockAdapter.FILES

            with tempfile.TemporaryDirectory() as outdir:
                names = ["file1.txt", "file2.txt", "file4.txt"]
                result = library.get_work_item_files("*.txt", outdir)
                expected = [os.path.join(outdir, name) for name in names]
                for lhs, rhs in zip(result, expected):
                    assert is_equal_files(lhs, rhs)
                with open(result[-1]) as fd:
                    assert fd.read() == "some-input-content"

    def test_get_file_unsaved_no_copy(self, library: WorkItems) -> None:
        library.get_input_work_item()

        with tempfile.TemporaryDirectory() as outdir:
            path = os.path.join(outdir, "nomove.txt")
            with open(path, "w") as fd:
                fd.write("my content")

            mtime = os.path.getmtime(path)
            library.add_work_item_file(path)

            files = library.list_work_item_files()
            assert files == ["file1.txt", "file2.txt", "file3.png", "nomove.txt"]

            paths = library.get_work_item_files("*.txt", outdir)
            assert is_equal_files(paths[-1], path)
            assert os.path.getmtime(path) == mtime

    def test_get_file_unsaved_relative(self, library: WorkItems) -> None:
        library.get_input_work_item()

        with tempfile.TemporaryDirectory() as outdir:
            curdir = os.getcwd()
            try:
                os.chdir(outdir)
                with open("nomove.txt", "w") as fd:
                    fd.write("my content")

                mtime = os.path.getmtime("nomove.txt")
                library.add_work_item_file(os.path.join(outdir, "nomove.txt"))

                files = library.list_work_item_files()
                assert files == ["file1.txt", "file2.txt", "file3.png", "nomove.txt"]

                paths = library.get_work_item_files("*.txt")
                assert is_equal_files(paths[-1], os.path.join(outdir, "nomove.txt"))
                assert os.path.getmtime("nomove.txt") == mtime
            finally:
                os.chdir(curdir)

    def test_get_file_no_matches(self, library: WorkItems) -> None:
        library.get_input_work_item()

        with tempfile.TemporaryDirectory() as outdir:
            paths = library.get_work_item_files("*.pdf", outdir)
            assert len(paths) == 0

    def test_create_output_work_item(self, library: WorkItems) -> None:
        input_item = library.get_input_work_item()
        output_item = library.create_output_work_item()

        assert output_item.id is None
        assert output_item.parent_id == input_item.id

    def test_create_output_work_item_no_input(self, library: WorkItems) -> None:
        with pytest.raises(RuntimeError):
            library.create_output_work_item()

    @staticmethod
    @pytest.fixture(
        params=[
            lambda *files: files,  # files provided as tuple
            lambda *files: list(files),  # as list of paths
            lambda *files: ", ".join(files),  # comma separated paths
        ]
    )
    def out_files(request: FixtureRequest) -> Generator[Any, None, None]:
        """Output work item files."""
        with (
            temp_filename(b"out-content-1", suffix="-1.txt") as path1,
            temp_filename(b"out-content-2", suffix="-2.txt") as path2,
        ):
            func = request.param
            yield func(path1, path2)

    def test_create_output_work_item_variables_files(
        self, library: WorkItems, out_files: List[str]
    ) -> None:
        library.get_input_work_item()
        variables = {"my_var1": "value1", "my_var2": "value2"}
        library.create_output_work_item(variables=variables, files=out_files, save=True)

        assert library.get_work_item_variable("my_var1") == "value1"
        assert library.get_work_item_variable("my_var2") == "value2"

        # This actually "downloads" (creates) the files, so make sure we remove them
        #  afterwards.
        paths = library.get_work_item_files("*.txt", dirname=OUTPUT_DIR)
        try:
            assert len(paths) == 2
            for path in paths:
                with open(path) as stream:
                    content = stream.read()
                idx = Path(path).stem.split("-")[-1]
                assert content == f"out-content-{idx}"
        finally:
            for path in paths:
                os.remove(path)

    def test_custom_root(self, adapter: type[BaseAdapter]) -> None:
        library = WorkItems(default_adapter=adapter, root="vars")
        item = library.get_input_work_item()

        variables = library.get_work_item_variables()
        assert variables == {}

        library.set_work_item_variables(cool="beans", yeah="boi")
        assert item.payload == {
            **VARIABLES_FIRST,
            "vars": {"cool": "beans", "yeah": "boi"},
        }

    @pytest.mark.parametrize("limit", [0, 1, 2, 3, 4])  # no, existing and over limit
    def test_iter_work_items(self, library: WorkItems, limit: int) -> None:
        usernames = []

        def func(a: int, b: int, r: int = 3) -> bool:
            assert a + b == r
            # Collects the "username" variable from the payload if provided and returns
            #   True if found, False otherwise.
            payload = library.get_work_item_payload()
            if not isinstance(payload, dict):
                return False

            username = payload.get("username")
            if username:
                usernames.append(username)

            return username is not None

        library.get_input_work_item()
        results = library.for_each_input_work_item(func, 1, 2, items_limit=limit, r=3)

        expected_usernames = ["testguy", "another"]
        expected_results = [True, True, False]
        if limit:
            expected_usernames = expected_usernames[:limit]
            expected_results = expected_results[:limit]
        assert usernames == expected_usernames
        assert results == expected_results

    def test_iter_work_items_limit_and_state(self, library: WorkItems) -> None:
        def func() -> int:
            return 1

        # Pick one single item and make sure its state is set implicitly.
        results: Optional[List[Any]] = library.for_each_input_work_item(
            func, items_limit=1
        )
        assert results
        assert len(results) == 1
        assert library.current.state is State.DONE

        def func2() -> int:
            library.release_input_work_item(State.FAILED)
            return 2

        # Pick-up the rest of the two inputs and set state explicitly.
        results = library.for_each_input_work_item(func2)
        assert results
        assert len(results) == 2
        assert library.current.state is State.FAILED  # type: ignore

    @pytest.mark.parametrize("return_results", [True, False])
    def test_iter_work_items_return_results(
        self, library: WorkItems, return_results: bool
    ) -> None:
        def func() -> int:
            return 1

        library.get_input_work_item()
        results: Optional[List[Any]] = library.for_each_input_work_item(
            func, return_results=return_results
        )
        if return_results:
            assert results == [1] * 3
        else:
            assert results is None

    @pytest.mark.parametrize("processed_items", [0, 1, 2, 3])
    def test_successive_work_items_iteration(
        self, library: WorkItems, processed_items: int
    ) -> None:
        for _ in range(processed_items):
            library.get_input_work_item()
            library.release_input_work_item(State.DONE)

        def func() -> None:
            pass

        # Checks if all remaining input work items are processed once.
        results: Any = library.for_each_input_work_item(func)
        assert len(results) == 3 - processed_items

        # Checks if there's no double processing of the last already processed item.
        results = library.for_each_input_work_item(func)
        assert len(results) == 0

    @staticmethod
    @pytest.fixture(
        params=[
            None,
            {"exception_type": "BUSINESS"},
            {
                "exception_type": "APPLICATION",
                "code": "UNEXPECTED_ERROR",
                "message": "This is an unexpected error",
            },
            {
                "exception_type": "APPLICATION",
                "code": None,
                "message": "This is an unexpected error",
            },
            {
                "exception_type": "APPLICATION",
                "code": None,
                "message": None,
            },
            {
                "exception_type": None,
                "code": None,
                "message": None,
            },
            {
                "exception_type": None,
                "code": "APPLICATION",
                "message": None,
            },
            {
                "exception_type": None,
                "code": "",
                "message": "Not empty",
            },
            {
                "exception_type": "application",
                "code": None,
                "message": " ",  # gets discarded during API request
            },
        ]
    )
    def release_exception(request: FixtureRequest) -> Any:
        exception = request.param or {}
        effect: Any = nullcontext()
        success = True
        if not exception.get("exception_type") and any(
            map(lambda key: exception.get(key), ["code", "message"])
        ):
            effect = pytest.raises(RuntimeError)
            success = False
        return exception or None, effect, success

    def test_release_work_item_failed(
        self, library: WorkItems, release_exception: Any
    ) -> None:
        exception, effect, success = release_exception

        library.get_input_work_item()
        with effect:
            library.release_input_work_item(
                "FAILED", **(exception or {})
            )  # intentionally providing a string for the state
        if success:
            assert library.current.state == State.FAILED

        exception_type = (exception or {}).pop("exception_type", None)
        if exception_type:
            exception["type"] = Error(exception_type.upper()).value
            exception.setdefault("code", None)
            exception.setdefault("message", None)
        else:
            exception = None
        if success:
            assert library.adapter.releases == [  # type: ignore
                ("workitem-id-first", State.FAILED, exception)
            ]

    @pytest.mark.parametrize("exception", [None, {"exception_type": Error.APPLICATION}])
    def test_release_work_item_done(
        self, library: WorkItems, exception: Dict[str, Any]
    ) -> None:
        library.get_input_work_item()
        library.release_input_work_item(State.DONE, **(exception or {}))
        assert library.current.state is State.DONE
        assert library.adapter.releases == [  # type: ignore
            # No exception sent for non failures.
            ("workitem-id-first", State.DONE, None)
        ]

    def test_auto_release_work_item(self, library: WorkItems) -> None:
        library.get_input_work_item()
        library.get_input_work_item()  # this automatically sets the state of the last

        assert library.current.state is None  # because the previous one has a state
        assert library.adapter.releases == [  # type: ignore
            ("workitem-id-first", State.DONE, None)
        ]  # pyright: ignore

    @pytest.mark.parametrize(
        "source_var, value",
        [
            ("email", "no e-mail here"),
            ("email", {}),
            ("email", {"field": "something"}),
        ],
    )
    def test_email_var_no_parse(
        self, library: WorkItems, source_var: str, value: str
    ) -> None:
        library.adapter.DATA["workitem-id-first"][source_var] = value  # type: ignore

        library.get_input_work_item()  # auto-parsing should pass without errors
        assert library.get_work_item_variable(source_var) == value  # left untouched


class TestFileAdapter:
    """Tests the local dev env `FileAdapter` on Work Items."""

    @contextmanager
    def _input_work_items(self) -> Generator[Tuple[str, str], None, None]:
        with tempfile.TemporaryDirectory() as datadir:
            items_in = os.path.join(datadir, "items.json")
            with open(items_in, "w") as fd:
                json.dump(ITEMS_JSON, fd)
            with open(os.path.join(datadir, "file.txt"), "w") as fd:
                fd.write("some mock content")

            output_dir = os.path.join(datadir, "output_dir")
            os.makedirs(output_dir)
            items_out = os.path.join(output_dir, "items-out.json")

            yield items_in, items_out

    @pytest.fixture(
        params=[
            ("RPA_WORKITEMS_PATH", "N/A"),
            ("RPA_INPUT_WORKITEM_PATH", "RPA_OUTPUT_WORKITEM_PATH"),
        ]
    )
    def adapter(
        self, monkeypatch: MonkeyPatch, request: FixtureRequest
    ) -> Generator[FileAdapter, None, None]:
        with self._input_work_items() as (items_in, items_out):
            monkeypatch.setenv(request.param[0], items_in)
            monkeypatch.setenv(request.param[1], items_out)
            yield FileAdapter()

    @staticmethod
    @pytest.fixture
    def empty_adapter() -> FileAdapter:
        # No work items i/o files nor envs set.
        return FileAdapter()

    def test_load_data(self, adapter: FileAdapter) -> None:
        item_id = adapter.reserve_input()
        data = adapter.load_payload(item_id)
        assert data == {"a-key": "a-value"}

    def test_list_files(self, adapter: FileAdapter) -> None:
        item_id = adapter.reserve_input()
        files = adapter.list_files(item_id)
        assert files == ["a-file"]

    def test_get_file(self, adapter: FileAdapter) -> None:
        item_id = adapter.reserve_input()
        content = adapter.get_file(item_id, "a-file")
        assert content == b"some mock content"

    def test_add_file(self, adapter: FileAdapter) -> None:
        item_id = adapter.reserve_input()
        adapter.add_file(
            item_id,
            "secondfile.txt",
            original_name="secondfile2.txt",
            content=b"somedata",
        )
        assert adapter.input_path
        assert adapter.inputs[0]["files"]["secondfile.txt"] == "secondfile2.txt"
        assert os.path.isfile(Path(adapter.input_path).parent / "secondfile2.txt")

    def test_save_data_input(self, adapter: FileAdapter) -> None:
        item_id = adapter.reserve_input()
        adapter.save_payload(item_id, {"key": "value"})
        assert adapter.input_path
        with open(adapter.input_path) as fd:
            data = json.load(fd)
            assert data == [
                {"payload": {"key": "value"}, "files": {"a-file": "file.txt"}}
            ]

    def test_save_data_output(self, adapter: FileAdapter) -> None:
        assert adapter.input_path
        item_id = adapter.create_output("0", {})
        adapter.save_payload(item_id, {"key": "value"})

        output: Optional[Union[str, Path]] = os.getenv("RPA_OUTPUT_WORKITEM_PATH")
        if output:
            # checks automatic dir creation
            assert "output_dir" in output  # type: ignore
        else:
            assert adapter.input_path
            output = Path(adapter.input_path).with_suffix(".output.json")

        assert output
        assert os.path.isfile(output)
        with open(output) as fd:
            data = json.load(fd)
            assert data == [{"payload": {"key": "value"}, "files": {}}]

    def test_missing_file(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("RPA_WORKITEMS_PATH", "not-exist.json")
        adapter = FileAdapter()
        assert adapter.inputs == [{"payload": {}}]

    def test_empty_queue(self, monkeypatch: MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as datadir:
            items = os.path.join(datadir, "items.json")
            with open(items, "w") as fd:
                json.dump([], fd)

            monkeypatch.setenv("RPA_WORKITEMS_PATH", items)
            adapter = FileAdapter()
            assert adapter.inputs == [{"payload": {}}]

    def test_malformed_queue(self, monkeypatch: MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as datadir:
            items = os.path.join(datadir, "items.json")
            with open(items, "w") as fd:
                json.dump(["not-an-item"], fd)

            monkeypatch.setenv("RPA_WORKITEMS_PATH", items)
            adapter = FileAdapter()
            assert adapter.inputs == [{"payload": {}}]

    def test_without_items_paths(self, empty_adapter: FileAdapter) -> None:
        assert empty_adapter.inputs == [{"payload": {}}]
        # Can't save inputs nor outputs since there's no path defined for them.
        with pytest.raises(RuntimeError):
            empty_adapter.save_payload("0", {"input": "value"})
        with pytest.raises(RuntimeError):
            _ = empty_adapter.output_path
        with pytest.raises(RuntimeError):
            empty_adapter.create_output("1", {"var": "some-value"})
