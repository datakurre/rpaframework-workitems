from abc import ABC
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from robot.api.deco import keyword
from robot.api.deco import library
from robot.libraries.BuiltIn import BuiltIn  # type: ignore
from RPA.helpers import import_by_name
from RPA.helpers import required_env
from RPA.Robocorp.utils import find_files
from RPA.Robocorp.utils import is_json_equal
from RPA.Robocorp.utils import json_dumps
from RPA.Robocorp.utils import JSONType
from RPA.Robocorp.utils import resolve_path
from RPA.Robocorp.utils import truncate
from RPA.RobotLogListener import deprecation
from shutil import copy2
from threading import Event
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
import copy
import fnmatch
import json
import logging
import os


UNDEFINED = object()  # Undefined default value
ENCODING = "utf-8"


class State(Enum):
    """Work item state. (set when released)"""

    DONE = "COMPLETED"
    FAILED = "FAILED"


class Error(Enum):
    """Failed work item error type."""

    BUSINESS = "BUSINESS"  # wrong/missing data, shouldn't be retried
    APPLICATION = "APPLICATION"  # logic issue/timeout, can be retried


class EmptyQueue(IndexError):
    """Raised when trying to load an input item and none available."""


class BaseAdapter(ABC):
    """Abstract base class for work item adapters."""

    @abstractmethod
    def reserve_input(self) -> str:
        """Get next work item ID from the input queue and reserve it."""
        raise NotImplementedError

    @abstractmethod
    def release_input(
        self, item_id: str, state: State, exception: Optional[Dict[Any, Any]] = None
    ) -> None:
        """Release the lastly retrieved input work item and set state."""
        raise NotImplementedError

    @abstractmethod
    def create_output(self, parent_id: str, payload: Optional[JSONType] = None) -> str:
        """Create new output for work item, and return created ID."""
        raise NotImplementedError

    @abstractmethod
    def load_payload(self, item_id: str) -> JSONType:
        """Load JSON payload from work item."""
        raise NotImplementedError

    @abstractmethod
    def save_payload(self, item_id: str, payload: JSONType) -> None:
        """Save JSON payload to work item."""
        raise NotImplementedError

    @abstractmethod
    def list_files(self, item_id: str) -> List[str]:
        """List attached files in work item."""
        raise NotImplementedError

    @abstractmethod
    def get_file(self, item_id: str, name: str) -> bytes:
        """Read file's contents from work item."""
        raise NotImplementedError

    @abstractmethod
    def add_file(
        self, item_id: str, name: str, *, original_name: str, content: bytes
    ) -> None:
        """Attach file to work item."""
        raise NotImplementedError

    @abstractmethod
    def remove_file(self, item_id: str, name: str) -> None:
        """Remove attached file from work item."""
        raise NotImplementedError


class FileAdapter(BaseAdapter):
    """Adapter for simulating work item input queues.

    Reads inputs from the given database file, and writes
    all created output items into an adjacent file
    with the suffix ``<filename>.output.json``. If the output path is provided by an
    env var explicitly, then the file will be saved with the provided path and name.

    Reads and writes all work item files from/to the same parent
    folder as the given input database.

    Optional environment variables:

    * RPA_INPUT_WORKITEM_PATH:  Path to work items input database file
    * RPA_OUTPUT_WORKITEM_PATH:  Path to work items output database file
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._input_path = UNDEFINED
        self._output_path = UNDEFINED

        self.inputs: List[Dict[str, Any]] = self.load_database()
        self.outputs: List[Dict[str, Any]] = []
        self.index: int = 0

    def _get_item(self, item_id: str) -> Tuple[str, Dict[str, Any]]:
        # The work item ID is analogue to inputs/outputs list queues index.
        idx = int(item_id)
        if idx < len(self.inputs):
            return "input", self.inputs[idx]

        if idx < (len(self.inputs) + len(self.outputs)):
            return "output", self.outputs[idx - len(self.inputs)]

        raise ValueError(f"Unknown work item ID: {item_id}")

    def reserve_input(self) -> str:
        if self.index >= len(self.inputs):
            raise EmptyQueue("No work items in the input queue")

        try:
            return str(self.index)
        finally:
            self.index += 1

    def release_input(
        self, item_id: str, state: State, exception: Optional[Dict[Any, Any]] = None
    ) -> None:
        # Nothing happens for now on releasing local dev input Work Items.
        log_func = logging.error if state == State.FAILED else logging.info
        log_func(
            "Releasing item %r with %s state and exception: %s",
            item_id,
            state.value,
            exception,
        )

    @property
    def input_path(self) -> Optional[Path]:
        if self._input_path is UNDEFINED:
            # pylint: disable=invalid-envvar-default
            old_path = os.getenv("RPA_WORKITEMS_PATH")
            if old_path:
                deprecation(
                    "Work items load - Old path style usage detected, please use the "
                    "'RPA_INPUT_WORKITEM_PATH' env var instead "
                    "(more details under documentation: https://robocorp.com/docs/development-guide/control-room/data-pipeline#developing-with-work-items-locally)"  # noqa: E501
                )
            path = os.getenv("RPA_INPUT_WORKITEM_PATH", default=old_path)
            if path:
                logging.info("Resolving input path: %s", path)
                self._input_path = resolve_path(path)
            else:
                # Will raise `TypeError` during inputs loading and will populate the
                # list with one empty initial input.
                self._input_path = None

        assert self._input_path is None or isinstance(self._input_path, Path)
        return self._input_path

    @property
    def output_path(self) -> Path:
        if self._output_path is UNDEFINED:
            # This is usually set once per loaded input work item.
            new_path = os.getenv("RPA_OUTPUT_WORKITEM_PATH")
            if new_path:
                self._output_path = resolve_path(new_path)
                self._output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                deprecation(
                    "Work items save - Old path style usage detected, please use the "
                    "'RPA_OUTPUT_WORKITEM_PATH' env var instead "
                    "(more details under documentation: https://robocorp.com/docs/development-guide/control-room/data-pipeline#developing-with-work-items-locally)"  # noqa: E501
                )
                if not self.input_path:
                    raise RuntimeError(
                        "You must provide a path for at least one of the input or "
                        "output work items files"
                    )
                self._output_path = self.input_path.with_suffix(".output.json")

        assert isinstance(self._output_path, Path)
        return self._output_path

    def _save_to_disk(self, source: str) -> None:
        if source == "input":
            if not self.input_path:
                raise RuntimeError(
                    "Can't save an input item without a path defined, use "
                    "'RPA_INPUT_WORKITEM_PATH' env for this matter"
                )
            path = self.input_path
            data = self.inputs
        else:
            path = self.output_path
            data = self.outputs

        with open(path, "w", encoding=ENCODING) as fd:
            fd.write(json_dumps(data, indent=4))

        logging.info("Saved into %s file: %s", source, path)

    def create_output(self, _: str, payload: Optional[JSONType] = None) -> str:
        # Note that the `parent_id` is not used during local development.
        item: Dict[str, Any] = {"payload": payload, "files": {}}
        self.outputs.append(item)

        self._save_to_disk("output")
        return str(len(self.inputs) + len(self.outputs) - 1)  # new output work item ID

    def load_payload(self, item_id: str) -> Any:
        _, item = self._get_item(item_id)
        return item.get("payload", {})

    def save_payload(self, item_id: str, payload: JSONType) -> None:
        source, item = self._get_item(item_id)
        item["payload"] = payload
        self._save_to_disk(source)

    def list_files(self, item_id: str) -> List[str]:
        _, item = self._get_item(item_id)
        files = item.get("files", {})
        return list(files.keys())

    def get_file(self, item_id: str, name: str) -> bytes:
        source, item = self._get_item(item_id)
        files = item.get("files", {})

        path = files[name]
        if not Path(path).is_absolute():
            assert self.input_path
            parent = (
                self.input_path.parent if source == "input" else self.output_path.parent
            )
            path = parent / path

        with open(path, "rb") as infile:
            return infile.read()

    def add_file(
        self, item_id: str, name: str, *, original_name: str, content: bytes
    ) -> None:
        source, item = self._get_item(item_id)
        files = item.setdefault("files", {})

        assert self.input_path
        parent = (
            self.input_path.parent if source == "input" else self.output_path.parent
        )
        path = parent / original_name  # the file on disk will keep its original name
        with open(path, "wb") as fd:
            fd.write(content)
        logging.info("Created file: %s", path)
        files[name] = original_name  # file path relative to the work item

        self._save_to_disk(source)

    def remove_file(self, item_id: str, name: str) -> None:
        source, item = self._get_item(item_id)
        files = item.get("files", {})

        path = files[name]
        logging.info("Would remove file: %s", path)
        # Note that the file doesn't get removed from disk as well.
        del files[name]

        self._save_to_disk(source)

    def load_database(self) -> List[Dict[Any, Any]]:
        try:
            try:
                with open(f"{self.input_path}", "r", encoding=ENCODING) as infile:
                    data = json.load(infile)
            except (TypeError, FileNotFoundError):
                logging.warning("No input work items file found: %s", self.input_path)
                data = []

            if isinstance(data, list):
                assert all(
                    isinstance(d, dict) for d in data
                ), "Items should be dictionaries"
                if len(data) == 0:
                    data.append({"payload": {}})
                return data

            # Attempt to migrate from old format
            assert isinstance(data, dict), "Not a list or dictionary"
            deprecation("Work items file as mapping is deprecated")
            workspace = next(iter(data.values()))
            work_item = next(iter(workspace.values()))
            return [{"payload": work_item}]
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Invalid work items file because of: %s", exc)
            return [{"payload": {}}]


class WorkItem:
    """Base class for input and output work items.

    :param adapter:   Adapter instance
    :param item_id:   Work item ID (optional)
    :param parent_id: Parent work item's ID (optional)
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        item_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ):
        #: Adapter for loading/saving content
        self.adapter = adapter
        #: This item's and/or parent's ID
        self.id: Optional[str] = item_id
        self.parent_id: Optional[str] = parent_id
        assert self.id is not None or self.parent_id is not None
        #: Item's state on release; can be set once
        self.state: Optional[State] = None
        #: Remote JSON payload, and queued changes
        self._payload: JSONType = {}
        self._payload_cache: JSONType = {}
        #: Remote attached files, and queued changes
        self._files: List[str] = []
        self._files_to_add: Dict[str, Path] = {}
        self._files_to_remove: List[str] = []

    def __repr__(self) -> str:
        payload = truncate(str(self.payload), 64)
        files = len(self.files)
        return f"WorkItem(id={self.id}, payload={payload}, files={files})"

    @property
    def is_dirty(self) -> bool:
        """Check if work item has unsaved changes."""
        return bool(
            self.id is None
            or not is_json_equal(self._payload, self._payload_cache)
            or self._files_to_add
            or self._files_to_remove
        )

    @property
    def payload(self) -> JSONType:
        return self._payload_cache

    @payload.setter
    def payload(self, value: Any) -> None:
        self._payload_cache = value

    @property
    def files(self) -> List[str]:
        """List of filenames, including local files pending upload and
        excluding files pending removal.
        """
        current = [item for item in self._files if item not in self._files_to_remove]
        current.extend(self._files_to_add)
        return list(sorted(set(current)))

    def load(self) -> None:
        """Load data payload and list of files."""
        assert self.id
        self._payload = self.adapter.load_payload(self.id)
        self._payload_cache = copy.deepcopy(self._payload)

        self._files = self.adapter.list_files(self.id)
        self._files_to_add = {}
        self._files_to_remove = []

    def save(self) -> None:
        """Save data payload and attach/remove files."""
        if self.id is None:
            assert self.parent_id, "Parent work item not saved yet"
            self.id = self.adapter.create_output(self.parent_id, payload=self.payload)
        else:
            self.adapter.save_payload(self.id, self.payload)

        for name in self._files_to_remove:
            self.adapter.remove_file(self.id, name)

        for name, path in self._files_to_add.items():
            with open(path, "rb") as infile:
                self.adapter.add_file(
                    self.id, name, original_name=path.name, content=infile.read()
                )

        # Empty unsaved values
        self._payload = self._payload_cache
        self._payload_cache = copy.deepcopy(self._payload)

        self._files = self.files
        self._files_to_add = {}
        self._files_to_remove = []

    def get_file(self, name: str, path: Optional[str] = None) -> str:
        """Load an attached file and store it on the local filesystem.

        :param name: Name of attached file
        :param path: Destination path. Default to current working directory.
        :returns:    Path to created file
        """
        if name not in self.files:
            raise FileNotFoundError(f"No such file: {name}")

        if not path:
            root = os.getenv("ROBOT_ROOT", "")
            path = os.path.join(root, name)

        if name in self._files_to_add:
            local_path = self._files_to_add[name]
            if Path(local_path).resolve() != Path(path).resolve():
                copy2(local_path, path)
        else:
            assert self.id, "Work item not saved yet"
            content = self.adapter.get_file(self.id, name)
            with open(path, "wb") as outfile:
                outfile.write(content)

        # Always return absolute path
        return str(Path(path).resolve())

    def add_file(self, path: Union[Path, str], name: Optional[str] = None) -> str:
        """Add file to current work item. Does not upload
        until ``save()`` is called.

        :param path: Path to file to upload
        :param name: Name of file in work item. If not given,
                     name of file on disk is used.
        """
        path = Path(path).resolve()

        if path in self._files_to_add.values():
            logging.warning("File already added: %s", path)

        if not path.is_file():
            raise FileNotFoundError(f"Not a valid file: {path}")

        name = name or path.name
        self._files_to_add[name] = path

        if name in self._files_to_remove:
            self._files_to_remove.remove(name)

        return name

    def remove_file(self, name: str, missing_ok: bool = True) -> str:
        """Remove file from current work item. Change is not applied
        until ``save()`` is called.

        :param name: Name of attached file
        """
        if not missing_ok and name not in self.files:
            raise FileNotFoundError(f"No such file: {name}")

        if name in self._files:
            self._files_to_remove.append(name)

        if name in self._files_to_add:
            del self._files_to_add[name]

        return name


@library
class WorkItems:
    """A library for interacting with Control Room work items.

    Work items are used for managing data that go through multiple
    steps and tasks inside a process. Each step of a process receives
    input work items from the previous step, and creates output work items for
    the next step.

    **Item structure**

    A work item's data payload is JSON and allows storing anything that is
    serializable. This library by default interacts with payloads that
    are a dictionary of key-value pairs, which it treats as individual
    variables. These variables can be exposed to the Robot Framework task
    to be used directly.

    In addition to the data section, a work item can also contain files,
    which are stored by default in Robocorp Control Room. Adding and using
    files with work items requires no additional setup from the user.

    **Loading inputs**

    The library automatically loads the first input work item, if the
    library input argument ``autoload`` is truthy (default).

    After an input has been loaded its payload and files can be accessed
    through corresponding keywords, and optionally these values can be modified.

    Example:

    After starting the process by sending an e-mail with a body like:

    .. code-block:: json

        {
            "message": "Hello world!"
        }

    **Creating outputs**

    It's possible to create multiple new work items as an output from a
    task. With the keyword ``Create Output Work Item`` a new empty item
    is created as a child for the currently loaded input.

    All created output items are sent into the input queue of the next
    step in the process.

    **Active work item**

    Keywords that read or write from a work item always operate on the currently
    active work item. Usually that is the input item that has been automatically
    loaded when the execution started, but the currently active item is changed
    whenever the keywords ``Create Output Work Item`` or ``Get Input Work Item``
    are called. It's also possible to change the active item manually with the
    keyword ``Set current work item``.

    **Saving changes**

    While a work item is loaded automatically when a suite starts, changes are
    not automatically reflected back to the source. The work item will be modified
    locally and then saved when the keyword ``Save Work Item`` is called.
    This also applies to created output work items.

    It is recommended to defer saves until all changes have been made to prevent
    leaving work items in a half-modified state in case of failures.

    **Local Development**

    While Control Room is the default implementation, it can also be replaced
    with a custom adapter. The selection is based on either the ``default_adapter``
    argument for the library, or the ``RPA_WORKITEMS_ADAPTER`` environment
    variable. The library has a built-in alternative adapter called FileAdapter for
    storing work items to disk.

    The FileAdapter uses a local JSON file for input work items.
    It's a list of work items, each of which has a data payload and files.

    An example of a local file with one work item:

    .. code-block:: json

        [
            {
                "payload": {
                    "variable1": "a-string-value",
                    "variable2": ["a", "list", "value"]
                },
                "files": {
                    "file1": "path/to/file.ext"
                }
            }
        ]

    Output work items (if any) are saved to an adjacent file
    with the same name, but with the extension ``.output.json``. You can specify
    through the "RPA_OUTPUT_WORKITEM_PATH" env var a different path and name for this
    file.

    **Simulating the Cloud with Robocorp Code VSCode Extension**

    If you are developing in VSCode with the `Robocorp Code extension`_, you can
    utilize the built in local development features described in the
    `Developing with work items locally`_ section of the
    `Using work items`_ development guide.

    .. _Robocorp Code extension: https://robocorp.com/docs/setup/development-environment#visual-studio-code-with-robocorp-extensions
    .. _Developing with work items locally: https://robocorp.com/docs/development-guide/control-room/work-items#developing-with-work-items-locally
    .. _Using work items: https://robocorp.com/docs/development-guide/control-room/work-items

    **Examples**

    **Robot Framework**

    In the following example a task creates an output work item,
    and attaches some variables to it.

    .. code-block:: robotframework

        *** Settings ***
        Library    RPA.Robocorp.WorkItems

        *** Tasks ***
        Save variables to Control Room
            Create Output Work Item
            Set work item variables    user=Dude    mail=address@company.com
            Save Work Item

    In the next step of the process inside a different robot, we can use
    previously saved work item variables. Also note how the input work item is
    loaded implicitly when the suite starts.

    .. code-block:: robotframework

        *** Settings ***
        Library    RPA.Robocorp.WorkItems

        *** Tasks ***
        Use variables from Control Room
            Set task variables from work item
            Log    Variables are now available: s${user}, ${mail}

    **Python**

    The library can also be used through Python, but it does not implicitly
    load the first work item.

    .. code-block:: python

        import logging
        from RPA.Robocorp.WorkItems import WorkItems

        def list_variables(item_id):
            library = WorkItems()
            library.get_input_work_item()

            variables = library.get_work_item_variables()
            for variable, value in variables.items():
                logging.info("%s = %s", variable, value)
    """  # noqa: E501

    ROBOT_LIBRARY_SCOPE = "GLOBAL"
    ROBOT_LIBRARY_DOC_FORMAT = "REST"
    ROBOT_LISTENER_API_VERSION = 2

    def __init__(
        self,
        autoload: bool = True,
        root: Optional[str] = None,
        default_adapter: Union[Type[BaseAdapter], str] = FileAdapter,
    ):
        self.ROBOT_LIBRARY_LISTENER = self

        #: Current selected work item
        self._current: Optional[WorkItem] = None
        #: Input work items
        self.inputs: List[WorkItem] = []
        #: Output work items
        self.outputs: List[WorkItem] = []
        #: Variables root object in payload
        self.root = root
        #: Auto-load first input item and automatically parse e-mail content if
        # present.
        self.autoload: bool = autoload
        #: Adapter for reading/writing items
        self._adapter_class = self._load_adapter(default_adapter)  # type: ignore
        self._adapter: Optional[BaseAdapter] = None

        # Know when we're iterating (and consuming) all the work items in the queue.
        self._under_iteration = Event()

    @property
    def adapter(self) -> BaseAdapter:
        if self._adapter is None:
            self._adapter = self._adapter_class()
        return self._adapter

    @property
    def current(self) -> WorkItem:
        if self._current is None:
            raise RuntimeError("No active work item")

        return self._current

    @current.setter
    def current(self, value: Any) -> None:
        if not isinstance(value, WorkItem):
            raise ValueError(f"Not a work item: {value}")

        self._current = value

    @property
    def active_input(self) -> Optional[WorkItem]:
        if self._current and self._current.parent_id is None:  # input set as current
            return self._current
        if self.inputs:  # other current item set, and taking the last input
            return self.inputs[-1]
        return None

    def _load_adapter(self, default: type[BaseAdapter]) -> Type[BaseAdapter]:
        """Load adapter by name, using env or given default."""
        adapter = required_env("RPA_WORKITEMS_ADAPTER", default)

        if isinstance(adapter, str):
            adapter = import_by_name(adapter, __name__)

        assert issubclass(
            adapter, BaseAdapter  # type: ignore
        ), "Adapter does not inherit from BaseAdapter"

        return adapter  # type: ignore

    def _start_suite(self, *_: Any) -> None:
        """Robot Framework listener method, called when suite starts."""
        if not self.autoload:
            return

        try:
            self.get_input_work_item(_internal_call=True)
        # pylint: disable=broad-except
        except Exception as exc:
            logging.warning("Failed to load input work item: %s", exc)
        finally:
            self.autoload = False

    def _release_on_failure(self, attributes: Dict[str, Any]) -> None:
        """Automatically releases current input Work Item when encountering failures
        with tasks and/or suites.
        """
        if attributes["status"] != "FAIL":
            return

        message = attributes["message"]
        logging.info("Releasing FAILED input item with APPLICATION error: %s", message)
        self.release_input_work_item(
            state=State.FAILED,
            exception_type=Error.APPLICATION,
            message=message,
            _internal_release=True,
        )

    def _end_suite(self, _: str, attributes: Dict[str, Any]) -> None:
        """Robot Framework listener method, called when the suite ends."""
        # pylint: disable=unused-argument
        for item in self.inputs + self.outputs:
            if item.is_dirty:
                logging.warning("%s has unsaved changes that will be discarded", item)

        self._release_on_failure(attributes)

    def _end_test(self, _: str, attributes: Dict[str, Any]) -> None:
        """Robot Framework listener method, called when each task ends."""
        self._release_on_failure(attributes)

    @keyword
    def set_current_work_item(self, item: WorkItem) -> None:
        # pylint: disable=anomalous-backslash-in-string
        """Set the currently active work item.

        The current work item is used as the target by other keywords
        in this library.

        Keywords ``Get Input Work Item`` and ``Create Output Work Item``
        set the active work item automatically, and return the created
        instance.

        With this keyword the active work item can be set manually.

        Robot Framework Example:

        .. code-block:: robotframework

            *** Tasks ***
            Creating outputs
                ${input}=    Get Input Work Item
                ${output}=   Create Output Work Item
                Set current work item    ${input}

        Python Example:

        .. code-block:: python

            from RPA.Robocorp.WorkItems import WorkItems

            wi = WorkItems()
            parent_wi = wi.get_input_work_item()
            child_wi = wi.create_output_work_item()
            wi.set_current_work_item(parent_wi)
        """  # noqa: W605
        self.current = item

    @keyword
    def get_input_work_item(self, _internal_call: bool = False) -> WorkItem:
        """Load the next work item from the input queue, and set it as the active work
        item.

        Each time this is called, the previous input work item is released (as DONE)
        prior to reserving the next one.
        If the library import argument ``autoload`` is truthy (default),
        this is called automatically when the Robot Framework suite
        starts.
        """
        if not _internal_call:
            self._raise_under_iteration("Get Input Work Item")

        # Automatically release (with success) the lastly retrieved input work item
        # when asking for the next one. (or the currently set input if such)
        self.release_input_work_item(State.DONE, _internal_release=True)

        item_id = self.adapter.reserve_input()
        item = WorkItem(item_id=item_id, parent_id=None, adapter=self.adapter)
        item.load()
        self.inputs.append(item)
        self.current = item

        return self.current

    @keyword
    def create_output_work_item(
        self,
        variables: Optional[Dict[str, Any]] = None,
        files: Optional[Union[str, List[str]]] = None,
        save: bool = False,
    ) -> WorkItem:
        """Create a new output work item with optional variables and files.

        An output work item is always created as a child for an input item, therefore
        a non-released input is required to be loaded first.
        All changes to the work item are done locally and are sent to the output queue
        after the keyword ``Save Work Item`` is called only, except when `save` is
        `True`.

        :param variables: Optional dictionary with variables to be set into the new
            output work item.
        :param files: Optional list or comma separated paths to files to be included
            into the new output work item.
        :param save: Automatically call ``Save Work Item`` over the newly created
            output work item.
        :returns: The newly created output work item object.

        **Examples**

        **Robot Framework**

        .. code-block:: robotframework

            *** Tasks ***
            Create output items with variables then save
                ${customers} =  Load customer data
                FOR     ${customer}    IN    @{customers}
                    Create Output Work Item
                    Set Work Item Variables    id=${customer.id}
                    ...     name=${customer.name}
                    Save Work Item
                END

            Create and save output items with variables and files in one go
                ${customers} =  Load customer data
                FOR     ${customer}    IN    @{customers}
                    &{customer_vars} =    Create Dictionary    id=${customer.id}
                    ...     name=${customer.name}
                    Create Output Work Item     variables=${customer_vars}
                    ...     files=devdata${/}report.csv   save=${True}
                END

        **Python**

        .. code-block:: python

            from RPA.Robocorp.WorkItems import WorkItems

            wi = WorkItems()
            wi.get_input_work_item()
            customers = wi.get_work_item_variable("customers")
            for customer in customers:
                wi.create_output_work_item(customer, save=True)

        """
        if not self.inputs:
            raise RuntimeError(
                "Unable to create output work item without an input, "
                "call `Get Input Work Item` first"
            )

        parent = self.active_input
        assert parent
        if parent.state is not None:
            raise RuntimeError(
                "Can't create any more output work items since the last input was "
                "released, get a new input work item first"
            )

        item = WorkItem(item_id=None, parent_id=parent.id, adapter=self.adapter)
        self.outputs.append(item)
        self.current = item
        if variables:
            self.set_work_item_variables(**variables)
        if files:
            if isinstance(files, str):
                files = [path.strip() for path in files.split(",")]
            for path in files:
                # Assumes the name would be the same as the file name itself.
                self.add_work_item_file(path)
        if save:
            logging.debug("Auto-saving the just created output work item.")
            self.save_work_item()

        return self.current

    @keyword
    def save_work_item(self) -> None:
        """Save the current data and files in the work item. If not saved,
        all changes are discarded when the library goes out of scope.
        """
        self.current.save()

    @keyword
    def clear_work_item(self) -> None:
        """Remove all data and files in the current work item.

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Clearing a work item
                Clear work item
                Save work item

        .. code-block:: python

            from RPA.Robocorp.WorkItems import WorkItems

            wi = WorkItems()
            wi.get_input_work_item()
            wi.clear_work_item()
            wi.save_work_item()
        """
        assert isinstance(self.current.payload, dict)
        self.current.payload.clear()
        self.remove_work_item_files("*")

    @keyword
    def get_work_item_payload(self) -> JSONType:
        """Get the full JSON payload for a work item.

        **NOTE**: Most use cases should prefer higher-level keywords.

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                ${payload}=    Get work item payload
                Log    Entire payload as dictionary: ${payload}
        """
        return self.current.payload

    @keyword
    def set_work_item_payload(self, payload: JSONType) -> None:
        # pylint: disable=anomalous-backslash-in-string
        """Set the full JSON payload for a work item.

        :param payload: Content of payload, must be JSON-serializable

        **NOTE**: Most use cases should prefer higher-level keywords.
        Using this keyword may cause errors when getting the payload via
        the normal ``Get work item variable`` and
        ``Get work item variables`` keywords if you do not set the payload
        to a ``dict``.

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                ${output}=    Create dictionary    url=example.com    username=Mark
                Set work item payload    ${output}

        """  # noqa: W605
        self.current.payload = payload

    @keyword
    def list_work_item_variables(self) -> List[str]:
        """List the variable names for the current work item.

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                ${variables}=    List work item variables
                Log    Available variables in work item: ${variables}

        """
        return list(self.get_work_item_variables().keys())

    @keyword
    def get_work_item_variable(self, name: str, default: Any = UNDEFINED) -> Any:
        """Return a single variable value from the work item,
        or default value if defined and key does not exist.

        If key does not exist and default is not defined, raises `KeyError`.

        :param name: Name of variable
        :param default: Default value if key does not exist

        Robot Framework Example:

        .. code-block:: robotframework

            *** Tasks ***
            Using a work item
                ${username}=    Get work item variable    username    default=guest

        Python Example:

        .. code-block:: python

            from RPA.Robocorp.WorkItems import WorkItems

            wi = WorkItems()
            wi.get_input_work_item()
            customers = wi.get_work_item_variable("customers")
            print(customers)
        """
        variables = self.get_work_item_variables()
        value = variables.get(name, default)

        if value is UNDEFINED:
            raise KeyError(f"Undefined variable: {name}")

        return value

    @keyword
    def get_work_item_variables(self) -> Dict[Any, Any]:
        """Read all variables from the current work item and
        return their names and values as a dictionary.

        Robot Framework Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                ${variables}=    Get work item variables
                Log    Username: ${variables}[username], Email: ${variables}[email]

        Python Example:

            from RPA.Robocorp.WorkItems import WorkItems
            wi = WorkItems()
            wi.get_input_work_item()
            input_wi = wi.get_work_item_variables()
            print(input_wi["username"])
            print(input_wi["email"])
        """

        payload = self.current.payload
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected work item payload to be `dict`, was `{type(payload)}`"
            )

        if self.root is not None:
            payload = payload.setdefault(self.root, {})
            assert isinstance(payload, dict)

        return payload

    @keyword
    def set_work_item_variable(self, name: str, value: Any) -> None:
        """Set a single variable value in the current work item.

        :param name: Name of variable
        :param value: Value of variable

        Robot Framework Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                Set work item variable    username    MarkyMark
                Save Work Item

        Python Example:

        .. code-block:: python

            from RPA.Robocorp.WorkItems import WorkItems

            customers = [{"id": 1, "name": "Apple"}, {"id": 2, "name": "Microsoft"}]
            wi = WorkItems()
            wi.get_input_work_item()
            wi.set_work_item_variable("customers", customers)
        """
        variables = self.get_work_item_variables()
        logging.info("%s = %s", name, value)
        variables[name] = value

    @keyword
    def set_work_item_variables(self, **kwargs: Any) -> None:
        """Set multiple variables in the current work item.

        :param kwargs: Pairs of variable names and values

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                Set work item variables    username=MarkyMark    email=mark@example.com
                Save Work Item
        """
        variables = self.get_work_item_variables()
        for name, value in kwargs.items():
            logging.info("%s = %s", name, value)
            variables[name] = value

    @keyword
    def delete_work_item_variables(self, *names: str, force: bool = True) -> None:
        """Delete variable(s) from the current work item.

        :param names: Names of variables to remove
        :param force: Ignore variables that don't exist in work item

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                Delete work item variables    username    email
                Save Work Item
        """
        variables = self.get_work_item_variables()
        for name in names:
            if name in variables:
                del variables[name]
                logging.info("Deleted variable: %s", name)
            elif not force:
                raise KeyError(f"No such variable: {name}")

    @keyword
    def set_task_variables_from_work_item(self) -> None:
        """Convert all variables in the current work item to
        Robot Framework task variables, see `variable scopes`_.

        .. _variable scopes: https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html#variable-scopes

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                # Work item has variable INPUT_URL
                Set task variables from work item
                Log    The variable is now available: ${INPUT_URL}
        """  # noqa: E501
        variables = self.get_work_item_variables()
        for name, value in variables.items():
            BuiltIn().set_task_variable(f"${{{name}}}", value)

    @keyword
    def list_work_item_files(self) -> List[str]:
        """List the names of files attached to the current work item.

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                ${names}=    List work item files
                Log    Work item has files with names: ${names}
        """
        return [f"{path}" for path in self.current.files]

    @keyword
    def get_work_item_file(self, name: str, path: Optional[str] = None) -> str:
        """Get attached file from work item to disk.
        Returns the absolute path to the created file.

        :param name: Name of attached file
        :param path: Destination path of file. If not given, current
                     working directory is used.

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                ${path}=    Get work item file    input.xls
                Open workbook    ${path}
        """
        path = self.current.get_file(name, path)
        logging.info("Downloaded file to: %s", path)
        return f"{path}"

    @keyword
    def add_work_item_file(self, path: str, name: Optional[str] = None) -> str:
        """Add given file to work item.

        :param path: Path to file on disk
        :param name: Destination name for file. If not given, current name
                     of local file is used.

        **NOTE**: Files are not uploaded before work item is saved

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                Add work item file    output.xls
                Save Work Item
        """
        logging.info("Adding file: %s", path)
        return self.current.add_file(path, name=name)

    @keyword
    def remove_work_item_file(self, name: str, missing_ok: bool = True) -> str:
        """Remove attached file from work item.

        :param name: Name of attached file
        :param missing_ok: Do not raise exception if file doesn't exist

        **NOTE**: Files are not deleted before work item is saved

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                Remove work item file    input.xls
                Save Work Item
        """
        logging.info("Removing file: %s", name)
        return self.current.remove_file(name, missing_ok)

    @keyword
    def get_work_item_files(
        self, pattern: str, dirname: Optional[Union[str, Path]] = None
    ) -> List[str]:
        """Get files attached to work item that match given pattern.
        Returns a list of absolute paths to the downloaded files.

        :param pattern: Filename wildcard pattern
        :param dirname: Destination directory, if not given robot root is used

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                ${paths}=    Get work item files    customer_*.xlsx
                FOR  ${path}  IN  @{paths}
                    Handle customer file    ${path}
                END
        """
        paths = []
        for name in self.list_work_item_files():
            if fnmatch.fnmatch(name, pattern):
                if dirname:
                    path = self.get_work_item_file(name, os.path.join(dirname, name))
                else:
                    path = self.get_work_item_file(name)
                paths.append(path)

        logging.info("Downloaded %d file(s)", len(paths))
        return paths

    @keyword
    def add_work_item_files(self, pattern: str) -> List[Optional[Union[Path, str]]]:
        """Add all files that match given pattern to work item.

        :param pattern: Path wildcard pattern

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                Add work item files    %{ROBOT_ROOT}/generated/*.csv
                Save Work Item
        """
        matches = find_files(pattern, include_dirs=False)

        paths: List[Optional[Union[Path, str]]] = []
        for match in matches:
            path = self.add_work_item_file(f"{match}")
            paths.append(path)

        logging.info("Added %d file(s)", len(paths))
        return paths

    @keyword
    def remove_work_item_files(
        self, pattern: str, missing_ok: bool = True
    ) -> List[str]:
        """Removes files attached to work item that match the given pattern.

        :param pattern: Filename wildcard pattern
        :param missing_ok: Do not raise exception if file doesn't exist

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                Remove work item files    *.xlsx
                Save Work Item
        """
        names = []

        for name in self.list_work_item_files():
            if fnmatch.fnmatch(name, pattern):
                name = self.remove_work_item_file(name, missing_ok)
                names.append(name)

        logging.info("Removed %d file(s)", len(names))
        return names

    def _raise_under_iteration(self, action: str) -> None:
        if self._under_iteration.is_set():
            raise RuntimeError(f"Can't {action} while iterating input work items")

    def _ensure_input_for_iteration(self) -> bool:
        active_input = self.active_input
        active_state = active_input.state if active_input else None
        if not active_input or active_state:
            # There are no inputs loaded yet or the lastly retrieved input work
            #  item is already processed. Time for trying to load a new one.
            try:
                self.get_input_work_item(_internal_call=True)
            except EmptyQueue:
                return False

        return True

    @keyword
    def for_each_input_work_item(
        self,
        keyword_or_func: Union[
            str, Callable[[int, int, int], bool], Callable[[], Optional[int]]
        ],
        *args: Any,
        items_limit: int = 0,
        return_results: bool = True,
        **kwargs: Any,
    ) -> Optional[List[Any]]:
        """Run a keyword or function for each work item in the input queue.

        Automatically collects and returns a list of results, switch
        ``return_results`` to ``False`` for avoiding this.

        :param keyword_or_func: The RF keyword or Py function you want to map through
            all the work items
        :param args: Variable list of arguments that go into the called keyword/function
        :param kwargs: Variable list of keyword arguments that go into the called
            keyword/function
        :param items_limit: Limit the queue item retrieval to a certain amount,
            otherwise all the items are retrieved from the queue until depletion
        :param return_results: Collect and return a list of results given each
            keyword/function call if truthy

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Log Payloads
                @{lengths} =     For Each Input Work Item    Log Payload
                Log   Payload lengths: @{lengths}

            *** Keywords ***
            Log Payload
                ${payload} =     Get Work Item Payload
                Log To Console    ${payload}
                ${len} =     Get Length    ${payload}
                [Return]    ${len}

        OR

        .. code-block:: python

            import logging
            from RPA.Robocorp.WorkItems import WorkItems

            library = WorkItems()

            def log_payload():
                payload = library.get_work_item_payload()
                print(payload)
                return len(payload)

            def log_payloads():
                library.get_input_work_item()
                lengths = library.for_each_input_work_item(log_payload)
                logging.info("Payload lengths: %s", lengths)

            log_payloads()
        """

        self._raise_under_iteration("iterate input work items")

        if isinstance(keyword_or_func, str):
            to_call = lambda: BuiltIn().run_keyword(  # noqa: E731,E501, pylint: disable=unnecessary-lambda-assignment
                keyword_or_func, *args, **kwargs
            )
        else:
            to_call = lambda: keyword_or_func(  # noqa: E731,E501, pylint: disable=unnecessary-lambda-assignment
                *args, **kwargs
            )
        results = []

        try:
            self._under_iteration.set()
            count = 0
            while True:
                input_ensured = self._ensure_input_for_iteration()
                if not input_ensured:
                    break

                result = to_call()  # type: ignore
                if return_results:
                    results.append(result)
                self.release_input_work_item(State.DONE, _internal_release=True)
                count += 1
                if items_limit and count >= items_limit:
                    break
        finally:
            self._under_iteration.clear()

        return results if return_results else None

    @keyword
    def release_input_work_item(
        self,
        state: Union[State, str],
        exception_type: Optional[Union[Error, str]] = None,
        code: Optional[str] = None,
        message: Optional[str] = None,
        _internal_release: bool = False,
    ) -> None:
        """Release the lastly retrieved input work item and set its state.

        This can be released with DONE or FAILED states. With the FAILED state, an
        additional exception can be sent to Control Room describing the problem that
        you encountered by specifying a type and optionally a code and/or message.
        After this has been called, no more output work items can be created
        unless a new input work item has been loaded again.

        :param state: The status on the last processed input work item
        :param exception_type: Error type (BUSINESS, APPLICATION). If this is not
            specified, then the cloud will assume UNSPECIFIED
        :param code: Optional error code identifying the exception for future
            filtering, grouping and custom retrying behaviour in the cloud
        :param message: Optional human-friendly error message supplying additional
            details regarding the sent exception

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                Login into portal
                    ${user} =     Get Work Item Variable    user
                    ${doc} =    Get Work Item Variable    doc
                    TRY
                        Login Keyword    ${user}
                        Upload Doc Keyword    ${doc}

                    EXCEPT    Login Failed
                        Release Input Work Item     FAILED
                        ...    exception_type=APPLICATION
                        ...    code=LOGIN_PORTAL_DOWN
                        ...    message=Unable to login, retry again later.

                    EXCEPT    Format Error    AS    ${err}
                        ${message} =    Catenate
                        ...    Document format is not correct and cannot be uploaded.
                        ...    Correct the format in this work item and try again.
                        ...    Full error message received: ${err}
                        Release Input Work Item     FAILED
                        ...    exception_type=BUSINESS
                        ...    code=DOC_FORMAT_ERROR
                        ...    message=${message}

                    END

        OR

        .. code-block:: python

            from RPA.Robocorp.WorkItems import State, WorkItems

            library = WorkItems()

            def process_and_set_state():
                library.get_input_work_item()
                library.release_input_work_item(State.DONE)
                print(library.current.state)  # would print "State.DONE"

            process_and_set_state()
        """
        # Note that `_internal_release` here is True when automatically releasing items
        #  within our internal library logic.

        active_input = self.active_input
        if not active_input:
            if _internal_release:
                # Have nothing to release and that's normal (reserving for the first
                # time).
                return
            raise RuntimeError(
                "Can't release without reserving first an input work item"
            )
        if active_input.state is not None:
            if _internal_release:
                # Item already released and that's normal when reaching an empty queue
                # and we ask for another item again. We don't want to set states twice.
                return
            raise RuntimeError("Input work item already released")
        assert active_input.parent_id is None, "set state on output item"
        assert active_input.id is not None, "set state on input item with null ID"

        # RF automatically converts string "DONE" to State.DONE object if only `State`
        #  type annotation is used in the keyword definition.
        if not isinstance(state, State):
            # But since we support strings as well now, to stay compatible with Python
            #  behaviour, a "COMPLETE" value is expected instead of "DONE".
            state = state.upper()
            state = State.DONE.value if state == "DONE" else state
            state = State(state)
        exception = None
        if state is State.FAILED:
            if exception_type:
                exception_type = (
                    exception_type
                    if isinstance(exception_type, Error)
                    else Error(exception_type.upper())
                )
                exception = {
                    "type": exception_type.value,
                    "code": code,
                    "message": message,
                }
            elif code or message:
                exc_types = ", ".join(list(Error.__members__))
                raise RuntimeError(f"Must specify failure type from: {exc_types}")

        self.adapter.release_input(active_input.id, state, exception=exception)
        active_input.state = state

    @keyword
    def get_current_work_item(self) -> WorkItem:
        """Get the currently active work item.

        The current work item is used as the target by other keywords
        in this library.

        Keywords ``Get Input Work Item`` and ``Create Output Work Item``
        set the active work item automatically, and return the created
        instance.

        With this keyword the active work item can be retrieved manually.

        Example:

        .. code-block:: robotframework

            *** Tasks ***
            Example task
                ${input} =    Get Current Work Item
                ${output} =   Create Output Work Item
                Set Current Work Item    ${input}
        """
        return self.current
