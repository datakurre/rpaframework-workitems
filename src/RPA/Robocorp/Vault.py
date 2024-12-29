from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from robot.libraries.BuiltIn import BuiltIn  # type: ignore
from robot.libraries.BuiltIn import RobotNotRunningError
from RPA.helpers import import_by_name
from RPA.helpers import required_env
from RPA.Robocorp.utils import resolve_path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union
import _collections_abc
import collections
import json
import logging


class Secret(_collections_abc.Mapping):  # type: ignore
    """Container for a secret with name, description, and
    multiple key-value pairs. Immutable and avoids logging
    internal values when possible.

    :param name:        Name of secret
    :param description: Human-friendly description for secret
    :param values:      Dictionary of key-value pairs stored in secret
    """

    def __init__(self, name: str, description: str, values: Dict[str, str]):
        self._name = name
        self._desc = description
        self._dict = collections.OrderedDict(**values)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> Optional[str]:
        return self._desc

    def update(self, kvpairs: Dict[str, str]) -> None:
        self._dict.update(kvpairs)

    def __getitem__(self, key: str) -> Any:
        return self._dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._dict[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def __iter__(self) -> Any:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return "Secret(name={name}, keys=[{keys}])".format(
            name=self.name, keys=", ".join(str(key) for key in self.keys())
        )


class BaseSecretManager(metaclass=ABCMeta):
    """Abstract class for secrets management. Should be used as a
    base-class for any adapter implementation.
    """

    @abstractmethod
    def get_secret(self, secret_name: str) -> Secret:
        """Return ``Secret`` object with given name."""

    @abstractmethod
    def set_secret(self, secret: Secret) -> None:
        """Set a secret with a new value."""


class FileSecrets(BaseSecretManager):
    """Adapter for secrets stored in a database file. Supports only
    plaintext secrets, and should be used mainly for debugging.

    The path to the secrets file can be set with the
    environment variable ``RPA_SECRET_FILE``, or as
    an argument to the library.

    The format of the secrets file should be one of the following:

    .. code-block:: JSON

      {
        "name1": {
          "key1": "value1",
          "key2": "value2"
        },
        "name2": {
          "key1": "value1"
        }
      }
    """

    SERIALIZERS = {
        ".json": (json.load, json.dump),
    }

    def __init__(self, secret_file: Union[Path, str] = "secrets.json"):
        self.logger = logging.getLogger(__name__)

        path = required_env("RPA_SECRET_FILE", secret_file)
        self.logger.info("Resolving path: %s", path)
        self.path = resolve_path(path)

        extension = self.path.suffix
        serializer = self.SERIALIZERS.get(extension)
        # NOTE(cmin764): This will raise instead of returning an empty secrets object
        #  because it is wrong starting from the "env.json" configuration level.
        if not serializer:
            raise ValueError(
                f"Not supported local vault secrets file extension {extension!r}"
            )
        self._loader, self._dumper = serializer

        self.data = self.load()

    def load(self) -> Dict[str, Dict[str, str]]:
        """Load secrets file."""
        try:
            with open(self.path, encoding="utf-8") as fd:
                data = self._loader(fd)

            if not isinstance(data, dict):
                raise ValueError("Invalid content format")

            return data
        except (IOError, ValueError) as err:
            self.logger.error("Failed to load secrets file: %s", err)
            return {}

    def save(self) -> None:
        """Save the secrets content to disk."""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                if not isinstance(self.data, dict):
                    raise ValueError("Invalid content format")
                self._dumper(self.data, f, indent=4)
        except (IOError, ValueError) as err:
            self.logger.error("Failed to save secrets file: %s", err)

    def get_secret(self, secret_name: str) -> Secret:
        """Get secret defined with given name from file.

        :param secret_name: Name of secret to fetch
        :returns:           Secret object
        :raises KeyError:   No secret with given name
        """
        values = self.data.get(secret_name)
        if values is None:
            raise KeyError(f"Undefined secret: {secret_name}")

        return Secret(secret_name, "", values)

    def set_secret(self, secret: Secret) -> None:
        """Set the secret value in the local Vault
        with the given ``Secret`` object.

        :param secret:                 A ``Secret`` object.
        :raises IOError, ValueError:   Writing the local vault failed.
        """
        self.data[secret.name] = dict(secret)
        self.save()


class Vault:
    """`Vault` is a library for interacting with secrets, which can be
    taken into use by setting some environment variables.

    File-based secrets can be set by defining two environment variables.

    - ``RPA_SECRET_MANAGER``: RPA.Robocorp.Vault.FileSecrets
    - ``RPA_SECRET_FILE``: Absolute path to the secrets database file

    Example content of local secrets file:

    .. code-block:: json

        {
            "swaglabs": {
                "username": "standard_user",
                "password": "secret_sauce"
            }
        }

    **Examples of Using Secrets in a Robot**

    **Robot Framework**

    .. code-block:: robotframework

        *** Settings ***
        Library    Collections
        Library    RPA.Robocorp.Vault

        *** Tasks ***
        Reading secrets
            ${secret}=    Get Secret  swaglabs
            Log Many      ${secret}

        Modifying secrets
            ${secret}=          Get Secret      swaglabs
            ${level}=           Set Log Level   NONE
            Set To Dictionary   ${secret}       username    nobody
            Set Log Level       ${level}
            Set Secret          ${secret}


    **Python**

    .. code-block:: python

        from RPA.Robocorp.Vault import Vault

        VAULT = Vault()

        def reading_secrets():
            print(f"My secrets: {VAULT.get_secret('swaglabs')}")

        def modifying_secrets():
            secret = VAULT.get_secret("swaglabs")
            secret["username"] = "nobody"
            VAULT.set_secret(secret)

    """  # noqa: E501

    ROBOT_LIBRARY_SCOPE = "GLOBAL"
    ROBOT_LIBRARY_DOC_FORMAT = "REST"

    def __init__(self, *args: Any, **kwargs: Any):
        """The selected adapter can be set with the environment variable
        ``RPA_SECRET_MANAGER``, or the keyword argument ``default_adapter``.
        Defaults to Robocorp Vault if not defined.

        All other library arguments are passed to the adapter.

        :param default_adapter: Override default secret adapter
        """
        self.logger = logging.getLogger(__name__)

        default = kwargs.pop("default_adapter", FileSecrets)
        adapter = required_env("RPA_SECRET_MANAGER", default)

        self._adapter_factory = self._create_factory(adapter, args, kwargs)
        self._adapter = None

        try:
            BuiltIn().import_library("RPA.RobotLogListener")
        except RobotNotRunningError:
            pass

    @property
    def adapter(self) -> BaseSecretManager:
        if self._adapter is None:
            self._adapter = self._adapter_factory()  # type: ignore

        return self._adapter  # type: ignore

    def _create_factory(
        self, adapter: Any, args: Any, kwargs: Any
    ) -> Callable[[], BaseSecretManager]:
        if isinstance(adapter, str):
            adapter = import_by_name(adapter, __name__)

        if not issubclass(adapter, BaseSecretManager):
            raise ValueError(
                f"Adapter '{adapter}' does not inherit from BaseSecretManager"
            )

        def factory() -> BaseSecretManager:
            return adapter(*args, **kwargs)  # type: ignore

        return factory

    def get_secret(self, secret_name: str) -> Secret:
        """Read a secret from the configured source, e.g. Robocorp Vault,
        and return it as a ``Secret`` object.

        :param secret_name: Name of secret
        """
        return self.adapter.get_secret(secret_name)

    def set_secret(self, secret: Secret) -> None:
        """Overwrite an existing secret with new values.

        Note: Only allows modifying existing secrets, and replaces
              all values contained within it.

        :param secret: Secret as a ``Secret`` object, from e.g. ``Get Secret``
        """
        self.adapter.set_secret(secret)
