from pathlib import Path
from pytest import MonkeyPatch
from RPA.Robocorp.Vault import BaseSecretManager
from RPA.Robocorp.Vault import FileSecrets
from RPA.Robocorp.Vault import Secret
from RPA.Robocorp.Vault import Vault
from typing import Any
from typing import Optional
import json
import pytest


RESOURCES: Path = Path(__file__).parent / ".." / "resources"


class InvalidBaseClass(object):
    def get_secret(self, secret_name: str) -> bool:
        assert False, "Should not be called"


class MockAdapter(BaseSecretManager):
    args = None
    name = None
    value: Optional[Any] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        MockAdapter.args = (args, kwargs)

    def get_secret(self, secret_name: str) -> Any:
        MockAdapter.name = secret_name
        return MockAdapter.value

    def set_secret(self, secret: Secret) -> None:
        MockAdapter.name = secret.name
        MockAdapter.value = dict(secret)


@pytest.fixture
def mock_env_default(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("RPA_SECRET_MANAGER", raising=False)


@pytest.fixture
def mock_env_vault(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("RC_API_SECRET_HOST", "mock-url")
    monkeypatch.setenv("RC_API_SECRET_TOKEN", "mock-token")
    monkeypatch.setenv("RC_WORKSPACE_ID", "mock-workspace")


@pytest.fixture(params=["secrets.json"])
def secrets_file(request: Any) -> Any:
    return request.param


def test_secrets_vault_as_default(
    mock_env_default: MonkeyPatch, mock_env_vault: MonkeyPatch
) -> None:
    library = Vault()
    assert isinstance(library.adapter, FileSecrets)


def test_secrets_custom_adapter_arguments(mock_env_default: MonkeyPatch) -> None:
    library = Vault("pos-value", key="key-value", default_adapter=MockAdapter)
    library.get_secret("not-relevant")  # Adapter created on first request
    assert MockAdapter.args == (("pos-value",), {"key": "key-value"})


def test_secrets_custom_adapter_get_secret(mock_env_default: MonkeyPatch) -> None:
    MockAdapter.value = "mock-secret"
    library = Vault(default_adapter=MockAdapter)
    assert library.get_secret("mock-name") == "mock-secret"  # type: ignore
    assert MockAdapter.name == "mock-name"


def test_secrets_adapter_missing_import(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("RPA_SECRET_MANAGER", "RPA.AdapterNotExist")
    with pytest.raises(ValueError):
        Vault()


def test_secrets_adapter_invalid_baseclass(mock_env_default: MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        Vault(default_adapter=InvalidBaseClass)


def test_secret_properties() -> None:
    secret = Secret(
        name="name-value",
        description="description-value",
        values={},
    )

    assert secret.name == "name-value"
    assert secret.description == "description-value"


def test_secret_get() -> None:
    secret = Secret(
        name="name-value",
        description="description-value",
        values={"key_one": "value_one", "key_two": "value_two"},
    )

    assert secret["key_one"] == "value_one"
    assert secret["key_two"] == "value_two"
    with pytest.raises(KeyError):
        _ = secret["key_invalid"]


def test_secret_set() -> None:
    secret = Secret(
        name="name-value",
        description="description-value",
        values={"key_one": "value_one", "key_two": "value_two"},
    )

    secret["key_one"] = "one"
    secret["key_two"] = "two"

    assert secret["key_one"] == "one"
    assert secret["key_two"] == "two"
    with pytest.raises(KeyError):
        _ = secret["key_invalid"]


def test_secret_update() -> None:
    secret = Secret(
        name="name-value",
        description="description-value",
        values={"key_one": "value_one", "key_two": "value_two"},
    )

    secret.update({"key_three": "value_three"})
    expected = {
        "key_one": "value_one",
        "key_two": "value_two",
        "key_three": "value_three",
    }

    assert secret == expected


def test_secret_iterate() -> None:
    secret = Secret(
        name="name-value",
        description="description-value",
        values={"key_one": "value_one", "key_two": "value_two"},
    )

    assert list(secret) == ["key_one", "key_two"]


def test_secret_contains() -> None:
    secret = Secret(
        name="name-value",
        description="description-value",
        values={"key_one": "value_one", "key_two": "value_two"},
    )

    assert "key_two" in secret


def test_secret_print() -> None:
    secret = Secret(
        name="name-value",
        description="description-value",
        values={"key_one": "value_one", "key_two": "value_two"},
    )

    repr_string = repr(secret)
    assert "value_one" not in repr_string
    assert "value_two" not in repr_string

    str_string = str(secret)
    assert "value_one" not in str_string
    assert "value_two" not in str_string


def test_adapter_filesecrets_from_arg(
    monkeypatch: MonkeyPatch, secrets_file: Path
) -> None:
    monkeypatch.delenv("RPA_SECRET_FILE", raising=False)

    adapter = FileSecrets(RESOURCES / secrets_file)
    secret = adapter.get_secret("windows")
    assert isinstance(secret, Secret)
    assert "password" in secret
    assert secret["password"] == "secret"


def test_adapter_filesecrets_from_env(
    monkeypatch: MonkeyPatch, secrets_file: Path
) -> None:
    monkeypatch.setenv("RPA_SECRET_FILE", str(RESOURCES / secrets_file))

    adapter = FileSecrets()
    secret = adapter.get_secret("windows")
    assert isinstance(secret, Secret)
    assert "password" in secret
    assert secret["password"] == "secret"


def test_adapter_filesecrets_invalid_file(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("RPA_SECRET_FILE", str(RESOURCES / "not-a-file.json"))

    # Should not raise
    adapter = FileSecrets()
    assert adapter.data == {}


def test_adapter_filesecrets_invalid_file_extension(monkeypatch: MonkeyPatch) -> None:
    # Note the invalid ".foo" extension not recognized by the serializers.
    monkeypatch.setenv("RPA_SECRET_FILE", str(RESOURCES / "secrets.foo"))

    # Should raise immediately, because this is invalid straight from the
    # configuration. Only certain extensions (formats) are accepted.
    with pytest.raises(ValueError):
        FileSecrets()


def test_adapter_filesecrets_unknown_secret(
    monkeypatch: MonkeyPatch, secrets_file: Path
) -> None:
    monkeypatch.setenv("RPA_SECRET_FILE", str(RESOURCES / secrets_file))

    adapter = FileSecrets()
    with pytest.raises(KeyError):
        _ = adapter.get_secret("not-exist")


def test_adapter_filesecrets_saving(
    monkeypatch: MonkeyPatch, tmp_path: Path, secrets_file: Path
) -> None:
    tmp_file = tmp_path / secrets_file
    tmp_file.write_text((RESOURCES / secrets_file).read_text())
    monkeypatch.setenv("RPA_SECRET_FILE", str(tmp_file))

    adapter = FileSecrets()
    secret = adapter.get_secret("credentials")
    secret["sap"]["password"] = "my-different-secret"
    adapter.set_secret(secret)
    adapter.save()

    assert tmp_file.suffix in (".json")
    secret_dict = json.load(tmp_file.open())
    assert secret_dict["credentials"]["sap"]["password"] == "my-different-secret"
