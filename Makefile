help:
	@grep -Eh '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' | uniq

INDEX_URL ?= https://pypi.python.org/simple
INDEX_HOSTNAME ?= pypi.python.org

export PYTHONPATH=$(PWD)/src

MODULE := RPA

env:  ## Build and link the Python virtual environment
	ln -s $(shell devenv build outputs.python.virtualenv) env

check:  ## Run static analysis checks
	black --check src tests
	isort -c src tests
	flake8 src
	MYPYPATH=$(PWD)/stubs mypy --show-error-codes --strict src tests

clean:  ## Remove build artifacts and temporary files
	devenv gc
	$(RM) -r env htmlcov .devenv

devenv-test: ## Run all test and checks with background services
	devenv test

format:  ## Format the codebase
	treefmt

shell:  ## Start an interactive development shell
	@devenv shell

show:  ## Show build environment information
	@devenv info

test: check test-pytest  ## Run all tests and checks

test-coverage: htmlcov  ## Generate HTML coverage reports

test-pytest:  ## Run unit tests with pytest
	pytest --cov=$(MODULE) tests

watch-mypy:  ## Continuously run mypy for type checks
	find src tests -name "*.py"|MYPYPATH=$(PWD)/stubs entr mypy --show-error-codes --strict src tests

watch-pytest:  ## Continuously run pytest
	find src tests -name "*.py"|entr pytest tests

watch-tests:  ## Continuously run all tests
	  $(MAKE) -j watch-mypy watch-pytest

###

.coverage: test

htmlcov: .coverage
	coverage html

devenv-%:  ## Run command in devenv shell
	devenv shell -- $(MAKE) $*

nix-%:  ## Run command in devenv shell
	devenv shell -- $(MAKE) $*

FORCE:
