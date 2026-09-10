TEMP_FILE := .mypy_cache data/processed/ data/output

MYPY_FLAG := --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

UV := uv
PDB := pdb

SRC_DIR := src

run: install
		$(UV) run python3 -m $(SRC_DIR)

install:
		$(UV) sync

debug:
		$(UV) run python3 -m pdb $(SRC_DIR)/__main__.py

clean:
		rm -rf $(TEMP_FILE)
		find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
		@$(UV) run flake8 */*.py
		@$(UV) run mypy */*.py $(MYPY_FLAG)