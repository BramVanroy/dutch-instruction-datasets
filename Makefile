# Format source code automatically
style:
	black src/dutch_data
	isort src/dutch_data

# Control quality
quality:
	black --check --diff src/dutch_data
	isort --check-only src/dutch_data
	flake8 src/dutch_data
	mypy src/dutch_data --check-untyped-defs

# Run tests
test:
	pytest
