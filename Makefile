# Format source code automatically
style:
	black src/dutch_data tests
	isort src/dutch_data tests

# Control quality
quality:
	black --check --diff src/dutch_data tests
	isort --check-only src/dutch_data tests
	flake8 src/dutch_data tests

# Run tests
test:
	pytest
