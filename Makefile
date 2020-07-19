include .env

coverage:
		coverage run -m pytest

codecov: coverage
		codecov --token=${CODECOV_TOKEN}