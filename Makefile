include .env

coverage:
		coverage run -m pytest

codecov:
		codecov --token=${CODECOV_TOKEN}