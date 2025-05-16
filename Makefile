# This file is for development usage

SHELL=/bin/bash

.PHONY:reinstall
reinstall:
	rm -f ./*.so
	pip uninstall torch_memory_saver -y
	# pip install --no-cache-dir -e .
	pip install --no-cache-dir .

.PHONY:clean
clean:
	rm -rf dist/*

.PHONY:build-wheel
build-wheel:
	bash scripts/build.sh

.PHONY:build-sdist
build-sdist:
	# python3 -m build --no-isolation
	python3 setup.py sdist --dist-dir=dist

.PHONY:upload
upload:
	python3 -m twine upload dist/*
