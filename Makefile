.PHONY: init clean update lint format


init:
	python3.11 -m venv venv
	(source venv/bin/activate; pip install -r requirements.txt)

clean:
	rm -rf venv/

update: clean init

lint:
	ruff isosurface_vase/

format:
	black isosurface_vase/
	isort isosurface_vase/
