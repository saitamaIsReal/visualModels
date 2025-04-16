.PHONY: help train test export clean env

help:
	@echo "Verfügbare Befehle:"
	@echo "  make train    # startet vit_model.py"
	@echo "  make test     # startet test_vit.py"
	@echo "  make export   # aktualisiert env.yaml"
	@echo "  make clean    # löscht das Modell"
	@echo "  make env      # erstellt Umgebung aus env.yaml"


env:
	conda env create -f env.yaml

train:
	python vit_model.py

test:
	python test_vit.py

export:
	conda env export --no-builds > env.yaml



clean:
	rm -f vision_transformer_model.pth
