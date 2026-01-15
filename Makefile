.PHONY: figures figure

FIGURE_SCRIPTS := $(wildcard scripts/figure_*.py)

figures:
	@for script in $(FIGURE_SCRIPTS); do \
		echo "RUN $$script"; \
		uv run python "$$script"; \
	done

figure:
	@test -n "$(SCRIPT)" || (echo "Usage: make figure SCRIPT=scripts/figure_02_heatmap.py" && exit 1)
	uv run python "$(SCRIPT)"
