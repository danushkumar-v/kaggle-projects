.PHONY: run-local push push-fast pull status open

NB := notebook.ipynb

run-local:
	jupyter notebook $(NB)

push:
	kaggle kernels push -p .

push-fast:
	kaggle kernels push -p . --no-execute || \
	  (echo "Note: --no-execute may not be supported by your kaggle CLI version. Falling back."; \
	   kaggle kernels push -p .)

pull:
	kaggle kernels pull -p . -m

status:
	@kid=$$(jq -r .id kernel-metadata.json); \
	kaggle kernels status $$kid

open:
	@kid=$$(jq -r .id kernel-metadata.json); \
	echo "https://www.kaggle.com/code/$$kid"
