.PHONY: check-kaggle list-projects new-project

check-kaggle:
	@bash scripts/check_kaggle_auth.sh

list-projects:
	@ls -1 projects/ | grep -v '^\.' || echo "No projects yet."

new-project:
	@read -p "Project slug (e.g. 02-titanic-eda): " slug; \
	bash scripts/new_project.sh $$slug
