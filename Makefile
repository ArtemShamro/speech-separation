.PHONY: all
all: uv
	@echo "✅ Установка зависимостей с помощью uv..."
	uv pip install --no-progress -r pyproject.toml
	@echo "✅ Установка завершена!"
.PHONY: uv
uv:
	@echo "📦 Установка uv..."
	pip install uv
	