.PHONY: all run open

all: run open

run:
	@echo "Starting Streamlit server..."
	@streamlit run app.py & \
	echo $$! > .streamlit.pid
	@echo "Waiting for server to start..."
	@sleep 2

open:
	@echo "Opening Streamlit in browser..."
	@UNAME_S=$$(uname -s); \
	if [ "$$UNAME_S" = "Linux" ] && [ "$$(grep -i microsoft /proc/version)" ]; then \
		echo "Detected WSL"; \
		cmd.exe /c start http://localhost:8501; \
	elif [ "$$UNAME_S" = "Linux" ]; then \
		echo "Detected Linux"; \
		xdg-open http://localhost:8501 || echo "Failed to open browser"; \
	else \
		echo "Unknown OS. Please open http://localhost:8501 manually"; \
	fi
	@echo "Streamlit is running at http://localhost:8501"

stop:
	@if [ -f .streamlit.pid ]; then \
		echo "Stopping Streamlit server..."; \
		kill $$(cat .streamlit.pid) 2>/dev/null || true; \
		rm .streamlit.pid; \
	fi

clean: stop
	@echo "Cleaning up..."
	@rm -f .streamlit.pid