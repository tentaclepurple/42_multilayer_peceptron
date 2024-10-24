.PHONY: all run open

all: run open

app:
	@echo "Starting Streamlit server..."
	@(streamlit run app.py 2>&1 | \
	while read line; do \
		echo "$$line"; \
		if echo "$$line" | grep -q "Local URL:"; then \
			PORT=$$(echo "$$line" | grep -o "localhost:[0-9]*" | cut -d':' -f2); \
			LOCAL_URL="http://localhost:$$PORT"; \
			if [ "$$(uname -s)" = "Linux" ] && [ "$$(grep -i microsoft /proc/version)" ]; then \
				cmd.exe /c start "$$LOCAL_URL"; \
			elif [ "$$(uname -s)" = "Linux" ]; then \
				xdg-open "$$LOCAL_URL"; \
			fi; \
		fi \
	done) & \
	echo $$! > .streamlit.pid

101:
	@echo "Starting Streamlit server..."
	@(streamlit run mlp101.py 2>&1 | \
	while read line; do \
		echo "$$line"; \
		if echo "$$line" | grep -q "Local URL:"; then \
			PORT=$$(echo "$$line" | grep -o "localhost:[0-9]*" | cut -d':' -f2); \
			LOCAL_URL="http://localhost:$$PORT"; \
			if [ "$$(uname -s)" = "Linux" ] && [ "$$(grep -i microsoft /proc/version)" ]; then \
				cmd.exe /c start "$$LOCAL_URL"; \
			elif [ "$$(uname -s)" = "Linux" ]; then \
				xdg-open "$$LOCAL_URL"; \
			fi; \
		fi \
	done) & \
	echo $$! > .streamlit.pid


stop:
	@if [ -f .streamlit.pid ]; then \
		echo "Stopping Streamlit server..."; \
		kill $$(cat .streamlit.pid) 2>/dev/null || true; \
		rm .streamlit.pid; \
	fi

clean: stop
	@echo "Cleaning up..."
	@rm -f .streamlit.pid