# One command to run everything locally:  make up   (stop with: make down)
# Docker is installed on this Mac, but containers cannot use the Apple GPU ("mps"), which
# would make embedding ~10x slower. So the four services run as plain uvicorn processes.
# Activate the conda env first:  conda activate fda-label-intelligence   (or: make up PYTHON=/path/to/python)
PYTHON ?= python
RUN := .run
SERVICES := ingest:8001 chunk:8002 embed:8003 query:8004

.PHONY: up down restart status logs test index

up:
	@mkdir -p $(RUN)
	@for s in $(SERVICES); do name=$${s%%:*}; port=$${s##*:}; \
	  if [ -s $(RUN)/$$name.pid ] && kill -0 $$(cat $(RUN)/$$name.pid) 2>/dev/null; then echo "$$name already running"; continue; fi; \
	  nohup $(PYTHON) -m uvicorn src.services.$$name:app --host 127.0.0.1 --port $$port --log-level warning \
	    > $(RUN)/$$name.log 2>&1 & echo $$! > $(RUN)/$$name.pid; \
	  echo "started $$name on :$$port (pid $$(cat $(RUN)/$$name.pid))"; done
	@$(PYTHON) scripts/wait_healthy.py

# Stops each service and empties its pid file (an empty pid file means "not running").
down:
	@for s in $(SERVICES); do name=$${s%%:*}; \
	  if [ -s $(RUN)/$$name.pid ]; then kill $$(cat $(RUN)/$$name.pid) 2>/dev/null && echo "stopped $$name"; : > $(RUN)/$$name.pid; fi; done

restart: down up

status:
	@$(PYTHON) scripts/wait_healthy.py --once

logs:
	@tail -n 20 $(RUN)/*.log

test:
	$(PYTHON) -m pytest -q

index:   # build the chunk store + embeddings + Pinecone index through the running services
	$(PYTHON) scripts/build_index.py
