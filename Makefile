.PHONY: all build debug test run clean docker-build docker-run submodules help

# Default trace for testing
TRACE ?= libCacheSim/data/cloudPhysicsIO.vscsi
TRACE_TYPE ?= vscsi
CACHE_SIZE ?= 1GB
NUM_REQUESTS ?= 100000
ALGO ?= s4fifo

all: build

build:
	cd libCacheSim && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=OFF && make -j$$(nproc)

debug:
	cd libCacheSim && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-Wall -Wextra -Wpedantic" -DENABLE_TESTS=OFF && make -j$$(nproc)

test: build
	@if [ ! -f $(TRACE) ]; then \
		echo "ERROR: Trace file $(TRACE) not found."; \
		echo "Please download or symlink the trace into the data/ directory."; \
		exit 1; \
	fi
	@echo "Running smoke test with $(ALGO) on $(TRACE) ($(NUM_REQUESTS) requests)..."
	libCacheSim/build/bin/cachesim $(TRACE) $(TRACE_TYPE) $(ALGO) $(CACHE_SIZE) -n $(NUM_REQUESTS)
	@echo "SUCCESS: Smoke test passed."

run: build
	@if [ ! -f $(TRACE) ]; then \
		echo "ERROR: Trace file $(TRACE) not found."; \
		echo "Usage: make run TRACE=path/to/trace TRACE_TYPE=vscsi"; \
		exit 1; \
	fi
	libCacheSim/build/bin/cachesim $(TRACE) $(TRACE_TYPE) $(ALGO) $(CACHE_SIZE)

clean:
	rm -rf libCacheSim/build

docker-build:
	docker build -t s4fifo-artifact .

docker-run:
	docker run --rm -it s4fifo-artifact

submodules:
	git submodule update --init --recursive

help:
	@echo "S4-FIFO Artifact Evaluation"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build libCacheSim (default)"
	@echo "  build         - Full release build of libCacheSim"
	@echo "  debug         - Debug build with warnings"
	@echo "  test          - Run a quick smoke test (requires trace file)"
	@echo "  run           - Run cachesim with configurable parameters"
	@echo "  clean         - Remove build artifacts"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  submodules    - Initialize and update git submodules"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  TRACE=path/to/trace   - Trace file path (default: data/cloudPhysicsIO.vscsi)"
	@echo "  TRACE_TYPE=type       - Trace type: vscsi, csv, etc. (default: vscsi)"
	@echo "  ALGO=algorithm        - Eviction algorithm (default: s4fifo)"
	@echo "  CACHE_SIZE=size       - Cache size, e.g. 1GB (default: 1GB)"
	@echo "  NUM_REQUESTS=n        - Number of requests for test (default: 100000)"
