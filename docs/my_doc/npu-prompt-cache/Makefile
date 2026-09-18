CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2 -Isrc -pthread

TESTS = build/test_kv_prefix_cache build/test_lcp_ubatch_split build/test_backend_stream_tracker

all: $(TESTS)

build:
	mkdir -p build

build/test_kv_prefix_cache: tests/test_kv_prefix_cache.cpp src/kv_prefix_cache.cpp src/kv_prefix_cache.hpp | build
	$(CXX) $(CXXFLAGS) tests/test_kv_prefix_cache.cpp src/kv_prefix_cache.cpp -o $@

build/test_lcp_ubatch_split: tests/test_lcp_ubatch_split.cpp src/kv_prefix_cache.cpp src/kv_prefix_cache.hpp | build
	$(CXX) $(CXXFLAGS) tests/test_lcp_ubatch_split.cpp src/kv_prefix_cache.cpp -o $@

build/test_backend_stream_tracker: tests/test_backend_stream_tracker.cpp src/backend_stream_tracker.hpp | build
	$(CXX) $(CXXFLAGS) tests/test_backend_stream_tracker.cpp -o $@

test: $(TESTS)
	./build/test_kv_prefix_cache
	./build/test_lcp_ubatch_split
	./build/test_backend_stream_tracker
	@echo "ALL TESTS PASSED"

clean:
	rm -rf build

.PHONY: all test clean
