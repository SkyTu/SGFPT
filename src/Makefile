CUDA_HOME ?= /usr/local/cuda-11.8
CUDA_ARCH ?= 86
NVCC ?= $(CUDA_HOME)/bin/nvcc
REPO_ROOT := $(abspath $(CURDIR)/..)
EIGEN_INCLUDE ?= $(REPO_ROOT)/.deps/usr/include/eigen3
SYTORCH_PATH := $(REPO_ROOT)/ext/sytorch
SYTORCH_BUILD := $(SYTORCH_PATH)/build
LLAMA_PATH := $(SYTORCH_PATH)/ext/llama
UTIL_FILES := utils/gpu_mem.cu utils/gpu_file_utils.cpp utils/sigma_comms.cpp utils/gpu_ctx_init.cu
INCLUDES := -I$(REPO_ROOT)/ext/cutlass/include -I$(REPO_ROOT)/ext/cutlass/tools/util/include -I$(SYTORCH_PATH)/include -I$(LLAMA_PATH)/include -I$(SYTORCH_PATH)/ext/cryptoTools -I$(EIGEN_INCLUDE) -I.
LIB_PATHS := -L$(SYTORCH_BUILD) -L$(SYTORCH_BUILD)/ext/cryptoTools -L$(SYTORCH_BUILD)/ext/llama -L$(SYTORCH_BUILD)/ext/bitpack
LIBS := -lsytorch -lLLAMA -lcryptoTools -lbitpack -lcuda -lcudart -lcurand
FLAGS := -O3 -gencode arch=compute_$(CUDA_ARCH),code=[sm_$(CUDA_ARCH),compute_$(CUDA_ARCH)] -std=c++17 -m64 -Xcompiler="-O3,-std=c++17,-fpermissive,-fPIC,-pthread,-fopenmp,-march=native"
.DEFAULT_GOAL := sgfpt_client
# CUDA implementation templates are included from headers; rebuild on changes to either.
CORE_SOURCES := $(shell find Server-SGFPT fss utils -type f \( -name '*.h' -o -name '*.cu' \))
.PHONY: all clean sgfpt_tests
all: sgfpt_client sgfpt_tests
sgfpt_tests: test_sample test_select_top test_standard_normal test_update test_masked_protocols

.PHONY: sgfpt_client
sgfpt_client: experiments/sgfpt/sgfpt_client
experiments/sgfpt/sgfpt_client: experiments/sgfpt/client.cu $(UTIL_FILES) $(CORE_SOURCES) $(SYTORCH_BUILD)/libsytorch.a
	$(NVCC) $(FLAGS) $(INCLUDES) $< $(UTIL_FILES) $(LIB_PATHS) $(LIBS) -o $@

.PHONY: test_masked_protocols
test_masked_protocols: tests/spt/test_masked_protocols
tests/spt/test_masked_protocols: tests/spt/test_masked_protocols.cu $(UTIL_FILES) $(CORE_SOURCES) $(SYTORCH_BUILD)/libsytorch.a
	$(NVCC) $(FLAGS) -lineinfo $(INCLUDES) $< $(UTIL_FILES) $(LIB_PATHS) $(LIBS) -o $@

.PHONY: test_sample
test_sample: tests/spt/test_sample
tests/spt/test_sample: tests/spt/test_sample.cu $(UTIL_FILES) $(CORE_SOURCES) $(SYTORCH_BUILD)/libsytorch.a
	$(NVCC) $(FLAGS) $(INCLUDES) $< $(UTIL_FILES) $(LIB_PATHS) $(LIBS) -o $@

.PHONY: test_select_top
test_select_top: tests/spt/test_select_top
tests/spt/test_select_top: tests/spt/test_select_top.cu $(UTIL_FILES) $(CORE_SOURCES) $(SYTORCH_BUILD)/libsytorch.a
	$(NVCC) $(FLAGS) $(INCLUDES) $< $(UTIL_FILES) $(LIB_PATHS) $(LIBS) -o $@

.PHONY: test_standard_normal
test_standard_normal: tests/spt/test_standard_normal
tests/spt/test_standard_normal: tests/spt/test_standard_normal.cu $(UTIL_FILES) $(CORE_SOURCES) $(SYTORCH_BUILD)/libsytorch.a
	$(NVCC) $(FLAGS) $(INCLUDES) $< $(UTIL_FILES) $(LIB_PATHS) $(LIBS) -o $@

.PHONY: test_update
test_update: tests/spt/test_update
tests/spt/test_update: tests/spt/test_update.cu $(UTIL_FILES) $(CORE_SOURCES) $(SYTORCH_BUILD)/libsytorch.a
	$(NVCC) $(FLAGS) $(INCLUDES) $< $(UTIL_FILES) $(LIB_PATHS) $(LIBS) -o $@

clean:
	rm -f experiments/sgfpt/sgfpt_client tests/spt/test_sample tests/spt/test_select_top tests/spt/test_standard_normal tests/spt/test_update tests/spt/test_masked_protocols
