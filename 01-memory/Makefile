NVCC=nvcc
override INCLUDE_DIRS += -I. -I./include
override CXXFLAGS += 
override LDFLAGS +=
BINDIR := bin

SRC_FILES:=$(shell find . -iname "*.cu")
TARGETS:=$(patsubst %.cu, $(BINDIR)/%, $(SRC_FILES))

all: $(TARGETS)

$(TARGETS): $(SRC_FILES)
	@mkdir -p $(BINDIR)
	$(NVCC) -o $@ $<

.PHONY: clean

clean:
	if [ -d "$(BINDIR)" ]; then rm -rf $(BINDIR); fi