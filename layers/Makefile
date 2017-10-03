CC = gcc
CFLAGS = -std=c++11 -shared -fPIC -O2

TARGETS = rpl nms

# Uncomment line before if compiling on OSX platform
#CFLAGS += -undefined dynamic_lookup

# Comment line below if using gcc4 OR if compiled tensorflow from source with gcc5 or later
CFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0


TENSORFLOW_INCLUDE_DIR := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') 


all: $(TARGETS)

$(TARGETS): % : %.cc
		$(CC) $(CFLAGS) -o $@.so -I$(TENSORFLOW_INCLUDE_DIR) $@.cc

.PHONY: clean
clean: $(RM) $(TARGETS).so
