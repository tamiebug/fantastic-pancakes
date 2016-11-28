CC = g++
CFLAGS = -std=c++11 -shared -fPIC -O2

TARGET = rpl

# Uncomment line before if compiling on OSX platform
#CFLAGS += -undefined dynamic_lookup

# Comment line below if using gcc4 OR if compiled tensorflow from source with gcc5 or later
CFLAGS += -DGLIBCXX_USE_CXX11_ABI=0


TENSORFLOW_INCLUDE_DIR := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())') 

all: $(TARGET)

$(TARGET): $(TARGET).cc
		$(CC) $(CFLAGS) -o $(TARGET).so -I$(TENSORFLOW_INCLUDE_DIR) $(TARGET).cc

clean: $(RM) $(TARGET).so
