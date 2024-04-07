
CC := g++
CFLAGS := -Wall -O3 -std=c++17
INCLUDE := -I/usr/local/include/eigen -Icommon

SRCDIR = src
OBJDIR = obj
BINDIR = bin
TARGET := bin/DNN_test

SRCS := $(wildcard ./$(SRCDIR)/*.cpp)
#SRCS += dnn_main.cpp dnn_simple_3D.cpp
OBJS := $(patsubst ./$(SRCDIR)/%.cpp, ./$(OBJDIR)/%.o, $(SRCS))
LIBPNG_CXX := $(shell libpng-config --cppflags)
LIBPNG_LDF := $(shell libpng-config --ldflags)
LIBS := $(LIBPNG_CXX) $(LIBPNG_LDF)


.PHONY: clean all re

all: $(TARGET)

$(TARGET): $(OBJS) | $(BINDIR)
	$(CC) $(INCLUDE) $(OBJS) -o $@ $(CFLAGS) $(LIBS)

$(OBJDIR)/%.o:$(SRCDIR)/%.cpp | $(OBJDIR)
	$(CC) $(INCLUDE) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(BINDIR) $(OBJDIR)

re: clean all
