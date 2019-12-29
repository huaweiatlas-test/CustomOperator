# Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details at
# http://www.apache.org/licenses/LICENSE-2.0
#
TARGET := libcustomop_demo.so

#Ensure that there is no space at the end of each line. Otherwise, the problem is caused by the directory name.

ifeq ($(DDK_PATH),)
TOPDIR :=/home/muser/tools/che/ddk/ddk
else
TOPDIR := $(DDK_PATH)
endif

CC := LD_LIBRARY_PATH=$(TOPDIR)/uihost/lib:$$LD_LIBRARY_PATH g++

LOCAL_DIR  := ./
INCLUDE_DIR := $(TOPDIR)/include

SRC_DIR = $(LOCAL_DIR)
BUILD_DIR = tmp
OBJ_DIR = $(BUILD_DIR)/obj
DEPS_DIR  = $(BUILD_DIR)/deps

#Add other header file paths here.
INC_DIR = \
    -I$(SRC_DIR)\
    -I$(INCLUDE_DIR)/inc\
    -I$(INCLUDE_DIR)/inc/custom\
    -I$(INCLUDE_DIR)/inc/graph\
    -I$(INCLUDE_DIR)/third_party/protobuf/include\
    -I$(INCLUDE_DIR)/libc_sec/include \
    -I/usr/include/python2.7

#Add compilation parameters here.
CC_FLAGS := $(INC_DIR) -std=c++11 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack
LNK_FLAGS := \
    -L/usr/lib/python2.7/config-x86_64-linux-gnu -lpython2.7\
    -L$(TOPDIR)/uihost/lib -lomg -lte_fusion \
    -shared

#Recursively traverse level-3 subdirectories here.
DIRS := $(shell find $(SRC_DIR) -maxdepth 3 -type d)

#Add each subdirectory to the search path.
VPATH = $(DIRS)

#Search for all cpp files that contain subdirectories in src_dir.
SOURCES  = $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cpp))
OBJS   = $(addprefix $(OBJ_DIR)/,$(patsubst %.cpp,%.o,$(notdir $(SOURCES))))
DEPS  = $(addprefix $(DEPS_DIR)/, $(patsubst %.cpp,%.d,$(notdir $(SOURCES))))

$(TARGET):$(OBJS)
	$(CC) $^ $(LNK_FLAGS) -o $@

#Before compilation, create the OBJ directory and ensure that the directory exists.
$(OBJ_DIR)/%.o:%.cpp
	@if [ ! -d $(LOCAL_DIR)/proto/tensorflow ]; then mkdir -p $(LOCAL_DIR)/proto/tensorflow; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/versions.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/versions.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/types.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/types.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/tensor_shape.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/tensor_shape.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/resource_handle.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/resource_handle.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/tensor.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/tensor.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/op_def.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/op_def.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/node_def.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/node_def.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/function.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/function.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/graph.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/graph.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/attr_value.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/attr_value.proto; fi;
	@if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi;
	$(CC) -c $(CC_FLAGS) -o $@ $<

#Before compilation, create the DEPS directory to ensure that the directory exists.
#The symbol @ indicates that the execution of hidden commands is printed.
$(DEPS_DIR)/%.d:%.cpp
	@if [ ! -d $(LOCAL_DIR)/proto/tensorflow ]; then mkdir -p $(LOCAL_DIR)/proto/tensorflow; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/versions.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/versions.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/types.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/types.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/tensor_shape.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/tensor_shape.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/resource_handle.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/resource_handle.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/tensor.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/tensor.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/op_def.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/op_def.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/node_def.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/node_def.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/function.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/function.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/graph.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/graph.proto; fi;
	@if [ ! -f $(LOCAL_DIR)/proto/tensorflow/attr_value.pb.h ]; then LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$(TOPDIR)/uihost/lib $(TOPDIR)/uihost/bin/protoc --cpp_out=$(LOCAL_DIR)/proto/tensorflow --proto_path=$(INCLUDE_DIR)/inc/custom/proto/tensorflow $(INCLUDE_DIR)/inc/custom/proto/tensorflow/attr_value.proto; fi;
	@if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi;
	@if [ ! -d $(DEPS_DIR) ]; then mkdir -p $(DEPS_DIR); fi;
	set -e; rm -f $@;
	$(CC) -MM $(CC_FLAGS) $< > $@.$$$$;
	sed 's,\($*\)\.o[ :]*,$(OBJ_DIR)/\1.o $@ : ,g' < $@.$$$$ > $@;
	rm -f $@.$$$$

#The plus sign (-) indicates that the error is ignored.
ifneq ($(MAKECMDGOALS), clean)
	-include $(DEPS)
endif

.PHONY : clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET) proto

