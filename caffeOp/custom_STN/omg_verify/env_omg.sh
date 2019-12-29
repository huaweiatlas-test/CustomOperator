 # ============================================================================
 #
 # Copyright (C) 2019, Huawei Technologies Co., Ltd. All Rights Reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions are met:
 #
 #   1 Redistributions of source code must retain the above copyright notice,
 #     this list of conditions and the following disclaimer.
 #
 #   2 Redistributions in binary form must reproduce the above copyright notice,
 #     this list of conditions and the following disclaimer in the documentation
 #     and/or other materials provided with the distribution.
 #
 #   3 Neither the names of the copyright holders nor the names of the
 #   contributors may be used to endorse or promote products derived from this
 #   software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 # POSSIBILITY OF SUCH DAMAGE.
 # ============================================================================
export DDK_PATH=/home/muser/tools/che/ddk/ddk  # you should change this
version=`cat $DDK_PATH/ddk_info | jq '.VERSION'`
echo "$version"
if [[ $version == \"1.3.* ]];then
    export SLOG_PRINT_TO_STDOUT=1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DDK_PATH/uihost/lib/
    export PYTHONPATH=$DDK_PATH/site-packages
    export PATH=$PATH:$DDK_PATH/toolchains/ccec-linux/bin
    export TVM_AICPU_LIBRARY_PATH=$DDK_PATH/uihost/lib/:$DDK_PATH/uihost/toolchains/ccec-linux/aicpu_lib
    export TVM_AICPU_INCLUDE_PATH=$DDK_PATH/include/inc/tensor_engine
    export TVM_AICPU_OS_SYSROOT=$DDK_PATH/uihost/toolchains/aarch64-linux-gcc6.3/sysroot
elif [[ $version == \"1.31.* ]];then
    export PYTHONPATH=$DDK_PATH/site-packages/te-0.4.0.egg:$DDK_PATH/site-packages/topi-0.4.0.egg
    export LD_LIBRARY_PATH=$DDK_PATH/uihost/lib
    export PATH=$PATH:$DDK_PATH/toolchains/ccec-linux/bin:$DDK_PATH/uihost/bin
    export TVM_AICPU_LIBRARY_PATH=$DDK_PATH/uihost/lib/:$DDK_PATH/uihost/toolchains/ccec-linux/aicpu_lib
    export TVM_AICPU_INCLUDE_PATH=$DDK_PATH/include/inc/tensor_engine
    export TVM_AICPU_OS_SYSROOT=$DDK_PATH/toolchains/aarch64-linux-gcc6.3/sysroot
    export NPU_HOST_LIB=$DDK_PATH/EP/host-x86_64_ubuntu16.04/lib
    export NPU_DEVICE_LIB=$DDK_PATH/EP/device-aarch64_miniOS/lib
else
    echo "[error] only DDK version of C30 or C31 is supported"
fi

