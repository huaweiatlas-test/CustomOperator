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
#!/bin/bash
#params check
if [ "$DDK_PATH" == "" ]
then
    echo "no env DDK_PATH set, exit!"
    eit 1
fi 

cur_dir=`pwd`
$DDK_PATH/bin/x86_64-linux-gcc5.4/omg --model=$cur_dir/custom_minimum.pb \
--framework=3 --output=$cur_dir/custom_minimum \
--net_format=ND --plugin_path=$cur_dir/../plugin --ddk_version="1.31.T10.B100"
