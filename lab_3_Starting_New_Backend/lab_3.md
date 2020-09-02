# Starting a New Backend

## Preface

ONNC as an AI compiler framework, intends to be flexible and easy to incorporate a variety of deep learning accelerator (DLA) hardware.
The following figure shows the software architecture of ONNC. 

<img src="../figures/onnc-software-architecture-diagram.png" width="500">

The ONNC compiler has a general frontend framework to parse AI models and lower its representation to the ONNX IR graph. For each target hardware platform, compiler has a corresponding backend to deal with target-dependent tasks. There are two possible paths in porting a new backend. Processor-type targets follow the left path in the above diagram to emit LLVM IRs and be compiled to target machine code using the LLVM cross-compiler. Other proprietary DLA designs follow the right path to have a customized backend. In the case of NVDLA, we take the right path for ONNC porting to the NVDLA hardware. Each backend in ONNC performs target-specific conversion, and ONNC can have multiple backends for supporting different DLAs. ONNC provides a script to generate a code skeleton for a new backend. In this tutorial, we will describe how to use the script to have a jump start in backend porting. 

In terms of file structure, all backend code is placed inside the directory, `<path/to/onnc>/lib/Target`. There are two backends available in that directory, including NVDLA and X86. As the above figure shows, there are a couple of default stages in each backend, including TensorSel, TensorScheduling, MemoryAllocation, and CodeEmit. The backend design in the ONNC framework has a significant control of the compilation process. Developers may decide whether and how to design each stage on their own. We recommend generating a new backend using the provided backend-creating script and make the necessary modification based on your own needs. This lab will demonstrate how to generate a new backend, how to compile ONNC, and how to run ONNC to compile an AI model respectively. 


## Lab: Creating a Backend -- FooNvdla

### Step 1: Set up environment.

Please finish the following labs before continuing this lab.
* [lab 1: Environment Setup](../lab_1_Environment_Setup/lab_1.md) for preparing the Docker images and ONNC source codes.

The backend-creating script is included in the ONNC source code and can be run within the ONNC-community Docker container.

### Step 2: Run the backend-creating script.

Running the script needs certain packages installed in your working environment. We have prepared a pre-build working environment, the ONNC-community Docker, for fast setup. Please run the script within the Docker container. 

```sh
# Use the interactive mode to enter the Docker prompt. You will run the script inside.
$ docker run -ti --rm -v <path/to/onnc>:/onnc/onnc onnc/onnc-community
```

We have described how to set up a pre-built working environment in [Lab 1](../lab_1_Environment_Setup/lab_1.md). If you are not familiar with the ONNC-community Docker container, please go through Lab 1 first to setup your working environment. Once you enter the Docker container, type the following commands in the prompt to create a new backend called FooNvdla.

```sh
# Within the onnc/onnc-community Docker container

# Go to the path where the ONNC source codes are mounted to.
$ cd /onnc/onnc

# Run the script to create a new backend called FooNvdla.
$ ./scripts/create-new-backend.sh FooNvdla
```

The new backend FooNvdla will be placed inside the folder `<path/to/onnc>/lib/Target/FooNvdla`. Since we mount the ONNC source code to the Docker, any change inside the Docker container can be seen outside the container as well. You can find the generated files on your computer outside the Docker container.

### Step 3: Compile the new backend

After creating the new backend with that script, you have a runnable backend that just dumps the model information by default. In this step, let's rebuild ONNC and compile a DNN model.

Use the following commands to compile ONNC with the new backend.

```sh
# Within the onnc/onnc-community Docker container

$ cd /onnc/onnc-umbrella/build-normal/

# Use “-j8” to invoke 8 CPU cores to do the parallel compilation.
$ smake -j8 install
# ...
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc/Diagnostic/Bits/header.h
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc/Diagnostic/OFStreamLog.h
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc/Diagnostic/Diagnostic.h
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc/Diagnostic/MsgHandler.h
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc/Diagnostic/EngineFwd.h
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc/Diagnostic/StreamLog.h
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc/Diagnostic/MsgHandling.h
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc/Support
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc/Support/DataTypes.h
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc/Config
-- Up-to-date: /onnc/onnc-umbrella/install-normal/include/onnc/Config/ONNX.h
-- Installing: /onnc/onnc-umbrella/install-normal/include/onnc/Config/Platforms.def
-- Installing: /onnc/onnc-umbrella/install-normal/include/onnc/Config/Backends.def
-- Installing: /onnc/onnc-umbrella/install-normal/include/onnc/Config/Config.h
```

### Step 4: Compile an AI model

The following commands demonstrate how to compile the `AlexNet` model with the ONNC binary. 

```sh
# Within the onnc/onnc-community Docker container

$ onnc -mquadruple foonvdla /models/bvlc_alexnet/model.onnx
```

The option `-mquadruple foonvdla` is for invoking the new backend FooNvdla. Note that ONNC only accepts lowercase letters as the backend name in this option. When you use uppercase letters for the new backend name, the `create-new-backend.sh` script will convert them to lowercase letters automatically. 

The following log shows the compilation result and it dumps the model graph information in the `AlexNet` model. 

```sh
FooNvdla is invoked
%conv1_w_0<float>[96, 3, 11, 11] = Initializer<unimplemented>()
%conv1_b_0<float>[96] = Initializer<unimplemented>()
%conv2_w_0<float>[256, 48, 5, 5] = Initializer<unimplemented>()
%conv2_b_0<float>[256] = Initializer<unimplemented>()
%conv3_w_0<float>[384, 256, 3, 3] = Initializer<unimplemented>()
%conv3_b_0<float>[384] = Initializer<unimplemented>()
%conv4_w_0<float>[384, 192, 3, 3] = Initializer<unimplemented>()
%conv4_b_0<float>[384] = Initializer<unimplemented>()
%conv5_w_0<float>[256, 192, 3, 3] = Initializer<unimplemented>()
%conv5_b_0<float>[256] = Initializer<unimplemented>()
%fc6_w_0<float>[4096, 9216] = Initializer<unimplemented>()
%fc6_b_0<float>[4096] = Initializer<unimplemented>()
%fc7_w_0<float>[4096, 4096] = Initializer<unimplemented>()
%fc7_b_0<float>[4096] = Initializer<unimplemented>()
%fc8_w_0<float>[1000, 4096] = Initializer<unimplemented>()
%fc8_b_0<float>[1000] = Initializer<unimplemented>()
%OC2_DUMMY_1<int64>[2] = Initializer<unimplemented>()
%data_0<float>[1, 3, 224, 224] = InputOperator<unimplemented>()
%conv1_1<float>[1, 96, 54, 54] = Conv<auto_pad: "NOTSET", dilations: [1, 1], group: 1, kernel_shape: [11, 11], pads: [0, 0, 0, 0], strides: [4, 4]>(%data_0<float>[1, 3, 224, 224], %conv1_w_0<float>[96, 3, 11, 11], %conv1_b_0<float>[96])
%conv2_1<float>[1, 256, 26, 26] = Conv<auto_pad: "NOTSET", dilations: [1, 1], group: 2, kernel_shape: [5, 5], pads: [2, 2, 2, 2], strides: [1, 1]>(%pool1_1<float>[1, 96, 26, 26], %conv2_w_0<float>[256, 48, 5, 5], %conv2_b_0<float>[256])
%conv3_1<float>[1, 384, 12, 12] = Conv<auto_pad: "NOTSET", dilations: [1, 1], group: 1, kernel_shape: [3, 3], pads: [1, 1, 1, 1], strides: [1, 1]>(%pool2_1<float>[1, 256, 12, 12], %conv3_w_0<float>[384, 256, 3, 3], %conv3_b_0<float>[384])
%conv4_1<float>[1, 384, 12, 12] = Conv<auto_pad: "NOTSET", dilations: [1, 1], group: 2, kernel_shape: [3, 3], pads: [1, 1, 1, 1], strides: [1, 1]>(%conv3_2<float>[1, 384, 12, 12], %conv4_w_0<float>[384, 192, 3, 3], %conv4_b_0<float>[384])
%conv5_1<float>[1, 256, 12, 12] = Conv<auto_pad: "NOTSET", dilations: [1, 1], group: 2, kernel_shape: [3, 3], pads: [1, 1, 1, 1], strides: [1, 1]>(%conv4_2<float>[1, 384, 12, 12], %conv5_w_0<float>[256, 192, 3, 3], %conv5_b_0<float>[256])
 = OutputOperator<unimplemented>(%prob_1<float>[1, 1000])
```

Congratulations! Now you have your new backend ready. In the subsequent tutorial labs, you are going to add more functionalities to the new backend.

## Files within a new backend

By following the commands in the previous section, we have created a new backend FooNvdla and all the files are generated in the `lib/Target/FooNvdla` directory. The following table lists the files in the created folder. 

| File | Purpose |
| ---- | ------- |
| `FooNvdlaBackend.cpp & .h` | The main file of a backend. Developers need to modify this file to add optimization passes. |
| `CodeEmitVisitor.cpp & .h` | Implementation of the `CodeEmitVisitor` class. Developers need to modify this file to handle the code generation for each operators. |
| `TargetInfo/FooNvdlaTargetInfo.cpp & .h` | This file containing functions for registering this backend to the ONNC framework. |
| `TargetInfo/FooNvdlaTargetMemInfo.cpp & .h` | The file for configuring  memory size and alignment for each data type in neural network models. Developers need to modify this file based on the target hardware attributes to optimize memory allocation. |
| `CMakeLists.txt` | Configuration file for the CMake building system. |
| `Makefile.am` | Configuration file for the Autotools building system. |


