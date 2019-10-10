# ONNC Working Environment Setup

## Preface

This tutorial targets at using ONNC to generate Loadables that contains DNN model graph information for running inference on NVDLA-based SoCs. Most information in this tutorial is specifically tailored to the NVDLA backend porting.   
To facilitate the software development process, an ONNC Docker image is available in the Docker Hub for fast deployment. It has pre-installed dependent libraries and is a ready-to-run working environment. Users can mount ONNC source code into the Docker container and build the source code inside the container. In addition, the built ONNC binary can be executed to compile deep neural network (DNN) models inside the container. ONNC currently provides two backend implementations in the GitHub release v1.2. For the x86 backend, users may run model inference using the embedded interpreter, ONNI. For the NVDLA backend, a Loadable file that contains model graph information is generated after compilation. Users may simulate the model inference by running the Loadable files on an NVDLA virtual platform. The [NVIDIA Deep Learning Accelerator (NVDLA)](http://nvdla.org/index.html) release provides a full-featured virtual platform for full-system software simulation. We leverage the officially released virtual platform and make small changes for this tutorial. 

In the first Lab, we will describe and demonstrate how to build ONNC, compile models using ONNC, and simulate the model inference on our pre-packed virtual platform.

## Prerequisite

If Docker is not installed in your system, please download Docker (http://www.docker.com) and install it first.
You also need to install Git (https://git-scm.com/) to retrieve the source codes from Git servers.

## Preparing Source Code and Docker Images

The latest ONNC source code is available on GitHub. Please follow the following commands to download the source code.

```sh
$ git clone https://github.com/ONNC/onnc.git
$ cd onnc
$ git checkout tags/1.2.0
$ cd ..
```

Use the following command to download the tutorial material. There are some example DNN models and code snippets you will use in the subsequent labs.

```sh
$ git clone https://github.com/ONNC/onnc-tutorial.git
```

Pull the Docker image from the Docker Hub using the following commands.

```sh
# We need two Docker images.

$ docker pull onnc/onnc-community
$ docker pull onnc/vp
```

To verify that the Docker images were downloaded successfully, use the following command to show all available Docker images. You should see both `onnc/onnc-community` and `onnc/vp` images.

```sh
$ docker images
REPOSITORY                           TAG                                IMAGE ID            CREATED             SIZE
onnc/onnc-community                  latest                             fdd06c76c519        2 days ago          5.58GB
onnc/vp                              latest                             889c00396ea1        2 days ago          2.16GB
```


## Building ONNC and Compiling DNN Models

Use the following command to bring up the ONNC-community Docker.

```sh
$ docker run -ti --rm -v <absolute/path/to/onnc>:/onnc/onnc -v <absolute/path/to/tutorial>:/tutorial onnc/onnc-community
```

* `<absolute/path/to/onnc>` is the directory where you clone the ONNC source code. Note that it must be the absolute path other than a relative path.
* `<absolute/path/to/tutorial>` is the directory where you clone the ONNC tutorial material.
* The `-ti` option provides an interactive interface for the container.
* The `--rm` option will automatically clean up the container when the container exits.
* The `-v` option mounts the directory to the Docker container. With this option, you can make change to the source code (<path/to/onnc>) outside the Docker container with your favorite editor, and the change can be seen inside the Docker container and gets compiled.

Within the Docker container, use the following commands to build ONNC.

```sh
# Within onnc/onnc-community Docker container

$ cd /onnc/onnc-umbrella/build-normal

# Build ONNC.
$ smake -j8 install
```

* The `smake` command synchronizes the build directory with `<path/to/onnc>/onnc` and invokes the make command to build ONNC. 
* The `-j8` option is to parallelize compilation with 8 CPU cores.
* This command will automatically install the compiled binary in this container environment.

```sh
# Run ONNC to compile a DNN model.
$ onnc -mquadruple nvdla /tutorial/models/lenet/lenet.onnx

# Prepare the compiled output file for the virtual platform to run.
$ sudo mv out.nvdla /tutorial/models/lenet/
```

You may use the following command to exit the Docker prompt, 

```sh
# Within the onnc/onnc-community Docker container
$ exit
```

## Performing Model Inference on Virtual Platform

When you finish building ONNC and compiling a DNN model, you do not need the `onnc/onnc-community` Docker anymore. Start another console/terminal on your computer to enter the other Docker image called `onnc/vp` for model inference.

```sh
# Within your computer console

$ docker run -ti --rm -v <absolute/path/to/tutorial>:/tutorial onnc/vp
```

The virtual platform in this Docker is used to simulate the NVDLA runtime environment. As the following figure shows, the virtual platform contains a systemC model for the NVDLA hardware as well as a CPU emulator, where a Linux OS and NVDLA drivers are running to drive the NVDLA hardware.

<img src="../figures/runtime_env.png" width="400">

Within the VP Docker container, use the following commands to activate the virtual platform.

```sh
# Within onnc/vp Docker container

$ cd /usr/local/nvdla

# Prepare loadable, input, and golden output for the future use.
$ cp /tutorial/models/lenet/* .

# Run the virtual platform.
$ aarch64_toplevel -c aarch64_nvdla.lua

             SystemC 2.3.0-ASI --- Oct  9 2017 04:21:14
        Copyright (c) 1996-2012 by all Contributors,
        ALL RIGHTS RESERVED

No sc_log specified, will use the default setting
verbosity_level = SC_MEDIUM
bridge: tlm2c_elaborate..
[    0.000000] Booting Linux on physical CPU 0x0
# ...
Initializing random number generator... done.
Starting network: udhcpc: started, v1.27.2
udhcpc: sending discover
udhcpc: sending select for 10.0.2.15
udhcpc: lease of 10.0.2.15 obtained, lease time 86400
deleting routers
adding dns 10.0.2.3
OK
Starting sshd: [    4.590433] NET: Registered protocol family 10
[    4.606182] Segment Routing with IPv6
OK

Welcome to Buildroot
nvdla login:
```

By starting the virtual platform, a Linux kernel is brought up and stops at the login prompt.

* nvdla login: root
* Password: nvdla

After logging into the Linux prompt, use the following commands to install the drivers.

```sh
# Within the virtual platform

$ mount -t 9p -o trans=virtio r /mnt && cd /mnt

# Install KMD.
$ insmod drm.ko && insmod opendla.ko
[  469.730339] opendla: loading out-of-tree module taints kernel.
[  469.734509] reset engine done
[  469.737998] [drm] Initialized nvdla 0.0.0 20171017 for 10200000.nvdla on minor 0
```

Up to this point, everything is ready for running model inference. In this lab, we demonstrate with a real-world model, LeNet, which is used for hand-written digit recognition. We have prepared some 28x28 images (`.pgm` files) to represent digit numbers 0 to 9. We begin with running model inference to recognize digit number 0 with input file `input0.pgm`. The inference simulation will take about a few minutes. 

```sh
# Within the virtual platform

# Run the NVDLA runtime (containing UMD) to do model inference.
$ ./nvdla_runtime --loadable out.nvdla --image input0.pgm --rawdump
creating new runtime context...
Emulator starting
# ...
[  126.029817] Enter:dla_handle_events, processor:CDP
[  126.029995] Exit:dla_handle_events, ret:0
[  126.030146] Enter:dla_handle_events, processor:RUBIK
[  126.030323] Exit:dla_handle_events, ret:0
[  126.032432] reset engine done
Shutdown signal received, exiting
Test pass
```

After the simulation is done, we will derive an output file `output.dimg` containing the model output values.
In this example, the output file should look like the follows:

```sh
$ more output.dimg
149.25 -49.625 13.875 11.2344 -59.8125 -2.61523 7.80078 -44.7188 30.8594 17.3594
```

In the file, there are ten numbers indicating the confidence level of the 10 digits from 0 to 9, respectively.
For example, the first number 149.25 indicates the confidence level of digit 0, and the next -49.625 of digit 1, and so on. Among those numbers, the largest one implies the recognition result. In this case, the first number 149.25 is the largest one, so the corresponding digit 0 is the recognition result.

After the experiment, you can use the following command to exit the virtual platform.

```sh
# Within the virtual platform
$ poweroff
```

Use the following command to exit the `onnc/vp` Docker prompt.

```sh
# Within the onnc/vp Docker container
$ exit
```
