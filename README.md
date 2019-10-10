## Introduction

The NVIDIA Deep Learning Accelerator provides free intellectual property licensing to anyone wanting to build a chip that uses deep neural networks for inference applications. With extensive documentation and tools, many business proposals and research projects choose NVDLA as their inference engine design. However, lack of extensible compiler support becomes the major bottleneck for supporting more AI models and optimizations. This tutorial presents the first open source compiler that supports NVDLA-based designs. The ONNC compiler has more support than the official NVDLA compiler and relieves programmers from manually specifying the low-level details of models that are not supported by the official NVDLA compiler. It also enables the opportunities for hardware customization and proprietary optimization. We will cover the overview, porting and optimizations in three subsections. In each subsection, we will have hands-on labs to demonstrate how to run and customize the NVDLA backend in ONNC for product development and research projects.

ONNC (Open Neural Network Compiler) is a retargetable compilation framework designed specifically for proprietary deep learning accelerators. Its software architecture expedites porting ONNC to any Deep Learning Accelerator (DLA) design that supports [ONNX (Open Neural Network Exchange)](https://onnx.ai/) operators. ONNC guarantees executability across every DLA by means of transforming ONNX models into DLA-specific binary forms and leveraging the intermediate representation (IR) design of ONNX along with effective algorithms to eliminate the overhead of data movement. **ONNC is the first open source compiler available for NVDLA-based hardware designs**. Its NVDLA backend can compile a model into an executable NVDLA Loadable file. Integrating ONNC with the NVDLA software stack opens up opportunities for developers and researchers to explore the NVDLA-based inference design at system level. 

## Intended Audience

Researchers and practitioners in academia or industry looking for an open-source AI compiler for NVDLA-based neural network inference engines.

## Contributors

* Wei-Fen Lin (weifen@skymizer.com)
* Cheng-Tao Hsieh (cthsieh@skymizer.com)

## Hands-on Labs

* Lab 1. [ONNC Working Environment Setup](https://github.com/ONNC/onnc-tutorial/blob/master/lab_1_Environment_Setup/lab_1.md)
* Lab 2. [Digit Recognition with ARM Cortex-M](https://github.com/ONNC/onnc-tutorial/blob/master/lab_2_Digit_Recognition_with_ARM_CortexM/lab_2.md)
* Lab 3. [Starting a New Backend](https://github.com/ONNC/onnc-tutorial/blob/master/lab_3_Starting_New_Backend/lab_3.md)
* Lab 4. [Code Emitting](https://github.com/ONNC/onnc-tutorial/blob/master/lab_4_Code_Emitting/lab_4.md)
* Lab 5. [CPU Fallback Support](https://github.com/ONNC/onnc-tutorial/blob/master/lab_5_CPU_Fallback/lab_5.md)
* Lab 6. [Manipulating ONNC IR and Optimization](https://github.com/ONNC/onnc-tutorial/blob/master/lab_6_Manipulating_ONNC_IR/lab_6.md)
* Lab 7. [ONNC IR Extension](https://github.com/ONNC/onnc-tutorial/blob/master/lab_7_ONNC_IR_Extension/lab_7.md)
* Lab 8. [Hardware-specific Optimization](https://github.com/ONNC/onnc-tutorial/blob/master/lab_8_Mul_Add_Reordering_and_Fusion/lab_8.md)

## References

### Papers

* W. F. Lin, D. Y. Tsai, L. Tang, C. T. Hsieh, C. Y. Chou, P. H. Chang, and L. Hsu, “ONNC: A compilation framework connecting ONNX to proprietary deep learning accelerators,” in IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS 2019). IEEE, 2019.

* W.F. Lin, C. T. Hsieh, C. Y. Chou, "ONNC-based Software Development Platform for Configurable NVDLA Designs", to appear in IEEE International Symposium on VLSI Design, Automation and Test (VLSI-DAT 2019). IEEE, 2019

### Documentation

- [ONNC Utilities](docs/ONNC-Utilities.md)
- [ONNC Pass Manager Getting Started Guide](docs/ONNC-Pass-Manager-Getting-Started-Guide.md)
- [ONNC Backend Developer Guide](docs/ONNC-Backend-Porting-Guide.md)
- [The Code Emitting Pass User Guide](docs/The-Code-Emitting-Pass-User-Guide.md)
- [ONNC IR Extension Guide](docs/ONNC-IR-Extension-Guide.md)


