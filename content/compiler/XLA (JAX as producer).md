
XLA is a **cross-platform** **graph-level** compiler whose input is HLO/StableHLO:
- it optimizes computational graph
- its backends support different devices
JAX is one of the frontends that emits HLO/StableHLO computational graph. Other frontends that can give HLO: TensorFlow, PyTorch/XLA


optimized HLO → (XLA emits) LLVM IR → (LLVM's NVPTX backend) PTX → (ptxas) SASS

Can Inductor emit library calls? **Yes**, and it happens in **Inductor**, not Triton. For matmul/conv, Inductor's autotuner chooses among candidates and can select a **cuBLAS/cuBLASLt** call, or **CUTLASS** templates, or a **Triton**-generated kernel, whichever benchmarks fastest for your shapes. The decision and the emission of the library call are **Inductor's** job (it's a graph-level "how do I realize this op" choice, exactly parallel to XLA's backend picking cuBLAS).


Triton: a Python domain specific lauguage that allows users to write kernels with nearly same performance as cuBLAS 

**GPU backend (NVIDIA or AMD):** yes, the two-bucket story holds. Generated kernels → **LLVM IR** → LLVM's target backend → machine code (NVIDIA: NVPTX→PTX→SASS; AMD: AMDGPU backend→GCN/RDNA code). Library bucket → call a vendor library (**cuBLAS/cuDNN** on NVIDIA; **rocBLAS/MIOpen** on AMD). Note even the _library_ differs by vendor, cuBLAS is NVIDIA-only (more on this below).


illustration of JAX frontend -> XLA -> machine codes
![[jax_xla_tech_stack.png]]