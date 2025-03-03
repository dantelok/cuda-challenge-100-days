# 100-Day CUDA Challenge

This challenge is inspired by https://github.com/hkproj/100-days-of-gpu from Umar Jamil.

I generate the whole challenge roadmap by ChatGPT.
My target is to be an expert in CUDA after this challenge, so I requested a very challenging roadmap.
I have no knowledge to evaluate whether this roadmap makes sense at this moment, I will review my progress and see if I need modifications in my journey.

**Day 01: Vector Addition Kernel**  
Implement a kernel that adds two float arrays elementwise. Allocate host/device memory, copy data, launch the kernel, and verify results.

**Day 02: Vector Elementwise Multiplication**  
Implement a kernel to multiply two vectors elementwise. Validate your output against a CPU implementation.

**Day 03: Scalar Multiplication of a Vector**  
Write a kernel that multiplies every element of an array by a scalar value.

**Day 04: Naïve Dot Product (Reduction Part 1)**  
Implement a kernel where each thread computes a product of corresponding elements, storing results for later reduction.

**Day 05: Dot Product with Shared Memory Reduction**  
Optimize the dot product by using shared memory for intra-block reduction, then complete the reduction on the host.

**Day 06: Matrix-Vector Multiplication**  
Implement a kernel where each thread computes one output element as the dot product of one row with a vector.

**Day 07: Naïve Matrix-Matrix Multiplication**  
Write a kernel to compute C = A × B, where each thread computes one element of the output matrix.

**Day 08: Optimize Vector Addition for Memory Coalescing**  
Refactor your vector addition kernel to ensure optimal memory coalescing and benchmark the performance difference.

**Day 09: Tiled Matrix Multiplication with Shared Memory**  
Implement matrix multiplication using tiling. Load matrix tiles into shared memory and synchronize threads with `__syncthreads()`.

**Day 10: Reduce Warp Divergence in Conditional Kernels**  
Write a kernel with conditional branches, then refactor it to minimize warp divergence. Compare execution times.

**Day 11: Profile a Kernel Using CUDA Tools**  
Run your matrix multiplication kernel with `nvprof` or Nsight Systems. Record key metrics like execution time and memory throughput.
Here we got no coding but only compilation work, so I add a bit more on day 11.

**Day 11 (New): Implement Tokenization and Word Embeddings on CUDA**
Implement a basic tokenization algorithm on CUDA and map tokens to embeddings using GPU memory.
Parallelize embedding lookup using CUDA kernels and compare CPU vs GPU performance.

**Day 12: Experiment with Grid and Block Configurations**  
Vary the grid/block sizes of your matrix multiplication kernel to find the optimal configuration. Record performance changes.

**Day 13: Naïve Parallel Reduction for Array Sum**  
Implement a basic reduction kernel that sums an array without advanced optimizations. Validate with a CPU sum.

**Day 14: Shared Memory Optimized Reduction**  
Improve your reduction kernel by performing intra-block reduction in shared memory before writing results to global memory.

**Day 15: Exclusive Prefix Sum (Scan) Kernel**  
Implement an exclusive scan (prefix sum) algorithm on an array using shared memory.

**Until day 15, I found that the previous challenge got a lot of overlapping work on similar concepts. 
So I asked ChatGPT to generate a new challenges roadmap from day 16, focusing more on AI applications, including NLP, Computer Visions, Voice processing, and Reinforcement Learning.**

## **Foundational AI Operations in CUDA**
### **Day 16: Fast Tokenization on CUDA**
- Implement a GPU-accelerated subword tokenizer (e.g., BPE or WordPiece).
- Compare speed against Hugging Face’s tokenizer.

### **Day 17: Efficient Word Embedding Lookup on GPU**
- Implement an optimized CUDA kernel for embedding lookup.
- Optimize memory access patterns for minimal latency.

### **Day 18: GPU Accelerated Softmax with Stability Fix**
- Implement softmax using CUDA.
- Fix numerical instability using the **log-sum-exp trick**.

### **Day 19: Fast Layer Normalization Kernel**
- Implement layer normalization using **parallel reduction**.

### **Day 20: Matrix Multiplication Optimized for Deep Learning (Tensor Cores if available)**
- Implement a **CUDA GEMM (General Matrix Multiplication)** kernel optimized for FP16 Tensor Cores.

---

## **Transformer Blocks & Sequence Processing**
### **Day 21: Multi-Head Self-Attention (Naïve Version)**
- Implement a basic self-attention kernel with CUDA.
- Handle variable-length sequences efficiently.

### **Day 22: Efficient Multi-Head Attention with Shared Memory**
- Optimize self-attention using shared memory and tiling.

### **Day 23: Transformer Feedforward Layer with CUDA**
- Implement a **2-layer MLP with GELU activation** for Transformer models.

### **Day 24: LayerNorm + Self-Attention + FFN on GPU**
- Integrate previous optimizations into a full Transformer encoder block.

### **Day 25: CUDA Kernels for LLM Inference Acceleration**
- Implement **fast rotary position embeddings** (RoPE) for Llama models.
- Optimize with warp shuffles.

---

## **Vision Transformers & Large-Scale Processing**
### **Day 26: Image Patching with CUDA for ViTs**
- Implement a CUDA kernel to extract **non-overlapping patches** from an image.

### **Day 27: Vision Transformer Attention Optimization**
- Modify your attention kernel to support image-based **ViT models**.

### **Day 28: CUDA Optimized Convolution for CNNs**
- Implement a **2D convolution kernel with shared memory**.

### **Day 29: Depthwise & Separable Convolution for Mobile AI**
- Implement **depthwise + pointwise** convolution kernels.

### **Day 30: FP16 and INT8 Quantization for Faster Inference**
- Implement quantized matrix multiplication kernels.

---

## **Reinforcement Learning & Generative AI**
### **Day 31: Parallel MCTS for Reinforcement Learning on CUDA**
- Implement **Monte Carlo Tree Search (MCTS)** on CUDA.

### **Day 32: Accelerated Reward Calculation for RL Agents**
- Optimize parallel reward accumulation for RL environments.

### **Day 33: GAN Training Optimizations in CUDA**
- Implement CUDA-accelerated **upsampling** and **activation functions** for GANs.

### **Day 34: Diffusion Model Sampling Optimization**
- Implement a **parallel U-Net step** for diffusion models.

### **Day 35: Faster Text-to-Image Sampling on CUDA**
- Optimize DDIM/SDE sampling.

---

## **Multi-GPU Training & Large Models**
### **Day 36: Implement Data Parallel Training on Multi-GPU**
- Implement **gradient averaging across multiple GPUs**.

### **Day 37: Pipeline Parallelism for Large Transformers**
- Split Transformer layers across GPUs.

### **Day 38: Zero Redundancy Optimizer (ZeRO) Implementation**
- Implement **ZeRO Stage-1 optimizer** in CUDA.

### **Day 39: Multi-GPU KV Caching for LLM Inference**
- Implement **multi-GPU attention KV caching**.

---
## **Days 40-50: Train & Optimize a Full Transformer on CUDA**
Goal: Implement a full **Transformer model** in CUDA, covering **embedding layers, attention mechanisms, feedforward layers, and inference optimizations**.

### **Day 40: Implement Token and Positional Embeddings in CUDA**
- Implement **BPE token embedding lookup** in CUDA.
- Apply **rotary position embeddings** (RoPE) efficiently.

### **Day 41: Self-Attention Computation on CUDA**
- Implement **scaled dot-product attention** with shared memory optimization.
- Parallelize **softmax computation** using **warp intrinsics**.

### **Day 42: Multi-Head Attention Kernel Optimization**
- Implement **multi-head attention** in CUDA.
- Optimize **memory access patterns** to improve throughput.

### **Day 43: Transformer Feedforward Layer with CUDA Optimization**
- Implement **GELU activation function**.
- Use **tiling for matrix multiplications** in feedforward layers.

### **Day 44: Full Transformer Block Implementation**
- Integrate **attention, layer normalization, and feedforward network**.
- Implement **parallel layer norm** with efficient memory access.

### **Day 45: CUDA Kernel Fusion for Transformers**
- Combine **matmul, layer norm, activation, and softmax** into a single fused kernel.

### **Day 46: Mixed-Precision Training with Tensor Cores**
- Optimize GEMM using **FP16 & Tensor Cores**.
- Implement **automatic mixed-precision training**.

### **Day 47: Implement KV Cache Optimization for Fast Inference**
- Implement **efficient key-value caching** for large LLM inference.
- Optimize cache updates for batched queries.

### **Day 48: Transformer Inference Optimization with FlashAttention**
- Implement **FlashAttention** using shared memory and warp shuffles.

### **Day 49: Implement CUDA-Optimized Transformer Training Loop**
- Train a small Transformer from scratch.
- Measure memory and speed improvements with profiling tools.

### **Day 50: Profile Transformer Performance and Apply Further Optimizations**
- Profile memory usage with **Nsight Compute**.
- Optimize block/thread configurations and kernel occupancy.

---

## **Days 51-60: Train & Optimize a Diffusion Model on CUDA**
Goal: Implement **Stable Diffusion** components and optimize **denoising** and **sampling** steps.

### **Day 51: Implement a Simple Gaussian Noise Kernel**
- Implement a kernel that generates **Gaussian noise** for diffusion training.

### **Day 52: CUDA-Optimized U-Net Block**
- Implement **convolutions** and **skip connections** in CUDA.

### **Day 53: Implement a CUDA Kernel for Diffusion Timesteps**
- Implement **variance scheduling** and **noising function**.

### **Day 54: Implement Conditional Guidance for Diffusion Models**
- Implement **classifier-free guidance (CFG)** for stable diffusion.

### **Day 55: Implement Efficient DDIM Sampling**
- Optimize **Denoising Diffusion Implicit Models (DDIM)** sampling.

### **Day 56: Optimize U-Net with Tensor Cores**
- Use **FP16 mixed precision training** for speedup.

### **Day 57: Memory Optimization for Large U-Net Models**
- Implement **checkpointing and activation recomputation**.

### **Day 58: Implement Multi-GPU Training for Diffusion Models**
- Use **data parallel training** with multiple GPUs.

### **Day 59: Implement CUDA Graphs for Efficient Sampling**
- Use **CUDA Graphs** to reduce launch overhead in diffusion sampling.

### **Day 60: Benchmark and Optimize the Full Stable Diffusion Pipeline**
- Compare different **sampling strategies** and **memory layouts**.

---

## **Days 61-70: Build a CUDA-Optimized Reinforcement Learning Agent**
Goal: Implement RL agents like **PPO, SAC, or DDPG** from scratch in CUDA.

### **Day 61: Implement Parallel Action Sampling for Reinforcement Learning**
- Implement a CUDA kernel for **sampling actions from a probability distribution**.

### **Day 62: Implement Parallel Environment Simulation**
- Use **CUDA streams** to simulate **multiple environments** in parallel.

### **Day 63: Implement a CUDA-Optimized Replay Buffer**
- Optimize **experience replay** storage with efficient memory access.

### **Day 64: Implement Parallel Advantage Estimation**
- Compute **Generalized Advantage Estimation (GAE)** in parallel.

### **Day 65: CUDA-Optimized Policy Update for PPO**
- Implement **gradient updates** with **CUDA-accelerated loss calculation**.

### **Day 66: Implement TD3/SAC Policy Optimization with CUDA**
- Implement **twin Q-learning updates** in CUDA.

### **Day 67: Multi-GPU Training for RL Algorithms**
- Implement **distributed reinforcement learning** with multiple GPUs.

### **Day 68: Optimize GPU Inference for RL Agents**
- Implement **batched action inference** for fast decision-making.

### **Day 69: Implement Reward Shaping and Memory Optimizations**
- Optimize how rewards are stored and updated in **CUDA memory**.

### **Day 70: Benchmark RL Training Speed on CUDA**
- Profile training speed **with and without CUDA optimizations**.

---

## **Days 71-80: Implement a CUDA-Optimized Speech Model**
Goal: Implement a CUDA-accelerated **speech recognition** model.

### **Day 71: Implement a CUDA Kernel for MFCC Feature Extraction**
- Compute **Mel-Frequency Cepstral Coefficients (MFCCs)** on CUDA.

### **Day 72: Implement a CUDA Kernel for Spectrogram Computation**
- Compute **log-mel spectrograms** in parallel.

### **Day 73: Implement a CUDA-Optimized Conformer Block**
- Implement **Convolutional Self-Attention** in CUDA.

### **Day 74: Implement CUDA-Optimized Beam Search for ASR Decoding**
- Implement a fast **beam search decoder** with parallel processing.

### **Day 75: Implement CUDA-Optimized RNN-T Model**
- Optimize **transducer models** with efficient memory access.

### **Day 76: Implement Mixed-Precision Training for Speech Models**
- Optimize RNN/T models using **FP16 mixed precision**.

### **Day 77: Implement CUDA-Optimized Attention-Based CTC Loss**
- Implement a fast **Connectionist Temporal Classification (CTC) loss**.

### **Day 78: Benchmark Speech Model Inference on CUDA**
- Profile **latency and throughput** on CUDA.

### **Day 79: Optimize Speech Model for Real-Time Inference**
- Implement **low-latency optimizations**.

### **Day 80: Deploy ASR Model on Multi-GPU Inference System**
- Implement **multi-GPU inference using TensorRT**.

---

## **Days 81-90: CUDA-Optimized Video Processing & Streaming AI**
Goal: Implement **real-time video super-resolution**.

### **Day 81: Implement a CUDA-Optimized Video Frame Loader**
- Implement **asynchronous video frame loading**.

### **Day 82: Implement a Super-Resolution Kernel**
- Implement **ESRGAN or EDSR** models on CUDA.

### **Day 83: Implement Motion Compensation with Optical Flow**
- Implement **TV-L1 optical flow** in CUDA.

### **Day 84: Optimize Video Inference with Tiling**
- Process **large frames** efficiently using **tiled CUDA processing**.

### **Day 85: Implement Real-Time Video Enhancement**
- Optimize **frame interpolation and denoising**.

### **Day 86: Implement CUDA-Accelerated Object Tracking**
- Implement **Kalman filters** for **real-time object tracking**.

### **Day 87: Optimize Video Processing Pipelines**
- Implement **CUDA Streams** to process multiple frames in parallel.

### **Day 88: Implement End-to-End Video Enhancement**
- Build a **real-time AI-enhanced video streaming** pipeline.

### **Day 89: Benchmark and Optimize Video Processing on CUDA**
- Measure **FPS improvements with CUDA optimizations**.

### **Day 90: Deploy Video AI Model with TensorRT**
- Deploy the **video model on an edge device**.

---

## **Days 91-100: Build a Large-Scale CUDA-Optimized LLM Serving System**
Goal: Implement **CUDA-optimized LLM inference** and deploy it.

- **Days 91-94:** Implement KV cache for Transformer inference.
- **Days 95-97:** Implement Tensor Parallelism for LLM inference.
- **Days 98-99:** Optimize Memory and Load Balancing.
- **Day 100:** Deploy LLM system with **CUDA-accelerated serving**.

---