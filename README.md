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

**Day 16: Parallel Bitonic Sort for a Small Dataset**  
Write a kernel to sort an array (e.g., 1024 integers) using the bitonic sort algorithm.

**Day 17: Atomic Counter with Atomic Operations**  
Create a kernel where many threads increment a global counter using atomic operations. Verify the final count.

**Day 18: Histogram Computation Using Atomic Operations**  
Implement a histogram kernel that tallies occurrences in an array. Use atomic operations to update histogram bins.

**Day 19: Dynamic Parallelism**  
Write a kernel that conditionally launches another kernel using dynamic parallelism. Verify that the child kernel executes correctly.

**Day 20: Asynchronous Memory Copy with CUDA Streams**  
Implement host-to-device and device-to-host transfers using `cudaMemcpyAsync` in a stream. Measure the benefit over synchronous copies.

**Day 21: Overlap Data Transfers and Kernel Execution**  
Use multiple CUDA streams to overlap data transfer with kernel execution in a simple application (e.g., vector addition).

**Day 22: Unified Memory Example**  
Convert one of your previous tasks (e.g., vector addition) to use Unified Memory (`cudaMallocManaged`). Compare simplicity and performance.

**Day 23: Image Processing – RGB to Grayscale Conversion**  
Implement a kernel that converts an RGB image (stored as an array) to grayscale using the standard formula.

**Day 24: Image Convolution – Box Blur Filter**  
Write a kernel that applies a 3×3 box blur to an image. Handle image borders appropriately.

**Day 25: Image Edge Detection – Sobel Filter**  
Implement a Sobel edge detection kernel. Compute horizontal and vertical gradients and combine them to detect edges.

**Day 26: Use Texture Memory in the Sobel Filter**  
Modify your Sobel filter kernel to read the input image via texture memory. Compare the performance with global memory access.

**Day 27: Optimize Image Convolution Using Texture Memory**  
Rewrite your box blur kernel to leverage texture caching, and measure the performance improvement.

**Day 28: Compare Global, Shared, and Register Memory Access**  
Create simple kernels that read data from global memory, shared memory, and registers. Measure and compare their latencies.

**Day 29: Matrix Transposition with Coalesced Access**  
Implement a kernel to transpose a matrix. Use shared memory to achieve coalesced global memory accesses during the transpose.

**Day 30: Reduction with Warp Shuffle Intrinsics**  
Modify your reduction kernel to use warp shuffle functions (e.g., `__shfl_down_sync`) for the final stage of reduction within a warp.

**Day 31: Full Warp-Level Reduction**  
Implement a kernel that performs an entire reduction using only warp-level primitives (no shared memory for the final step).

**Day 32: Optimize Block Size Using the CUDA Occupancy Calculator**  
Experiment with different block sizes using the Occupancy Calculator to maximize occupancy for one of your kernels.

**Day 33: Monte Carlo Simulation – Estimating Pi (Initial Version)**  
Implement a kernel where each thread generates random points to estimate Pi using the Monte Carlo method. Sum the results using reduction.

**Day 34: Optimize Monte Carlo Simulation**  
Refine your Monte Carlo kernel to reduce branch divergence and minimize global memory accesses. Compare the estimated Pi value and runtime.

**Day 35: Thread-Level Pseudo-Random Number Generator**  
Implement a simple per-thread pseudo-random number generator (e.g., a linear congruential generator) and integrate it into your Monte Carlo simulation.

**Day 36: Use Pinned (Page-Locked) Memory**  
Modify one of your tasks to use pinned host memory (`cudaHostAlloc`) for faster data transfers. Benchmark the transfer speed.

**Day 37: Zero-Copy Memory Example**  
Implement an example that uses zero-copy memory, where the device directly accesses host-allocated memory. Validate its performance in a simple kernel.

**Day 38: Kernel Timing with CUDA Events**  
Write a wrapper that uses CUDA events to time the execution of any kernel. Use it to time your matrix multiplication kernel.

**Day 39: Benchmark a Kernel with CUDA Events**  
Profile a previously written kernel (e.g., matrix multiplication) using your timing wrapper and experiment with different parameters.

**Day 40: Concurrent Kernel Execution Using Multiple Streams**  
Launch two independent kernels concurrently on separate streams. Ensure correct synchronization and compare execution times.

**Day 41: Query and List GPU Device Properties**  
Write a host program that queries the number of available GPUs and prints properties (e.g., memory, cores) using `cudaGetDeviceProperties`.

**Day 42: Multi-GPU Vector Addition**  
Split the work for vector addition across two GPUs. Allocate separate device memory on each GPU, execute kernels, and merge results.

**Day 43: 2D Convolution for Neural Network Forward Pass**  
Implement a kernel for 2D convolution (3×3 kernel) on an input matrix (simulating an image) to produce a feature map.

**Day 44: ReLU Activation Kernel**  
Write a kernel that applies the ReLU function (max(0, x)) to every element of an input array.

**Day 45: Fully Connected Layer Forward Pass**  
Implement a kernel that performs the forward pass of a fully connected layer (matrix-vector multiplication plus bias addition).

**Day 46: Fully Connected Layer Backward Pass**  
Implement a kernel to compute the gradients (error backpropagation) for a fully connected layer.

**Day 47: Batch Normalization Forward Kernel**  
Write a kernel to compute the mean and variance for a batch of data and normalize it (batch normalization forward pass).

**Day 48: Dropout Layer Kernel**  
Implement a dropout kernel that randomly sets a fraction of the input elements to zero and scales the remaining elements.

**Day 49: Add Robust CUDA Error Checking**  
Create error-checking wrappers for CUDA API calls and integrate them into one of your existing projects.

**Day 50: Custom Device Memory Pool Allocator**  
Implement a simple memory pool allocator on the device to manage a preallocated buffer. Test it with small allocations/deallocations.

**Day 51: Use Custom Allocator in a Reduction Kernel**  
Integrate your custom memory pool into your reduction kernel for temporary storage allocation.

**Day 52: Warp Shuffle Data Exchange Kernel**  
Write a kernel that uses warp shuffle functions (e.g., `__shfl_xor_sync`) to exchange data among threads within a warp.

**Day 53: Reduction Kernel with Cooperative Groups**  
Implement a reduction kernel that leverages both warp shuffle and cooperative groups for intra-block synchronization.

**Day 54: Basic Cooperative Groups Synchronization**  
Write a kernel that groups threads using CUDA cooperative groups and synchronizes them to perform a simple computation.

**Day 55: Multi-Warp Reduction Using Cooperative Groups**  
Modify your reduction kernel to use cooperative groups for synchronization across multiple warps in a block.

**Day 56: Create a Simple CUDA Graph**  
Implement a CUDA Graph that captures a sequence of kernel launches (e.g., vector addition followed by a reduction) and then execute it.

**Day 57: Update and Re-run a CUDA Graph**  
Modify a parameter (such as array size or a constant value) in one kernel node within your CUDA Graph and re-run the graph.

**Day 58: Implement a New CUDA Math Intrinsic**  
Use a new or less-common CUDA math intrinsic (e.g., fast math functions) in a small kernel. Compare its precision and speed with standard functions.

**Day 59: Integrate New Math Functions into an Existing Kernel**  
Update one of your kernels (e.g., Monte Carlo simulation) to use the new math intrinsics and evaluate the performance impact.

**Day 60: 2D Particle System Simulation – Position Update**  
Implement a kernel that updates 2D particle positions based on velocity and a timestep. Use arrays for positions and velocities.

**Day 61: Particle System Optimization with Shared Memory**  
Optimize your particle update kernel by tiling particle data into shared memory. Measure the speedup compared to the naïve version.

**Day 62: Spatial Partitioning for Particles (Uniform Grid)**  
Implement a kernel that assigns each particle to a cell in a uniform grid (spatial partitioning) to prepare for collision detection.

**Day 63: Particle Collision Detection in Grid Cells**  
Write a kernel that checks for collisions between particles within the same grid cell and marks or counts collisions.

**Day 64: Integrate Particle Update and Collision Detection**  
Combine the kernels from Days 60–63 into a cohesive particle simulation where collisions affect particle velocities.

**Day 65: Ray-Sphere Intersection Kernel**  
Implement a kernel where each thread computes the intersection of a ray with a set of spheres. Output the distance to the closest intersection.

**Day 66: Basic CUDA Ray Tracer**  
Develop a simple ray tracer that casts rays from a camera, uses your intersection kernel, and shades pixels based on surface normals.

**Day 67: Bounding Volume Hierarchy (BVH) Construction**  
Implement a host-side BVH for a set of spheres and modify your ray tracer to traverse the BVH for faster intersection testing.

**Day 68: Optimize Ray Tracing with Parallel Reduction**  
Use parallel reduction to determine the closest intersection per pixel in your ray tracer and compare it with your initial approach.

**Day 69: Memory Access Optimization in Ray Tracing**  
Reorder data structures and adjust memory access patterns in your ray tracer for better coalescing. Benchmark the improvements.

**Day 70: Asynchronous Data Transfers in a Complex Kernel**  
Integrate `cudaMemcpyAsync` into your particle simulation to transfer data while the kernel is running, reducing idle time.

**Day 71: Double Buffering with CUDA Streams**  
Implement double buffering in your simulation: while one buffer is processed by a kernel, asynchronously copy data into another buffer.

**Day 72: Compare Unified Memory vs. Explicit Memory Management**  
Run a compute-intensive task (e.g., matrix multiplication) using both Unified Memory and explicit `cudaMalloc`/`cudaMemcpy` to compare performance.

**Day 73: Refactor a Project to Use Unified Memory Exclusively**  
Convert one of your existing projects (e.g., image processing) to use Unified Memory and simplify the memory management code.

**Day 74: Profile a Kernel with Nsight Compute and Optimize**  
Use Nsight Compute to profile your tiled matrix multiplication. Apply one or more optimizations (e.g., loop unrolling) based on the profile data.

**Day 75: Optimize a Challenging Kernel Further**  
Select a kernel you’ve built (e.g., LU decomposition) and apply additional optimizations to maximize throughput and minimize latency.

**Day 76: 1D FFT Kernel Implementation**  
Implement a basic 1D Fast Fourier Transform (FFT) kernel for a small dataset. Verify its correctness against a CPU FFT.

**Day 77: Extend FFT to 2D**  
Build a 2D FFT by applying your 1D FFT kernel on rows and then columns of an image or matrix.

**Day 78: Compare Your 2D FFT with cuFFT**  
Integrate the cuFFT library for the same dataset and compare performance and accuracy with your 2D FFT implementation.

**Day 79: LU Decomposition Kernel**  
Implement a kernel to perform LU decomposition on a small square matrix. Validate the decomposition with known results.

**Day 80: Optimize LU Decomposition with Shared Memory**  
Refine your LU decomposition kernel by using shared memory to speed up intermediate computations.

**Day 81: Solve Linear Systems with LU Decomposition**  
Implement forward and backward substitution kernels to solve Ax = b using the LU factors from Day 80.

**Day 82: Sparse Matrix-Vector Multiplication (CSR Format)**  
Write a kernel for multiplying a sparse matrix (stored in CSR format) by a vector. Verify correctness with a dense multiplication.

**Day 83: Optimize Sparse Matrix Kernel**  
Improve your sparse matrix kernel by ensuring coalesced memory accesses and efficient load balancing across threads.

**Day 84: Breadth-First Search (BFS) on a Graph**  
Implement a BFS kernel on a simple graph represented as an adjacency list. Validate the order of node traversal.

**Day 85: Bellman-Ford Shortest Path Kernel**  
Implement a kernel to compute shortest paths using the Bellman-Ford algorithm on a small weighted graph.

**Day 86: Optimize Graph Algorithms for Sparse Graphs**  
Refine your BFS or Bellman-Ford kernel for better performance on large, sparse graphs (optimize memory accesses and thread usage).

**Day 87: Conway’s Game of Life Kernel**  
Implement a kernel that computes one iteration of Conway’s Game of Life on a 2D grid (e.g., 512×512 cells).

**Day 88: Optimize Game of Life with Tiling and Shared Memory**  
Optimize your Game of Life kernel by using shared memory tiling and handling borders efficiently.

**Day 89: 2D Convolution with Variable Kernel Sizes**  
Implement a generalized 2D convolution kernel that supports different filter sizes and strides. Test it on synthetic data.

**Day 90: Bilateral Filtering on an Image**  
Write a kernel that applies a bilateral filter to an image for edge-preserving smoothing. Validate against a CPU implementation.

**Day 91: Median Filter for Noise Reduction**  
Implement a median filter kernel for image noise reduction. Handle border conditions and compare with a CPU version.

**Day 92: k-Means Clustering for Image Segmentation**  
Implement a kernel that performs k-means clustering on image pixel data. Iterate until cluster centers stabilize.

**Day 93: Particle-Based Fluid Simulation – Core Kernel**  
Implement a kernel that simulates fluid dynamics using a particle-based approach (update positions, velocities, and forces).

**Day 94: Optimize Fluid Simulation with Shared Memory and Tiling**  
Refine your fluid simulation by organizing particles into tiles and using shared memory for neighbor searches.

**Day 95: Matrix Inversion Kernel**  
Implement a kernel that computes the inverse of a small matrix using an iterative method (e.g., Gauss-Jordan elimination).

**Day 96: Singular Value Decomposition (SVD) for a Small Matrix**  
Write a kernel that performs SVD on a small matrix. Validate the decomposition against a known library output.

**Day 97: Conjugate Gradient Solver for Linear Systems**  
Implement a kernel to solve a linear system using the Conjugate Gradient method. Test with a symmetric positive-definite matrix.

**Day 98: Eigenvalue Computation Kernel**  
Write a kernel to compute the eigenvalues (and optionally eigenvectors) of a small matrix using an iterative algorithm.

**Day 99: Custom Optimization Algorithm – Gradient Descent**  
Implement a kernel that performs gradient descent on a mathematical function (e.g., quadratic) to find its minimum.

**Day 100: Final Integration Project**  
Combine techniques from previous tasks to build a high-performance simulation or computation engine (e.g., a full fluid simulation, advanced ray tracer, or a machine learning inference engine). Integrate memory optimizations, dynamic parallelism, and multi-GPU support.