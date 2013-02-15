A collection of tools for use with Nvidia GPUs and CUDA.

**nvlock:** Set a file lock on `/dev/nvidia?` and ignore locked devices when
creating a CUDA context.

**nvtop:** Display GPU utilisation and compute processes. Reformats XML output of
`nvidia-smi`.

**nvcuda-occupancy:** Compute device occupancy of a CUDA kernel. Wraps
OccupancyRecord of PyCUDA.
