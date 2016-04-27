// Since I decided to not do the union type for the time being, I want to find out how much I am missing out on.
// If the cuDNN v5 post was right then multiplying 128x128 4 times should be 2x slower than multiplying 128x512 once.

// Result: A single large MM is 50% faster with 32 concurrent streams and 110% with just one. There is something to that optimization.
// This is with 128x128 * 128x128 vs 128x128 * 128x512 matrices. Let me try larger ones.
// With 512x512 * 512x512 vs 512x512 * 512x2048. The advantage virtually disappears.
// As a matter of fact doing 4 512x512 * 512x512 matrix multiplies is in fact ~10% faster than a single big one.

// Edit: With a 10 fold multiply of 128x128 * 128x128 I get 0.24s (0.32s with just one stream). With 1 op of 128x128 * 128x1280 I get 0.08s.
// Lesson: Cuda gets significantly more efficient the more parallelism you expose (in constrast to concurrency.)
// The ideal situation would be for all the library functions (like gemm) to be inlined and fused as much as possible into one efficient kernel.

open System
open System.Collections.Generic
open System.IO

#if INTERACTIVE
#load "spiral_q_v2.fsx"
#endif
open SpiralV3

open ManagedCuda
open ManagedCuda.CudaDNNv5

ctx.GetFreeDeviceMemorySize() |> ignore

let a = d4MUnion.createConstant((128,128,1,1))
let b = d4MUnion.createConstant((128,128,1,1))


for i=1 to 10 do
    matmult a b |> ignore
ObjectPool.ResetPointers()
tape.Clear()
ctx.Synchronize()
let stopwatch = Diagnostics.Stopwatch.StartNew()
for i=1 to 1000 do
    for j=1 to 10 do
        matmult a b |> ignore
    ObjectPool.ResetPointers()
    tape.Clear()
    ctx.Synchronize()

let time_elapsed = stopwatch.ElapsedMilliseconds



