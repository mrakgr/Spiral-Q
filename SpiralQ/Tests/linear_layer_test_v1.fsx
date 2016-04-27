// This rewrite was a complete success from the performance perspective.
// In finishes the benchmark identically time as _v0 example. There is some variance
// depending on when I run the algorithm though. I am yet to determine why this is.

open System
open System.IO

#if INTERACTIVE
#load "spiral_q_v2.fsx"
#endif
open SpiralV3

let t =
    Array.init 500
    <| fun _ ->
       [|(d4MUnion.createConstant((256,256,1,1)),
          d4MUnion.createConstant((256,256,1,1)));
         (d4MUnion.createConstant((256,256,1,1)),
          d4MUnion.createConstant((256,256,1,1)));
         (d4MUnion.createConstant((256,256,1,1)),
          d4MUnion.createConstant((256,256,1,1)))|]

let unoptimized_lin_layer_test() =
    let stopwatch = Diagnostics.Stopwatch.StartNew()
    for i= 1 to 10 do
        // So it is 0.762 (1 stream) vs 0.34 (32 streams.)
        t |> Array.map (fun x -> linear_layer_matmult x None |> sigmoid |> sigmoid |> sigmoid) |> ignore
        ObjectPool.ResetPointers()
        tape.Clear()
    stopwatch.ElapsedMilliseconds

unoptimized_lin_layer_test()