// Spiral reverse AD example. Used for testing.
// Embedded Reber grammar LSTM example.

// Done with this for now. No matter what I do, I cannot really get the benefits of concurrency to 
// go above 30%. Certainly nowhere near as much in the cuDNN post.

open System
open System.Collections.Generic
open System.IO

#if INTERACTIVE
#load "spiral_q_v2.fsx"
#r "../packages/FSharp.Charting.0.90.14/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"
#endif
open SpiralV3

open ManagedCuda
open ManagedCuda.CudaDNNv5

ctx.GetFreeDeviceMemorySize() |> ignore

open FSharp.Charting

let hidden_size = 256

let d1 = d4MUnion.createConstant((hidden_size,hidden_size,1,1))
let d2 = d4MUnion.createConstant((hidden_size,hidden_size,1,1))

let l1 = LSTMLayer.createRandomLSTMLayer hidden_size hidden_size tanh_ tanh_
let l2 = LSTMLayer.createRandomLSTMLayer hidden_size hidden_size tanh_ tanh_
CudaContext.ProfilerStart()


for i=1 to 1 do
    let a,c = l1.runLayerNoH d1
    let a2,c2 = l1.runLayer d2 a c
    let a',c' = l2.runLayerNoH a
    let a2',c2' = l2.runLayer a2 a' c'
    let a,c = l1.runLayerNoH d1
    let a2,c2 = l1.runLayer d2 a c
    let a',c' = l2.runLayerNoH a
    let a2',c2' = l2.runLayer a2 a' c'
    ()
ctx.Synchronize()
let stopwatch = Diagnostics.Stopwatch.StartNew()
ObjectPool.ResetPointers()
tape.Clear()
for i=1 to 100 do
    let a,c = l1.runLayerNoH d1
    let a2,c2 = l1.runLayer d2 a c
    let a',c' = l2.runLayerNoH a
    let a2',c2' = l2.runLayer a2 a' c'
    let a,c = l1.runLayerNoH d1
    let a2,c2 = l1.runLayer d2 a c
    let a',c' = l2.runLayerNoH a
    let a2',c2' = l2.runLayer a2 a' c'
    ObjectPool.ResetPointers()
    tape.Clear()
    ctx.Synchronize()
    ()
ctx.Synchronize()

let time_elapsed = stopwatch.ElapsedMilliseconds
printfn "Time Elapsed: %i" time_elapsed

CudaContext.ProfilerStop()


