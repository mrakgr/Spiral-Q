// I do not really undestand why with two streams, the following example runs correctly.

// Edit: Ok, I think I understand it.
// It seems that event record adds a token to the stream queue and when a different 
// stream waits on that event, the only thing it is looking at is that specific token.

// For that reason, the following example works perfectly with 4 streams and 1 event.
// I thought it would definitely fail, but it did not.

// Actually, this way of doing concurrency is quite well thought out, if not particularly
// well explained.

// Edit2: Given what I wrote here it occurs to me that I do not need the `s.WaitEvent e.Event`
// in the StreamPool.

open System
open System.IO

#if INTERACTIVE
#load "spiral_q_v0.fsx"
#endif
open SpiralV3
open ManagedCuda
open ManagedCuda.BasicTypes

printfn "%A" <| ctx.GetFreeDeviceMemorySize()

let a = d4MUnion.createConstant((1024,1024,1,1))
a.setPrimal 1.0f
let a' = d4MUnion.createConstant((1024,1024,1,1))
a'.setPrimal 1.0f
let b' = d4MUnion.createConstant((1024,1024,1,1))
b'.setPrimal 1.0f
let c' = d4MUnion.createConstant((1024,1024,1,1))
c'.setPrimal 1.0f
let d' = d4MUnion.createConstant((1024,1024,1,1))
d'.setPrimal 1.0f
let e' = d4MUnion.createConstant((1024,1024,1,1))
e'.setPrimal 1.0f
let f' = d4MUnion.createConstant((1024,1024,1,1))
f'.setPrimal 1.0f

ctx.Synchronize()

CudaContext.ProfilerStart()

let b = matmult a a'
let c = matmult b b'
let d = matmult c c'
let e = matmult d d'
let f = matmult e e'

f.P.Gather() |> Array.sum
|> printfn "sum=%f"

CudaContext.ProfilerStop()