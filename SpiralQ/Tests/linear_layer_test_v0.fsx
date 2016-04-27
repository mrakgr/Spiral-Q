// The performance on the LSTM has been disappointing for the new library.
// Here I put all the pieces together and get something better.

// To start off, let me pull out the linear layers that I already have and
// test them as a baseline without activation and bias and hadamarad functions
// getting in the way.

// I want to see how long does it take to run through a bunch of them with one 
// and then with more streams.

open System
open System.IO

#if INTERACTIVE
#load "spiral_q_v1.fsx"
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
        // So it is 0.762 (1 stream) vs 0.508 (32 streams.)
        t |> Array.map (fun x -> linear_layer_matmult x None |> sigmoid |> sigmoid |> sigmoid) |> ignore
        ObjectPool.ResetOccupancy()
        tape.Clear()
    stopwatch.ElapsedMilliseconds

// In this test, the streams are passed along from the previous iteration.
let unoptimized_lin_layer_test2() =
    let stopwatch = Diagnostics.Stopwatch.StartNew()
    for i= 1 to 10 do
        // So it is 0.762 (1 stream) vs 0.305 (32 streams.)
        // It turns out that all that waiting chokes the scheduler. Shit.
        t |> Array.map (fun x -> 
            let [|a1;a2;a3|] = x
            matmult's a1 (StreamPool.P,None)
            |> matmult's a2
            |> matmult's a3
            |> fun (sem, Some x) -> sem, x
            |> sigmoid_s 
            |> sigmoid_s
            |> sigmoid_s
            ) |> ignore
        ObjectPool.ResetOccupancy()
        tape.Clear()
    stopwatch.ElapsedMilliseconds

unoptimized_lin_layer_test2()
