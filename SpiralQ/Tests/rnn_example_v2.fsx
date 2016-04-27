// Spiral reverse AD example. Used for testing.
// Embedded Reber grammar LSTM example.

// Adapted from the previous version of the library with some changes.

open System
open System.IO

#if INTERACTIVE
#load "spiral_q_v3_depth.fsx"
#r "../packages/FSharp.Charting.0.90.14/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"
#load "embedded_reber.fsx"
#endif
open SpiralV3

open ManagedCuda
open ManagedCuda.CudaDNNv5

open FSharp.Charting
open Embedded_reber

let reber_set = make_reber_set 3000

let make_data_from_set target_length =
    let twenties = reber_set |> Seq.filter (fun (a,b,c) -> a.Length = target_length) |> Seq.toArray
    let batch_size = (twenties |> Seq.length)

    let d_training_data =
        [|
        for i=0 to target_length-1 do
            let input = [|
                for k=0 to batch_size-1 do
                    let example = twenties.[k]
                    let s, input, output = example
                    yield input.[i] |] |> Array.concat
            yield d4MUnion.createConstant(((batch_size,7,1,1),input))|]

    let d_target_data =
        [|
        for i=1 to target_length-1 do // The targets are one less than the inputs. This has the effect of shifting them to the left.
            let output = [|
                for k=0 to batch_size-1 do
                    let example = twenties.[k]
                    let s, input, output = example
                    yield output.[i] |] |> Array.concat
            yield d4MUnion.createConstant(((batch_size,7,1,1),output))|]

    d_training_data, d_target_data

let lstm_embedded_reber_train num_iters learning_rate (data: d4MUnion[]) (targets: d4MUnion[]) (data_v: d4MUnion[]) (targets_v: d4MUnion[]) clip_coef (l1: LSTMLayer) (l2: INNet) =
    [|
    let l1 = l1
    let l2 = l2
    
    let base_nodes = [|l1.ToArray;l2.ToArray|] |> Array.concat

    let training_loop (data: d4MUnion[]) (targets: d4MUnion[]) =
        let costs = [|
            let mutable a, c = l1.runLayerNoH data.[0]
            let b = l2.runLayer a
            printfn "b=%i" b.depth
            let r = squared_error_cost targets.[0] b
            yield r
    
            for i=1 to data.Length-2 do
                let a',c' = l1.runLayer data.[i] a c
                a <- a'; c <- c'
                let b = l2.runLayer a
                let r = squared_error_cost targets.[i] b
                yield r
            |]
        scale (1.0f/float32 (costs.Length-1)) (sum_scalars costs)

    let mutable r' = 0.0f
    let mutable i = 1
    while i <= num_iters && System.Single.IsNaN r' = false do
        
        let rv = training_loop data_v targets_v
        ObjectPool.ResetPointers()
        tape.Clear()
        
        printfn "The validation cost is %f at iteration %i" rv.P.Value.Value i
        
        let r = training_loop data targets

        printfn "The training cost is %f at iteration %i" r.P.Value.Value i
        
        yield r.P.Value.Value, rv.P.Value.Value

        backprop_tape base_nodes r (sgd learning_rate)

        i <- i+1
        r' <- r.P.Value.Value
    |]

let d_training_data_20, d_target_data_20 = make_data_from_set 20
let d_training_data_validation, d_target_data_validation = make_data_from_set 30

let hidden_size = 64

let l1 = LSTMLayer.createRandomLSTMLayer 7 hidden_size tanh_ tanh_
let l2 = FeedforwardLayer.createRandomLayer (7,hidden_size,1,1) clipped_sigmoid

let learning_rate = 15.0f
// This iteration is to warm up the library. It compiles all the lazy Cuda modules.
lstm_embedded_reber_train 1 learning_rate d_training_data_20 d_target_data_20 d_training_data_validation d_target_data_validation 1.0f l1 l2
let stopwatch = Diagnostics.Stopwatch.StartNew()
let s = [|
        yield lstm_embedded_reber_train 99 learning_rate d_training_data_20 d_target_data_20 d_training_data_validation d_target_data_validation 1.0f l1 l2
        |] |> Array.concat
let time_elapsed = stopwatch.ElapsedMilliseconds
// On the GTX 970, I get 3-4s depending on how hot the GPU is.
let l,r = Array.unzip s

(Chart.Combine [|Chart.Line l;Chart.Line r|]).ShowChart()
