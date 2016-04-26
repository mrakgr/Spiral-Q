// Spiral reverse AD example. Used for testing.
// Embedded Reber grammar LSTM example.

// Adapted from the previous version of the library with some changes.
// This example has 5 LSTM layers. At this point I am just groping around trying to find a place where my concurrency upgrades give a boost
// 13.1 vs 18.5s. Luckily I found it. I would've died inside if this turned out to be slower with multiple streams.

// Next up is to try the wavefront iteration.

open System
open System.IO

#if INTERACTIVE
#load "spiral_q_v2.fsx"
#r "../packages/FSharp.Charting.0.90.14/lib/net40/FSharp.Charting.dll"
#r "System.Windows.Forms.DataVisualization.dll"
#load "embedded_reber.fsx"
#endif
open SpiralV3

open ManagedCuda
open ManagedCuda.CudaDNNv5

ctx.GetFreeDeviceMemorySize() |> ignore

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

let lstm_embedded_reber_train num_iters learning_rate (data: d4MUnion[]) (targets: d4MUnion[]) (data_v: d4MUnion[]) (targets_v: d4MUnion[]) clip_coef (lstm_layers: LSTMLayer[]) (l2: INNet) =
    [|
    let base_nodes = [|lstm_layers |> Array.collect (fun x -> x.ToArray);l2.ToArray|] |> Array.concat

    let training_loop (data: d4MUnion[]) (targets: d4MUnion[]) =
        let mutable ccc = 0
        let costs = [|
            let mutable ac =
                [|
                let mutable a,c = lstm_layers.[0].runLayerNoH data.[0]
                yield a,c
                for i=1 to lstm_layers.Length-1 do
                    let a',c' = lstm_layers.[i].runLayerNoH a
                    yield a',c'
                    a <- a'
                    c <- c'
                |]
            let b = l2.runLayer (ac |> Array.last |> fst)
            let r = squared_error_cost targets.[0] b
            yield r
    
            for i=1 to data.Length-2 do
                ac <-
                    [|
                    let pa,pc = ac.[0]
                    let mutable a,c = lstm_layers.[0].runLayer data.[0] pa pc
                    yield a,c
                    for i=1 to lstm_layers.Length-1 do
                        let pa,pc = ac.[i]
                        let a',c' = lstm_layers.[i].runLayer a pa pc
                        yield a',c'
                        a <- a'
                        c <- c'
                    |]
                let b = l2.runLayer (ac |> Array.last |> fst)
                let r = squared_error_cost targets.[i] b
                yield r
            |]
        printfn "ccc=%i" ccc
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

let hidden_size = 128

let l1 = LSTMLayer.createRandomLSTMLayer 7 hidden_size tanh_ tanh_
let l2 = LSTMLayer.createRandomLSTMLayer hidden_size hidden_size tanh_ tanh_
let l3 = LSTMLayer.createRandomLSTMLayer hidden_size hidden_size tanh_ tanh_
let l4 = LSTMLayer.createRandomLSTMLayer hidden_size hidden_size tanh_ tanh_
let l5 = LSTMLayer.createRandomLSTMLayer hidden_size hidden_size tanh_ tanh_
let l6 = FeedforwardLayer.createRandomLayer (hidden_size,7,1,1) clipped_sigmoid

let learning_rate = 15.0f

CudaContext.ProfilerStart()
// This iteration is to warm up the library.
lstm_embedded_reber_train 1 learning_rate d_training_data_20 d_target_data_20 d_training_data_validation d_target_data_validation 1.0f [|l1;l2;l3;l4;l5|] l6 |> ignore

let stopwatch = Diagnostics.Stopwatch.StartNew()
let s = [|
        yield lstm_embedded_reber_train 99 learning_rate d_training_data_20 d_target_data_20 d_training_data_validation d_target_data_validation 1.0f [|l1;l2;l3;l4;l5|] l6
        |] |> Array.concat
let time_elapsed = stopwatch.ElapsedMilliseconds
CudaContext.ProfilerStop()
// On the GTX 970, I get 3-4s depending on how hot the GPU is.
let l,r = Array.unzip s

//(Chart.Combine [|Chart.Line l;Chart.Line r|]).ShowChart()

printfn "Time Elapsed: %i" time_elapsed

