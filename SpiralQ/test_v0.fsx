// As expected, nothing works correctly and I have to do it piece by piece again.

// Edit: Well, one part of was the incorrect inits that have been fixed in both libraries,
// but the fact that I cannot get the streams to align even with ctx.Synchronize() everywhere
// is really flooring me.

// Let me see if I can at least get a sequence of geams to execute correctly.

// Edit2: I really spent 3 hours just now chasing down non bugs. First the inits were wrong in _v3
// and then it turns out that cudaRandom changes its behavior on different streams.

// I think the library is basically working correctly, but I still need to do some more test.
// I want to see whether cudaRandom outputs the same thing when it switches streams and whether
// the things actually run correctly on the Mnist example now.

open System
open System.IO

#if INTERACTIVE
#load "spiral_q_v1.fsx"
#endif
open SpiralV3
open ManagedCuda

printfn "Free Memory=%i" (int <| ctx.GetFreeDeviceMemorySize())

let minibatch_size = 128
let load_mnist filename =
    use f = File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read)
    use d = new BinaryReader(f)

    let magicnumber = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
    match magicnumber with
    | 2049 -> // Labels
        let n = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        d.ReadBytes n
        |> Array.collect (
            fun x -> 
                let t = Array.zeroCreate 10
                t.[int x] <- 1.0f
                t)
        |> Array.chunkBySize (minibatch_size*10)
        |> Array.map (fun x -> ((x.Length/10,10,1,1),x) |> d4MUnion.createConstant )
    | 2051 -> // Images
        let n = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        let rows = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        let cols = d.ReadInt32() |> Net.IPAddress.NetworkToHostOrder
        d.ReadBytes(n * rows * cols)
        |> Array.map (fun x -> float32 x / 255.0f)
        |> Array.chunkBySize (minibatch_size*rows*cols)
        |> Array.map (fun x ->  ((x.Length/(rows*cols),1,rows,cols),x) |> d4MUnion.createConstant)
    | _ -> failwith "Given file is not in the MNIST format."

let [|test_images;test_labels;train_images;train_labels|] = 
    [|"t10k-images.idx3-ubyte";"t10k-labels.idx1-ubyte";"train-images.idx3-ubyte";"train-labels.idx1-ubyte"|]
    |> Array.map (fun x -> Path.Combine(__SOURCE_DIRECTORY__,x) |> load_mnist)

CudaContext.ProfilerStart()

let l1 = ConvolutionalFeedforwardLayer.createRandomLayer (128,1,5,5) relu
let l2 = ConvolutionalFeedforwardLayer.createRandomLayer (128,128,5,5) relu
let l3 = ConvolutionalFeedforwardLayer.createRandomLayer (128,128,5,5) relu
let l4 = ConvolutionalFeedforwardLayer.createRandomLayer (128,128,5,5) relu
let l5 = ConvolutionalFeedforwardLayer.createRandomLayer (10,128,4,4) clipped_sigmoid

let base_nodes = [|l1;l2;l3;l4;l5|] |> Array.collect (fun x -> x.ToArray)
//base_nodes |> Array.iter (fun x -> x.setPrimal 1.0f)

let training_loop label data = // For now, this is just checking if the new library can overfit on a single minibatch.
    [|
    defaultConvPar,l1
    defaultConvPar,l2
    {defaultConvPar with stride_h=2; stride_w=2},l3
    defaultConvPar,l4
    defaultConvPar,l5
    |] 
    |> Array.fold (fun x (convPars,layer) -> layer.runLayer (convPars,x)) data
    |> fun x -> get_accuracy label x, cross_entropy_cost label x

let learning_rate = 0.03f

let t =
    [|
    for i=1 to 10 do
        let r = 
            l1.runLayer (defaultConvPar, train_images.[0])
            |> fun x -> l2.runLayer (defaultConvPar, x)
            |> fun x -> l3.runLayer ({defaultConvPar with stride_h=2; stride_w=2}, x)
            |> fun x -> l4.runLayer (defaultConvPar, x)
            |> fun x -> l5.runLayer (defaultConvPar, x)
            |> fun x -> cross_entropy_cost train_labels.[0] x
        let er = r.P.Value.Value
        yield er
        
        backprop_tape base_nodes r (sgd learning_rate)
        |]

CudaContext.ProfilerStop()

printfn "%A" t

