// As expected, nothing works correctly and I have to do it piece by piece again.

// Edit: Well, one part of was the incorrect inits that have been fixed in both libraries,
// but the fact that I cannot get the streams to align even with ctx.Synchronize() everywhere
// is really flooring me.

// Let me see if I can at least get a sequence of geams to execute correctly.

open System
open System.IO

#if INTERACTIVE
#load "spiral_q_v0.fsx"
#endif
open SpiralV3

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


let l1 = d4MUnion.createConstant((1024,1024,1,1))
let l2 = d4MUnion.createConstant((1024,1024,1,1))
let l3 = d4MUnion.createConstant((1024,1024,1,1))
let l4 = d4MUnion.createConstant((1024,1024,1,1))
let l5 = d4MUnion.createConstant((1024,1024,1,1))

l1.setPrimal 1.0f

tensor_add' true 0.0f l2 1.0f l1
tensor_add' true 0.0f l3 1.0f l2
tensor_add' true 0.0f l4 1.0f l3
tensor_add' true 0.0f l5 1.0f l4

l5.P.Gather() |> Array.sum