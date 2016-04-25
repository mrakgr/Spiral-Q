open System
open System.IO

#if INTERACTIVE
#load "spiral_q_v2.fsx"
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


let l1 = FeedforwardLayer.createRandomLayer (784,2048,1,1) relu :> INNet
let l2 = FeedforwardLayer.createRandomLayer (2048,2048,1,1) relu :> INNet
let l3 = FeedforwardLayer.createRandomLayer (2048,2048,1,1) relu :> INNet
let l4 = FeedforwardLayer.createRandomLayer (2048,10,1,1) clipped_sigmoid :> INNet

//let l1 = load_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl1")) false |> fun x -> FeedforwardLayer.fromArray x relu
//let l2 = load_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl2")) false |> fun x -> FeedforwardLayer.fromArray x relu
//let l3 = load_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl3")) false |> fun x -> FeedforwardLayer.fromArray x relu
//let l4 = load_data (IO.Path.Combine(__SOURCE_DIRECTORY__,"weightsl4")) false |> fun x -> FeedforwardLayer.fromArray x clipped_sigmoid

let base_nodes = [|l1;l2;l3;l4|]
let base_d4M_nodes = base_nodes |> Array.map (fun x -> x.ToArray) |> Array.concat

let training_loop label data =
    base_nodes
    |> Array.fold (fun x layer -> layer.runLayer x) data
    |> fun x -> get_accuracy label x, cross_entropy_cost label x

let learning_rate = 0.03f

let test() =
    for i=1 to 10 do
        let mutable er = 0.0f
        for j=0 to train_images.Length-1 do
            let _,r = training_loop train_labels.[j] train_images.[j] // Forward step
            er <- er + r.P.Value.Value
            //printfn "CE cost on the minibatch is %f at batch %i" r.P.Value.Value j

            if r.P.Value.Value |> Single.IsNaN then failwith "Nan!"

            backprop_tape base_d4M_nodes r (sgd learning_rate)

        printfn "-----"
        printfn "Squared error cost on the dataset is %f at iteration %i" (er / float32 train_images.Length) i

        let mutable acc = 0.0f
        for j=0 to test_images.Length-1 do
            let acc',r = training_loop test_labels.[j] test_images.[j] // Forward step
            acc <- acc'.Value + acc
            tape.Clear()
            ObjectPool.ResetPointers()
    
        printfn "Accuracy on the test set is %i/10000." (int acc)
        printfn "-----"

#time
test()
#time
