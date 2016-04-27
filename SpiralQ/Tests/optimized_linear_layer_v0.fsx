// Ok, so I finally did the optimized layer and it turns out that it is slower than the unoptimized one.

// How complicated. What is going on here is that the the atomic_saxp is resposible for the entirety of the
// difference in performance. The reason the 'unoptimized' version is is faster is because this problem I am
// testing it on - simple batched matrix multiplication - is more enough to saturate the entire GPU.

// When you add atomic_saxpy to the mix, it become slower. Shit.

// I did not see this coming. It casts that Optimizing RNNs Nvidia blog post in a different light.

// At the very least this explains why I am not getting much benefit from concurrency in the LSTM 
// Reber grammar example. The thing is simply too small.

// If I trained the same model on different length sequence sizes in parallel, I would start to see the benefit.
// Or maybe if I started off by calculating the inputs first, it would do better.

// Note: I've yet to test this for correctness.

// Edit2: Let me try compiling this to see if there any gaping holes in the timeline.

open System

#if INTERACTIVE
#load "spiral_q_v2.fsx"
#endif
open SpiralV3
open ManagedCuda
open ManagedCuda.BasicTypes
open ManagedCuda.VectorTypes

ctx.GetFreeDeviceMemorySize() |> ignore

/// Similar to binary caller except it assumes that the output will be modified atomically.
/// Does not do passing, only extraction.
let atomic_output_caller (a : d4MUnion) (b : d4MUnion) (f : CudaStream -> CudaEvent -> d4MUnion -> unit) = 
    match a.stream_state, b.stream_state with
    | Static, Static -> 
        let s,e,_ as sem = StreamPool.P
        sem
    | Unpassed (s,e,m), Static | Static, Unpassed (s,e,m) ->  
        s,e,m
    | Unpassed (s,e,m), Unpassed(s',e',m') -> 
        s.WaitEvent e'.Event
        s,e,m
    | Passed (s',e',m'), Unpassed (s,e,m) | Unpassed (s,e,m), Passed (s',e',m') ->  
        s.WaitEvent e'.Event
        s,e,m
    | Passed (s',e',m'), Passed (s'',e'',m'') ->
        let s,e,m as sem = StreamPool.P
        s.WaitEvent e'.Event
        s.WaitEvent e''.Event
        sem
    | Passed(s',e',m'), Static | Static, Passed(s',e',m') ->
        let s,e,_ as sem = StreamPool.P
        s.WaitEvent e'.Event
        sem
    |> fun (s,e,m) ->
        f s e m
        e.Record s.Stream
        e

/// o <- alpha*x+o (atomic addition)
type DeviceAtomicSaxpyModule() = 
    let block_size = 256

    let kernel_name = "AtomicSaxpyKernel"
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;

            // Device code
            __global__ void ";kernel_name;"(const floatType coef_A, const floatType* A, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    atomicAdd(&O[i], coef_A * A[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A(coef_x: float32, (x_nchw, x: CUdeviceptr), (o_nchw, o: CUdeviceptr), s) =
        if x_nchw <> o_nchw then failwith "x_nchw <> o_nchw in DeviceAtomicSaxpyModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(s, coef_x, x, o, n)

let atomicSaxpyModule = lazy new DeviceAtomicSaxpyModule()
let atomic_saxpy (alpha:float32) (x_nchw, x : CUdeviceptr as xv) (o_nchw, o:CUdeviceptr as ov) s =
    atomicSaxpyModule.Value.A(alpha,xv,ov,s)

        
// I'll skip optimizing the biases as I'll be using BN for any real task anyway.
/// The optimized linear matmult layer without biases.
let linear_layer_matmult_opt (mm : (d4MUnion*d4MUnion)[]) =
    if mm.Length > stream_num then failwith "Not enough streams allocated. Linear layers needs an appropriate amount of stream for temporary memory."
    let s,e,m as sem = StreamPool.P
    let mutable c = None

    for i=0 to mm.Length-1 do
        let l,r = mm.[0]
        atomic_output_caller l r
        <| fun s' e' m' ->
            let num_rows, _ = l.rc
            let _, num_cols = r.rc
            let dims = num_cols,num_rows,1,1
            m'.ReplaceIf dims

            match c with
            | None -> c <- Some <| ObjectPool.getd4M (false, dims)
            | Some c -> if c.nchw <> dims then failwithf "c.Value.nchw(%A) <> dims(%A)" c.nchw dims

            gemm nT nT 1.0f l.P' r.P' 0.0f m'.P' s'.Stream
            atomic_saxpy 1.0f m'.P' c.Value.P' s'.Stream
        |> fun e' -> s.WaitEvent e'.Event

    e.Record s.Stream
    
    match c with
    | Some c ->
        c.stream_state <- Unpassed sem    
        c
    | None -> failwith "The input array cannot be empty."

let optimized_lin_layer_test_v0() =
    let t =
        Array.init 500
        <| fun _ ->
           [|(d4MUnion.createConstant((256,256,1,1)),
              d4MUnion.createConstant((256,256,1,1)));
             (d4MUnion.createConstant((256,256,1,1)),
              d4MUnion.createConstant((256,256,1,1)));
             (d4MUnion.createConstant((256,256,1,1)),
              d4MUnion.createConstant((256,256,1,1)))|]

    let stopwatch = Diagnostics.Stopwatch.StartNew()
    for i= 1 to 10 do
        // So it is 0.762 (1 stream) vs 0.47 (32 streams.)
        t |> Array.map (fun x -> linear_layer_matmult_opt x |> sigmoid |> sigmoid |> sigmoid) |> ignore
        ObjectPool.ResetPointers()
        tape.Clear()
    ctx.Synchronize()
    stopwatch.ElapsedMilliseconds

//optimized_lin_layer_test_v0()

// So, it does not come out on top. What the hell?

// Edit: To make this sensible I need to block after every iteration otherwise it just starts on the fresh without waiting for the rest to finish.
// After this fix, the new linear_layer still does fairly poorly. It is only around 20% faster.

let optimized_lin_layer_test_v1() =
    let t =
        Array.init 1
        <| fun _ ->
           [|(d4MUnion.createConstant((256,256,1,1)),
              d4MUnion.createConstant((256,256,1,1)));
             (d4MUnion.createConstant((256,256,1,1)),
              d4MUnion.createConstant((256,256,1,1)));
             (d4MUnion.createConstant((256,256,1,1)),
              d4MUnion.createConstant((256,256,1,1)))|]

    let stopwatch = Diagnostics.Stopwatch.StartNew()
    for i= 1 to 10000 do
        // So it is 1.48 (non-opt) vs 1.2 (opt)
        t |> Array.map (fun x -> linear_layer_matmult_opt x) |> ignore
        ObjectPool.ResetPointers()
        tape.Clear()
        ctx.Synchronize()
    stopwatch.ElapsedMilliseconds

//CudaContext.ProfilerStart()
//optimized_lin_layer_test_v1() |> printfn "Time in millis is %i"
//optimized_lin_layer_test_v1() |> printfn "Time in millis is %i"
//CudaContext.ProfilerStop()

// In general the throughput of this is low.
// I am throwing in the towel on this type of optimized linear layer.

// Before I do that let me just see how fast the following example would be with only one stream.

let optimized_lin_layer_test_v2() =
    let t =
        Array.init 1
        <| fun _ ->
           [|(d4MUnion.createConstant((256,256,1,1)),
              d4MUnion.createConstant((256,256,1,1)));
             (d4MUnion.createConstant((256,256,1,1)),
              d4MUnion.createConstant((256,256,1,1)));
             (d4MUnion.createConstant((256,256,1,1)),
              d4MUnion.createConstant((256,256,1,1)))|]

    let stopwatch = Diagnostics.Stopwatch.StartNew()
    for i= 1 to 10000 do
        // So it is 1.14 (1 stream) vs 0.3s (32 streams)
        t |> Array.map (fun x -> linear_layer_matmult x None) |> ignore
        ObjectPool.ResetPointers() // Yes, I am aware this part would cause memory corruption in a realistic example.
        tape.Clear()
    ctx.Synchronize()
    stopwatch.ElapsedMilliseconds

//optimized_lin_layer_test_v2() |> printfn "Time in millis is %i"
//optimized_lin_layer_test_v2() |> printfn "Time in millis is %i"