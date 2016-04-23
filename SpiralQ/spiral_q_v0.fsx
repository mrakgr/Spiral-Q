// Basic reverse mode AD on the GPU. This v3 of Spiral is focused on the union type and automatic parallelization. Uses the cuDNN v5 RC.
// It has come a long way since v1.

module SpiralV3

#if INTERACTIVE
#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\managedCuda\CudaDNNv5\bin\Release\CudaDNNv5.dll"
#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\managedCuda\CudaBlas\bin\x64\Release\CudaBlas.dll"
#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\managedCuda\NVRTC\bin\x64\Release\NVRTC.dll"
#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\managedCuda\CudaRand\bin\x64\Release\CudaRand.dll"
#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\managedCuda\NPP\bin\x64\Release\NPP.dll"
#r @"C:\Users\Marko\Documents\Visual Studio 2015\Projects\managedCuda\ManagedCUDA\bin\Release\ManagedCuda.dll"
#endif

// Open up the namespaces.
open ManagedCuda
open ManagedCuda.BasicTypes
open ManagedCuda.VectorTypes
open ManagedCuda.CudaBlas
open ManagedCuda.CudaRand
open ManagedCuda.NVRTC
open ManagedCuda.CudaDNNv5

open System
open System.Collections.Generic
open System.Runtime.InteropServices

// Initialize the context. Analogous to a CPU process. Cuda tries to offload as much as possible during context creation so there aren't
// any unexpected delays later.
let ctx = new CudaContext()
let numSm = ctx.GetDeviceInfo().MultiProcessorCount // The number of streaming multiprocessors on the device.

type StreamPool(num) =
    let ar = Array.init num (fun _ -> new CudaStream(), new CudaEvent(CUEventFlags.DisableTiming))
    let mutable p = -1

    // Pops a Stream, Event tuple from the ring buffer.
    member t.P = 
        p <- p + 1
        let (s,e as t) = ar.[p % num]
        t

let StreamPool = new StreamPool(16) // 8 < 16 > 32 > 64 > 128 > 1024 in terms of performance.

// Set the Cuda libraries handles to the above stream.
let cublas = CudaBlas(PointerMode.Host,AtomicsMode.Allowed) // Better performance for some solver functions with atomics allowed. The Spiral library does not use them though.
let cudnn = new CudaDNNContext()
let cudaRandom = new CudaRand.CudaRandDevice(GeneratorType.PseudoDefault)

// I'll skip aliasing float32 to floatType for this iteration of the library. There is not point to it as Cuda native functions cannot be overloaded this way.

type unit_to_unit_delegate = delegate of unit -> unit
let add_callback_to_stream (str : CudaStream) (callback : unit -> unit) =
    let callb (str : CUstream) (res : CUResult) (p : nativeint) =
        let t : unit_to_unit_delegate = Runtime.InteropServices.Marshal.GetDelegateForFunctionPointer(p)
        t.Invoke()

    let aux = new unit_to_unit_delegate (callback)
    let ptr_to_aux = Marshal.GetFunctionPointerForDelegate aux

    let cuda_callback = CUstreamCallback(callb)
    str.AddCallback(cuda_callback,ptr_to_aux,CUStreamAddCallbackFlags.None)

// Helper functions
/// Copies a host array to device.
let inline to_dev (host_ar: 't []) =
    let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
    d_a.CopyToDevice(host_ar)
    d_a

/// Copies a device array to host.
let inline to_host (dev_ar: CudaDeviceVariable<'t>) =
    let h_a = Array.zeroCreate<'t> (int dev_ar.Size)
    dev_ar.CopyToHost(h_a)
    ctx.Synchronize()
    h_a

/// Copies the device array to host. Extends the CudaDeviceVariable class.
type CudaDeviceVariable<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> with
    member inline this.Gather() =
        to_host this

/// Allocates a new device array without initializing it.
let inline new_dev<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> (n: int) =
    new CudaDeviceVariable<'t>(SizeT n)

let inline size_nchw (n:int,c,h,w) = n*c*h*w
let inline add_nchw (n:int,c,h,w) = n+c+h+w

let inline ptr_to_device_variable nchw (x : CUdeviceptr) =
    new CudaDeviceVariable<float32>(x,false,sizeof<float32> * size_nchw nchw |> SizeT)

/// The float scalar type
type Df = 
    {
    P : Lazy<float32> ref // primal
    A : float32 ref // adjoint
    }

    static member inline create P =
        {P=ref (lazy P);A=ref 0.0f}


/// Wait for all the event in the occupancy array to finish.
let wait_on_event (s : CudaStream) (occupied_ar : ResizeArray<CudaEvent>) =
        occupied_ar |> Seq.iter (fun x -> s.WaitEvent x.Event)

/// Projects nchw dimensions to a row, column dimension according to the following formula:
/// row = c*h*w
/// column = n
let nchw_to_2d (n,c,h,w:int) = c*h*w, n

type DeadFlagType =
| Undefined
| Dead
| Alive

let module_nullary_stream_function x_occ (f : CUstream -> unit) =
    let s,e = StreamPool.P
    wait_on_event s x_occ
    
    f s.Stream

    e.Record s.Stream
    x_occ.Add e

let module_unary_stream_function x_occ o_occ (f : CUstream -> unit) =
    let s,e = StreamPool.P
    wait_on_event s x_occ
    wait_on_event s o_occ

    f s.Stream

    e.Record s.Stream
    o_occ.Add e

let module_binary_stream_function x_occ y_occ o_occ (f : CUstream -> unit) =
    let s,e = StreamPool.P
    wait_on_event s x_occ
    wait_on_event s y_occ
    wait_on_event s o_occ

    f s.Stream

    e.Record s.Stream
    o_occ.Add e

let module_trinary_stream_function x_occ y_occ z_occ o_occ (f : CUstream -> unit) =
    let s,e = StreamPool.P
    wait_on_event s x_occ
    wait_on_event s y_occ
    wait_on_event s z_occ
    wait_on_event s o_occ

    f s.Stream

    e.Record s.Stream
    o_occ.Add e

type d4MUnion =
    {
    mutable elements : ((int * int * int * int) * CUdeviceptr * CUdeviceptr option)[]
    mutable P: CudaDeviceVariable<float32> // primal
    mutable A: CudaDeviceVariable<float32> option // adjoint
    mutable is_dead : DeadFlagType // flag to skip backprop
    mutable primal_occupied : ResizeArray<CudaEvent>
    mutable adjoint_occupied : ResizeArray<CudaEvent>
    }  

    static member ar_scan (ar : (int * int * int * int)[]) = ar |> Array.scan (fun state e -> state + sizeof<float32> * size_nchw e) 0
    static member ar_fold (ar : (int * int * int * int)[]) = ar |> Array.fold (fun state e -> state + sizeof<float32> * size_nchw e) 0
    static member make_elements (p : CudaDeviceVariable<float32>) (a : CudaDeviceVariable<float32> option) (ar : (int * int * int * int)[]) (l : int[]) =
        [|
        for i=0 to l.Length-2 do
            let x = ar.[i]
            let size = l.[i+1] - l.[i]
            yield 
                x, 
                p.DevicePointer + SizeT l.[i], 
                match a with
                | Some a -> a.DevicePointer + SizeT l.[i] |> Some
                | None -> None
        |]

    static member create' (ar : (int * int * int * int)[], is_constant) =
        let l = d4MUnion.ar_scan ar
        let p,a = l |> Array.last |> fun x -> x / sizeof<float32> |> SizeT |> fun x -> new CudaDeviceVariable<float32>(x), if is_constant = false then new CudaDeviceVariable<float32>(x) |> Some else None
        let elements = d4MUnion.make_elements p a ar l
        
        {elements = elements; P=p; A=a; is_dead=Undefined; primal_occupied = ResizeArray(); adjoint_occupied = ResizeArray()}

    static member create' (ar_data : ((int * int * int * int) * float32[]) [], is_constant) =
        let ar, data = Array.unzip ar_data
        let t = d4MUnion.create' (ar, is_constant)
        for i=0 to data.Length-1 do
            let x = data.[i]
            t.elements.[i]
            |> fun (nchw,p,a) ->
                let size = size_nchw nchw
                if size <> x.Length then failwithf "size(%i) <> data.[%i].Length(%i)" size i x.Length
                ctx.CopyToDevice(p,x)
        t

    static member inline create (ar : (int * int * int * int)[]) = d4MUnion.create'(ar, false)
    static member inline create (ar_data : ((int * int * int * int) * float32[]) []) = d4MUnion.create'(ar_data, false)
    static member inline createConstant (ar : (int * int * int * int)[]) = d4MUnion.create'(ar, true)
    static member inline createConstant (ar_data : ((int * int * int * int) * float32[]) []) = d4MUnion.create'(ar_data, true)

    // Constructors for the singular d4MUnion records.
    static member inline create (ar : int * int * int * int) = d4MUnion.create'([|ar|], false)
    static member inline create (ar_data : (int * int * int * int) * float32[]) = d4MUnion.create'([|ar_data|], false)
    static member inline createConstant (ar : int * int * int * int) = d4MUnion.create'([|ar|], true)
    static member inline createConstant (ar_data : (int * int * int * int) * float32[]) = d4MUnion.create'([|ar_data|], true)

    /// Checks if the type is singular and then returns the primal along with its dimensions.
    member t.P' = 
        if t.elements.Length <> 1 then failwithf "t.elements.Length(%i) <> 1" t.elements.Length
        else
            t.elements.[0]
            |> fun (nchw,p,_) -> nchw,p,t.primal_occupied

    /// Checks if the type is singular and then returns the adjoint along with its dimensions.
    member t.A' = 
        if t.elements.Length <> 1 then failwithf "t.elements.Length(%i) <> 1" t.elements.Length
        else
            t.elements.[0]
            |> fun (nchw,_,a) -> nchw,a.Value,t.adjoint_occupied

    /// Checks if the type is singular and then returns its dimensions.
    member t.nchw = 
        if t.elements.Length <> 1 then failwithf "t.elements.Length(%i) <> 1" t.elements.Length
        else
            t.elements.[0]
            |> fun (nchw,_,_) -> nchw

    /// Checks if the type is singular and then returns its dimensions projected 
    /// to a 2D space according to the following formula:
    /// row = c*h*w
    /// column = n
    member t.rc = t.nchw |> nchw_to_2d

    /// Gets the slice by iteratively incrementing n if the other dimensions are equal.
    /// Does not do any layout transformation. The same as GetSliceAlongChannel.
    member t.GetSliceAlongImage (l,r) =
        let mutable (n,c,h,w),p,a = t.elements.[l]
        if l > r then failwith "l > r"
        for i=l+1 to r do
            t.elements.[i] 
            |> fun ((n',c',h',w'),_,_) -> 
                if c' <> c || h' <> h || w' <> w then failwithf "c'(%i) <> c(%i) || h'(%i) <> h(%i) || w'(%i) <> w(%i)" c' c h' h w' w
                n <- n + n'
        (n,c,h,w), p, a

    /// Gets the slice by iteratively incrementing c if the other dimensions are equal.
    /// Does not do any layout transformation. The same as GetSliceAlongImage.
    member t.GetSliceAlongChannel (l,r) =
        let mutable (n,c,h,w),p,a = t.elements.[l]
        if l > r then failwith "l > r"
        for i=l+1 to r do
            t.elements.[i] 
            |> fun ((n',c',h',w'),_,_) -> 
                if n' <> n || h' <> h || w' <> w then failwithf "n'(%i) <> n(%i) || h'(%i) <> h(%i) || w'(%i) <> w(%i)" n' n h' h w' w
                c <- c + c'
        (n,c,h,w), p, a

    /// Sets the adjoint to zero.
    member inline t.setZeroAdjoint() = 
        match t.A with
        | Some A -> 
            module_nullary_stream_function t.adjoint_occupied
            <| fun s -> A.MemsetAsync(0u,s)
        | None -> ()

    /// Sets the primal to zero.
    member inline t.setZeroPrimal() = 
        module_nullary_stream_function t.primal_occupied
        <| fun s -> t.P.MemsetAsync(0u,s)

    /// Set the matrix to a value.
    member inline t.setPrimal (x: float32) = 
        let v = BitConverter.ToUInt32(BitConverter.GetBytes(x),0)
        module_nullary_stream_function t.primal_occupied
        <| fun s -> t.P.MemsetAsync(v,s)

    member t.ReplaceIf (ar : (int * int * int * int)[]) =
        let l = d4MUnion.ar_scan ar
        let new_size = l |> Array.last
        if int t.P.SizeInBytes < new_size
        then
            (t :> IDisposable).Dispose()
            let t' = d4MUnion.create'(ar,t.A.IsNone)
            t.elements <- t'.elements
            t.P <- t'.P
            t.A <- t'.A
        else
            let p,a = t.P, t.A
            let elements = d4MUnion.make_elements p a ar l
            t.elements <- elements

    member t.ReplaceIf (ar : (int * int * int * int)) = t.ReplaceIf [|ar|]

    interface IDisposable with
        member t.Dispose() = 
            t.P.Dispose()
            match t.A with
            | Some A -> A.Dispose()
            | None -> ()

let gather_pointer (nchw, p) =
    let size = size_nchw nchw
    let t = Array.zeroCreate<float32> size
    ctx.CopyToHost(t,p)
    t

let T = Operation.Transpose
let nT = Operation.NonTranspose

let defaultLayout = cudnnTensorFormat.NCHW
let defaultType = cudnnDataType.Float
let defaultMaxPoolingNanOption = cudnnNanPropagation.PropagateNan
let defaultReluNanOption = cudnnNanPropagation.PropagateNan

type TensorDescriptor with
    /// Extended method that works according to the bound defaultLayout and defaultType variables.
    member inline t.SetTensor4dDescriptor(n,c,h,w) = t.SetTensor4dDescriptor(defaultLayout,defaultType,n,c,h,w)

type FilterDescriptor with
    /// Extended method that works according to the bound defaultType variable.
    member inline t.SetFilter4dDescriptor(n,c,h,w) = t.SetFilter4dDescriptor(defaultType,defaultLayout,n,c,h,w)

type ConvolutionParameters = {
    pad_h : int
    pad_w : int
    stride_h : int
    stride_w : int
    upscale_h : int
    upscale_w : int
    mode : cudnnConvolutionMode
    }

type PoolingParameters =
    {
    mode : cudnnPoolingMode
    windowHeight : int
    windowWidth : int
    verticalPadding : int
    horizontalPadding : int
    verticalStride : int
    horizontalStride : int
    }

type PoolingDescriptor with
    member inline t.SetPooling2dDescriptor (p : PoolingParameters) =
        t.SetPooling2dDescriptor(p.mode,defaultMaxPoolingNanOption,p.windowHeight,p.windowWidth,p.verticalPadding,p.horizontalPadding,p.verticalStride,p.horizontalStride)

    member inline t.GetPooling2dForwardOutputDim s =
        let mutable n,c,h,w = 0,0,0,0
        t.GetPooling2dForwardOutputDim(s,&n,&c,&h,&w)
        n,c,h,w

let defaultConvPar = 
    {
    pad_h = 0
    pad_w = 0
    stride_h = 1
    stride_w = 1
    upscale_h = 1
    upscale_w = 1
    mode = cudnnConvolutionMode.Convolution
    }

type ConvolutionDescriptor with
    member inline t.SetConvolution2dDescriptor (p : ConvolutionParameters) =
        t.SetConvolution2dDescriptor(p.pad_h,p.pad_w,p.stride_h,p.stride_w,p.upscale_h,p.upscale_w,p.mode, defaultType)
    member inline t.GetConvolution2dForwardOutputDim (s,f) =
        let mutable n,c,h,w = 0,0,0,0
        t.GetConvolution2dForwardOutputDim(s,f,&n,&c,&h,&w)
        n,c,h,w

let mutable inference_only_flag = false
/// The new object pool. Zeroes out the adjoints concurrently on the forward phase.
type ObjectPool() =
    let zeroer_str = new CudaStream()
    let d4MPool = ResizeArray()
    let d4Mp = ref 0
    let workspacePool = ResizeArray()
    let wp = ref 0

    let tensorDescriptorPool = Dictionary(HashIdentity.Structural)
    let filterDescriptorPool = Dictionary(HashIdentity.Structural)
    let convolutionDescriptorPool = Dictionary(HashIdentity.Structural)
    let poolingDescriptorPool = Dictionary(HashIdentity.Structural)
    let activationDescriptorPool = Dictionary(HashIdentity.Structural)
    let BNDescriptorPool = Dictionary(HashIdentity.Structural)

    static member inline private getFromPool (pool : ResizeArray<_>) (pointer_to_pool : int ref) (creation_function) =
        if pool.Count > !pointer_to_pool then
            let t = pool.[!pointer_to_pool]
            pointer_to_pool := !pointer_to_pool+1
            t
        else
            let t = creation_function()
            pool.Add(t)
            pointer_to_pool := !pointer_to_pool+1
            t

    static member inline private getFromDict (pool : Dictionary<_,_>) k creation_function set_function =
        match pool.TryGetValue k with
        | true, v -> v
        | false, _ ->
            let t = creation_function()
            set_function t k
            pool.Add(k, t)
            t

    member t.getWorkspace n = 
        if n > 0 then
            let t' = 
                ObjectPool.getFromPool workspacePool wp 
                <| (fun _ -> 
                    new_dev<byte> n)
            if int t'.Size < n then // Resize the object if less than n
                t'.Dispose()
                let t'' = new_dev<byte> n
                workspacePool.[!wp-1] <- t''
                t''
            else t'
        else CudaDeviceVariable.Null
    
    member t.getd4M (is_constant, (nchw : (int*int*int*int)[] as p)) =
        let t' = 
            match is_constant with
            | false -> ObjectPool.getFromPool d4MPool d4Mp (fun _ -> d4MUnion.create p)
            | true -> ObjectPool.getFromPool d4MPool d4Mp (fun _ -> d4MUnion.createConstant p)

        t'.ReplaceIf p
        t'.primal_occupied.Clear()
        t'.adjoint_occupied.Clear()
        t'.is_dead <- Undefined
        if inference_only_flag = false && t'.A.IsSome then t'.A.Value.MemsetAsync(0uy,zeroer_str.Stream)
        t'

    member inline t.getd4M (is_constant, (n:int,c:int,h:int,w:int as p)) = t.getd4M (is_constant, [|p|])

    member t.getTensorDescriptor (nchw : int*int*int*int) = 
        ObjectPool.getFromDict tensorDescriptorPool nchw (fun _ -> new TensorDescriptor()) (fun (t: TensorDescriptor) x -> x |> t.SetTensor4dDescriptor)
    member t.getFilterDescriptor (nchw : int*int*int*int) = 
        ObjectPool.getFromDict filterDescriptorPool nchw (fun _ -> new FilterDescriptor()) (fun (t: FilterDescriptor) x -> x |> t.SetFilter4dDescriptor)
    member t.getConvolutionDescriptor (convPars : ConvolutionParameters) = 
        ObjectPool.getFromDict convolutionDescriptorPool convPars (fun _ -> new ConvolutionDescriptor()) (fun (t: ConvolutionDescriptor) x -> x |> t.SetConvolution2dDescriptor)
    member t.getPoolingDescriptor (p : PoolingParameters) = 
        ObjectPool.getFromDict poolingDescriptorPool p (fun _ -> new PoolingDescriptor()) (fun (t: PoolingDescriptor) x -> x |> t.SetPooling2dDescriptor)
    member t.getActivationDescriptor (mode : cudnnActivationMode, nanopt : cudnnNanPropagation, reluCeiling as p) = 
        ObjectPool.getFromDict activationDescriptorPool p (fun _ -> new ActivationDescriptor()) (fun (t: ActivationDescriptor) x -> x |> t.SetActivationDescriptor)
    member t.getBNDescriptor (((nchw : int*int*int*int), (mode : cudnnBatchNormMode), srcDesc : TensorDescriptor) as p) = 
        ObjectPool.getFromDict BNDescriptorPool p 
            (fun _ -> new TensorDescriptor()) 
            (fun (t: TensorDescriptor) (nchw, mode, srcDesc) -> cudnn.DeriveBNTensorDescriptor(t,srcDesc,mode))

    /// Sets only the object pool pointers to zero.
    /// Unlike in V2 of Spiral, in this version the adjoints are set to zero during the forward phase.
    member t.ResetPointers() =
        d4Mp := 0
        wp := 0

    /// Resets the occupancy arrays of all the objects in the pool and sets the pointers to zero.
    /// Blocks the device and also, triggers .NET GC because why not?
    member t.ResetOccupancy() =
        for i=0 to d4MPool.Count-1 do
            d4MPool.[i].primal_occupied.Clear()
            d4MPool.[i].adjoint_occupied.Clear()
        t.ResetPointers()
        ctx.Synchronize()

let ObjectPool = new ObjectPool() // In the past iteration of the library, the object pool's role was taken by the tape. Not anymore.

let tape = new Stack<(unit -> unit)>(1000) // Nice and simple way of passing in the closures for the backprop step.

let backprop_tape (base_nodes : d4MUnion[]) (top : Df) (update : d4MUnion -> unit) =
    base_nodes |> Array.iter (fun x -> x.setZeroAdjoint())
    top.A := 1.0f
    ObjectPool.ResetOccupancy() // This step is a good one to make here.
    while tape.Count > 0 do
        tape.Pop()()
    base_nodes |> Array.iter update
    ObjectPool.ResetOccupancy()

let inline divup a b = (a-1)/b+1 // Integer division with rounding up. (a+b-1)/b is another variant on this.

let kernels_dir = IO.Path.Combine(__SOURCE_DIRECTORY__,"Cuda Kernels")
IO.Directory.CreateDirectory(kernels_dir) |> ignore // Creates the Cuda Kernels directory if it does not exist. WriteAllBytes would otherwise throw an exception.

let load_kernel kernel_code kernel_name = 
    let kernel_path = IO.Path.Combine(kernels_dir,kernel_name)
        
    if IO.File.Exists(kernel_path) 
    then
        ctx.LoadKernelPTX(kernel_path,kernel_name) // For all the modules, it takes roughly 0.35s to compile them. Loading them from drive takes less than a millisecond.
    else
        let k = new ManagedCuda.NVRTC.CudaRuntimeCompiler(kernel_code,kernel_name)
        try k.Compile([|"-arch=compute_30"|])
        with 
        | :? NVRTCException as x -> 
            printfn "%s" (k.GetLogAsString())
            reraise()
        let ptx = k.GetPTX()
        IO.File.WriteAllBytes(kernel_path,ptx)
        ctx.LoadKernelPTX(ptx,kernel_name)

// DeviceTransformModules could be all potentially made generic, but I do not want to take the risk without the type system helping me.
// I would need something like a type provider for Alea modules. I am curious as to how v3 of Alea will turn out.

/// o <- f(x)
type DeviceUnaryTransformModule(op: string, unique_name : string) = 
    let block_size = 256

    let kernel_name = "Map1Kernel"+unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType x)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType* A, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(A[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A((x_nchw, x: CUdeviceptr, x_occ), (o_nchw, o: CUdeviceptr, o_occ)) =
        if x_nchw <> o_nchw then failwith "x_nchw <> o_nchw in DeviceUnaryTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)

        module_unary_stream_function x_occ o_occ
        <| fun s -> kernel.RunAsync(s, x,o,n)

/// o <- f(x,y)
type DeviceBinaryTransformModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map2Kernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType x, floatType y)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType* A, const floatType* B, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(A[i],B[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""
    
    let kernel = load_kernel kernel_code kernel_name

    member t.A((x_nchw, x: CUdeviceptr, x_occ),(y_nchw, y: CUdeviceptr, y_occ), (o_nchw, o: CUdeviceptr, o_occ)) =
        if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in DeviceBinaryTransformModule"
        if y_nchw <> o_nchw then failwith "y_nchw <> o_nchw in DeviceBinaryTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        module_binary_stream_function x_occ y_occ o_occ
        <| fun s -> kernel.RunAsync(s, x,y,o,n)


/// o <- f(x,y,z)
type DeviceTrinaryTransformModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map3Kernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType x, floatType y, floatType z)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType* A, const floatType* B, const floatType* C, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(A[i],B[i],C[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A((x_nchw, x: CUdeviceptr, x_occ), (y_nchw, y: CUdeviceptr, y_occ), (z_nchw, z: CUdeviceptr, z_occ), (o_nchw, o: CUdeviceptr, o_occ)) =
        if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in DeviceTrinaryTransformModule"
        if y_nchw <> z_nchw then failwith "y_nchw <> z_nchw in DeviceTrinaryTransformModule"
        if z_nchw <> o_nchw then failwith "z_nchw <> o_nchw in DeviceTrinaryTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        module_trinary_stream_function x_occ y_occ z_occ o_occ
        <| fun s -> kernel.RunAsync(s, x,y,z,o,n)


/// o <- sum(f(x))
type DeviceUnaryMapSumModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map1SumKernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType x)
            {
                return ";op;"
            }
        
            __device__ inline floatType warpDownReduce(floatType value){
                #pragma unroll
	            for (int i = 16; i>0; i = i / 2) value += __shfl_down(value, i);
	            return value;
            }

            // Device code
            __global__ void ";kernel_name;"(const floatType* A, floatType* O, const int N)
            {
	            int i = blockDim.x * blockIdx.x + threadIdx.x;
	            const int stride = blockDim.x * gridDim.x;
	            __shared__ floatType temp[32];
                if (threadIdx.x < 32) {
                    temp[threadIdx.x] = 0.0f; 
                    if (blockIdx.x == 0) O[0] = 0.0f;
                    }
                
                floatType acc = 0.0f;
	            while (i < N)
	            {
		            acc += op(A[i]);
		            i += stride;
	            }
	            __syncthreads(); 
                floatType out_partial = warpDownReduce(acc);
	            if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
	            __syncthreads();
	            if (threadIdx.x < 32) out_partial = warpDownReduce(temp[threadIdx.x]);
	            if (threadIdx.x == 0) atomicAdd(O, out_partial);
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    let o = new_dev<float32> 1

    member t.A((x_nchw, x: CUdeviceptr, x_occ)) =
        let n = size_nchw x_nchw
        
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)

        let s,e = StreamPool.P
        wait_on_event s x_occ

        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(s.Stream, x,o.DevicePointer,n)
        lazy o.[SizeT 0]

/// o <- sum(f(x,y))
type DeviceBinaryMapSumModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map2SumKernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType x, floatType y)
            {
                return ";op;"
            }
        
            __device__ inline floatType warpDownReduce(floatType value){
                #pragma unroll
	            for (int i = 16; i>0; i = i / 2) value += __shfl_down(value, i);
	            return value;
            }

            // Device code
            __global__ void ";kernel_name;"(const floatType* A, const floatType* B, floatType* O, const int N)
            {
	            int i = blockDim.x * blockIdx.x + threadIdx.x;
	            const int stride = blockDim.x * gridDim.x;
	            __shared__ floatType temp[32]; 
                if (threadIdx.x < 32) {
                    temp[threadIdx.x] = 0.0f; 
                    if (blockIdx.x == 0) O[0] = 0.0f;
                    }    
                floatType acc = 0.0f;
	            while (i < N)
	            {
		            acc += op(A[i],B[i]);
		            i += stride;
	            }
	            __syncthreads(); 
                floatType out_partial = warpDownReduce(acc);
	            if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
	            __syncthreads();
	            if (threadIdx.x < 32) out_partial = warpDownReduce(temp[threadIdx.x]);
	            if (threadIdx.x == 0) atomicAdd(O, out_partial);
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    let o = new_dev<float32> 1

    member t.A((x_nchw, x: CUdeviceptr, x_occ),(y_nchw, y: CUdeviceptr, y_occ)) =
        if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in DeviceBinaryMapSumModule"
        let n = size_nchw x_nchw

        let s,e = StreamPool.P
        wait_on_event s x_occ
        wait_on_event s y_occ
        
        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        kernel.RunAsync(s.Stream, x,y,o.DevicePointer,n)
        lazy o.[SizeT 0]

/// o <- f(coef_x,x)
type DeviceUnaryCoefTransformModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map1CoefKernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType coef_x, floatType x)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType coef_A, const floatType* A, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(coef_A,A[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A(coef_x: float32, (x_nchw, x: CUdeviceptr, x_occ), (o_nchw, o: CUdeviceptr, o_occ)) =
        if x_nchw <> o_nchw then failwith "x.nchw <> o.nchw in DeviceUnaryCoefTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        module_unary_stream_function x_occ o_occ
        <| fun s -> kernel.RunAsync(s, coef_x,x,o,n)


/// o <- f(coef_x,x,coef_y,y)
type DeviceBinaryCoefTransformModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map2CoefKernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;

            __device__ inline floatType op(floatType coef_x, floatType x, floatType coef_y, floatType y)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType coef_A, const floatType* A, const floatType coef_B, const floatType* B, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(coef_A,A[i],coef_B,B[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A(coef_x: float32, (x_nchw, x: CUdeviceptr, x_occ), coef_y: float32, (y_nchw, y: CUdeviceptr, y_occ), (o_nchw, o: CUdeviceptr, o_occ)) =
        if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in DeviceBinaryCoefTransformModule"
        if y_nchw <> o_nchw then failwith "y_nchw <> o_nchw in DeviceBinaryCoefTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        module_binary_stream_function x_occ y_occ o_occ
        <| fun s -> kernel.RunAsync(s, coef_x,x,coef_y,y,o,n)


/// o <- f(coef_x,x,coef_y,y,coef_z,z)
type DeviceTrinaryCoefTransformModule(op: string, unique_name) = 
    let block_size = 256

    let kernel_name = "Map3CoefKernel" + unique_name
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            __device__ inline floatType op(floatType coef_x, floatType x, floatType coef_y, floatType y, floatType coef_z, floatType z)
            {
                return ";op;"
            }
        
            // Device code
            __global__ void ";kernel_name;"(const floatType coef_A, const floatType* A, const floatType coef_B, const floatType* B, const floatType coef_C, const floatType* C, floatType* O, const int N)
            {
                int i = blockDim.x * blockIdx.x + threadIdx.x;
                const int stride = blockDim.x * gridDim.x;
                while (i < N)
                {
                    O[i] = op(coef_A,A[i],coef_B,B[i],coef_C,C[i]);
                    i += stride;
                }
            }
        }

        " |] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A(coef_x: float32, (x_nchw, x: CUdeviceptr, x_occ), coef_y: float32, (y_nchw, y: CUdeviceptr, y_occ), coef_z: float32, (z_nchw, z: CUdeviceptr, z_occ), (o_nchw, o: CUdeviceptr, o_occ)) =
        if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in DeviceTrinaryCoefTransformModule"
        if y_nchw <> z_nchw then failwith "y_nchw <> z_nchw in DeviceTrinaryCoefTransformModule"
        if z_nchw <> o_nchw then failwith "z_nchw <> o_nchw in DeviceTrinaryCoefTransformModule"
        let n = size_nchw x_nchw

        let gridSize = min (2*numSm*(1024/block_size)) (divup n block_size)
        kernel.GridDimensions <- dim3(gridSize)
        kernel.BlockDimensions <- dim3(block_size)
        module_trinary_stream_function x_occ y_occ z_occ o_occ
        <| fun s -> kernel.RunAsync(s, coef_x,x,coef_y,y,coef_z,z,o,n)


/// o <- max_col(x)
/// Sets all except one of the max of a column to zero.
type DeviceMaxColumnActivationModule() = 
    let block_size = 128

    let kernel_name = "MaxColumnActivationKernel"
    let kernel_code = 
        [|"
        //Kernel code:
        extern \"C\" {
            typedef float floatType;
            #define INIT __int_as_float(0xff800000) // The constant init for the reduce operations. This is float negative infinity.
            // The max reduce version.
            __device__ inline floatType warpReduce(floatType value){
                #pragma unroll
	            for (int i=1; i<32; i*=2) {
                    floatType tmp = __shfl_xor(value, i);
                    value = (tmp > value) ? tmp : value;
                    }
	            return value;
            }
              
            __device__ inline floatType blockReduce(floatType value){
	            __shared__ floatType temp[32];
                if (threadIdx.x < 32) temp[threadIdx.x] = INIT; 
                floatType out_partial = warpReduce(value);
                __syncthreads();
	            if (threadIdx.x % 32 == 0) temp[threadIdx.x / 32] = out_partial;
                __syncthreads();
	            if (threadIdx.x < 32) out_partial = warpReduce(temp[threadIdx.x]);
                return out_partial;
            }

            // Device code
            __global__ void ";kernel_name;"(const floatType* A, floatType* O, const int num_rows, const int num_cols)
            {
                int row = threadIdx.x;
                //const int col = blockIdx.x;
                int col_idx = blockIdx.x*num_rows; 
                floatType max = INIT; // This is the negative infinity for floats.
                int index = -1;
                while (row < num_rows)
                {
                   if (A[row+col_idx] > max) {
                        max = A[row+col_idx];
                        index = row;
                        }
                    row += blockDim.x;
                }
                
                __shared__ floatType max_index;
                if (max == blockReduce(max)) max_index = index;
                __syncthreads();
                index = max_index; // These last four lines are to make absolutely sure that only one max is selected in case there is more than one.
                row = threadIdx.x;
                while (row < num_rows)
                {
                    O[row+col_idx] = (row == index) ? max : 0.0f;
                    row += blockDim.x;
                }
            }
        }

        "|] |> String.concat ""

    let kernel = load_kernel kernel_code kernel_name

    member t.A(((n : int,c : int,h,w as x_nchw), x: CUdeviceptr, x_occ), (o_nchw, o: CUdeviceptr, o_occ)) =
        if x_nchw <> o_nchw then failwith "x_nchw <> o_nchw"
        let m = c*h*w

        kernel.GridDimensions <- dim3(n)
        kernel.BlockDimensions <- dim3(block_size)

        module_unary_stream_function x_occ o_occ
        <| fun s -> kernel.RunAsync(s,x,o,m,n)


      
// The gradient clipping module.
let gradclipModule = lazy DeviceUnaryCoefTransformModule("(x < -coef_x) ? -coef_x : (x > coef_x ? coef_x : x);", "GradClip") // Unique names like GradClip are necessary for load and saving to drive. Be very careful of collisions.

// coef_x = scale
// coef_y = location
// y does not get used.
let randMapModule = lazy DeviceBinaryCoefTransformModule("coef_x*(x-0.5f)+coef_y;","RandMapper")

/// Fills matrix by sampling from a random uniform distribution in <-1.0f,1.0f]
let fillRandomUniformMatrix (x_nchw, x : CUdeviceptr, x_occ as x') (scaling_factor : float32) location =
    
//    let s,e = StreamPool.P
//    cudaRandom.SetStream s.Stream
//
//    wait_on_event s x_occ
//
    use x'' = ptr_to_device_variable x_nchw x
    cudaRandom.GenerateUniform(x'') // Uses the null stream because using others make the results non deterministic.

//    e.Record s.Stream
//    x_occ.Add e

    // 2.0f*scaling_factor ensures that it is rescaled around zero if the scaling_factor is 1.0f.
    randMapModule.Value.A(2.0f*scaling_factor,x',location,x',x')

let inline private cublas_unary_stream_function input_occ output_occ (f : unit -> unit) =
    let s,e = StreamPool.P
    cublas.Stream <- s.Stream
    
    wait_on_event s input_occ
    wait_on_event s output_occ

    f()

    e.Record s.Stream
    output_occ.Add e

let inline private cublas_binary_stream_function left_occ right_occ output_occ (f : unit -> unit) =
    let s,e = StreamPool.P
    cublas.Stream <- s.Stream

    wait_on_event s left_occ
    wait_on_event s right_occ
    wait_on_event s output_occ

    f()

    e.Record s.Stream
    output_occ.Add e

// y <- alpha * x + y
let saxpy (alpha:float32) (x_nchw, x : CUdeviceptr, x_occ) (y_nchw, y:CUdeviceptr, y_occ) =
    if x_nchw <> y_nchw then failwith "x_nchw <> y_nchw in saxpy"

    use x = ptr_to_device_variable x_nchw x
    use y = ptr_to_device_variable y_nchw y

    cublas_unary_stream_function x_occ y_occ
    <| fun _ -> cublas.Axpy(alpha,x,1,y,1)


/// General matrix-matrix addition. Inplace version.
/// The function is not indented for transposes due to dimensional confusion.
#nowarn "49"
let geam transa transb (alpha: float32) ((A_num_images, A_num_channels, A_num_rows, A_num_cols as A_nchw), A:CUdeviceptr, A_occ) beta ((B_num_images, B_num_channels, B_num_rows, B_num_cols as B_nchw), B:CUdeviceptr, B_occ) ((C_num_images, C_num_channels, C_num_rows, C_num_cols as C_nchw), C:CUdeviceptr, C_occ) =
    let inline geam (A_num_rows, A_num_cols) (B_num_rows, B_num_cols) (C_num_rows, C_num_cols) =
        let a_row = if transa = nT then A_num_rows else A_num_cols
        let a_col = if transa = nT then A_num_cols else A_num_rows
        let b_row = if transb = nT then B_num_rows else B_num_cols
        let b_col = if transb = nT then B_num_cols else B_num_rows
        
        if a_row <> b_row then failwith (sprintf "a_row <> b_row in geam! %i <> %i" a_row b_row)
        if a_col <> b_col then failwith (sprintf "a_col <> b_col in geam! %i <> %i" a_col b_col)

        if a_row <> C_num_rows then failwith (sprintf "a_row <> C_num_rows in geam! %i <> %i" a_col C_num_rows)
        if a_col <> C_num_cols then failwith (sprintf "a_col <> C_num_cols in geam! %i <> %i" a_col C_num_cols)

        let lda = if transa = nT then a_row else a_col
        let ldb = if transa = nT then b_row else b_col
        let ldc = a_row

        use A = new CudaDeviceVariable<float32>(A,false,sizeof<float32> * A_num_rows * A_num_cols |> SizeT)
        use B = new CudaDeviceVariable<float32>(B,false,sizeof<float32> * B_num_rows * B_num_cols |> SizeT)
        use C = new CudaDeviceVariable<float32>(C,false,sizeof<float32> * C_num_rows * C_num_cols |> SizeT)

        cublas_binary_stream_function A_occ B_occ C_occ
        <| fun _ -> cublas.Geam(transa, transb, a_row, a_col, alpha, A, lda, B, ldb, beta, C, ldc)

    geam (A_num_channels*A_num_cols*A_num_rows,A_num_images) (B_num_channels*B_num_cols*B_num_rows,B_num_images) (C_num_channels*C_num_cols*C_num_rows,C_num_images)

/// General matrix-matrix multiply from cuBLAS. Inplace version
/// c,h,w get multiplied together to form the first dimension. n is the second dimension.
let gemm transa transb (alpha: float32) ((A_num_images, A_num_channels, A_num_rows, A_num_cols), A:CUdeviceptr, A_occ) ((B_num_images, B_num_channels, B_num_rows, B_num_cols), B:CUdeviceptr, B_occ) beta ((C_num_images, C_num_channels, C_num_rows, C_num_cols), C:CUdeviceptr, C_occ) =
    let inline gemm (A_num_rows, A_num_cols) (B_num_rows, B_num_cols) (C_num_rows, C_num_cols) =
        let a_col = if transa = nT then A_num_cols else A_num_rows
        let b_row = if transb = nT then B_num_rows else B_num_cols
        if a_col <> b_row then failwithf "a_col(%i) <> b_row(%i) in gemm!" a_col b_row
        let m = if transa = nT then A_num_rows else A_num_cols
        let n = if transb = nT then B_num_cols else B_num_rows
        let k = a_col
        let lda = if transa = nT then m else k
        let ldb = if transb = nT then k else n
        let ldc = m

        if m <> C_num_rows || n <> C_num_cols then failwithf "m(%i) <> C_num_rows(%i) || n(%i) <> C_num_cols(%i)" m C_num_rows n C_num_cols

        use A = new CudaDeviceVariable<float32>(A,false,sizeof<float32> * A_num_rows * A_num_cols |> SizeT)
        use B = new CudaDeviceVariable<float32>(B,false,sizeof<float32> * B_num_rows * B_num_cols |> SizeT)
        use C = new CudaDeviceVariable<float32>(C,false,sizeof<float32> * C_num_rows * C_num_cols |> SizeT)

        cublas_binary_stream_function A_occ B_occ C_occ
        <| fun _ -> cublas.Gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)

    gemm (A_num_channels*A_num_cols*A_num_rows,A_num_images) (B_num_channels*B_num_cols*B_num_rows,B_num_images) (C_num_channels*C_num_cols*C_num_rows,C_num_images)

/// Does not only check, but also sets the undefined nodes to Dead or Alive.
let inline deadness_check (c : d4MUnion) (a : d4MUnion) (f : unit -> unit) =
    match c.is_dead with
    | Undefined -> failwith "The upper node should not be undefined."
    | Dead -> 
        match a.is_dead with
        | Undefined -> a.is_dead <- Dead
        | Dead | Alive -> () // If the bottom node is Alive do not modify it to be Dead as there exists a path from elsewhere through it.
    | Alive -> 
        f()
        a.is_dead <- Alive

/// Matrix-matrix multiply.
let inline private matmult' (prev_output : d4MUnion option) ((a,b): d4MUnion*d4MUnion) =
    let c = 
        match prev_output with
        | None ->
            let num_rows,_ = a.rc
            let _,num_cols = b.rc
            ObjectPool.getd4M (false, (num_cols,num_rows,1,1))
            |> fun c ->
                gemm nT nT 1.0f a.P' b.P' 0.0f c.P'
                c
        | Some c ->
            gemm nT nT 1.0f a.P' b.P' 1.0f c.P'
            c

    if a.A.IsSome then 
        let matmult_backward_left () = 
            deadness_check c a <| fun _ -> gemm nT T 1.0f c.A' b.P' 1.0f a.A'

        tape.Push(matmult_backward_left)
    if b.A.IsSome then 
        let matmult_backward_right () = 
            deadness_check c b <| fun _ -> gemm T nT 1.0f a.P' c.A' 1.0f b.A'
        tape.Push(matmult_backward_right)
    Some c

let matmult (a: d4MUnion) (b:d4MUnion) = matmult' None (a, b) |> fun x -> x.Value

let inline private cudnn_unary_stream_function input_occ output_occ (f : unit -> unit) =
    let s,e = StreamPool.P
    cudnn.SetStream(s)
    
    wait_on_event s input_occ
    wait_on_event s output_occ

    f()

    e.Record s.Stream
    output_occ.Add e

let inline private cudnn_binary_stream_function left_occ right_occ output_occ (f : unit -> unit) =
    let s,e = StreamPool.P
    cudnn.SetStream(s)
    
    wait_on_event s left_occ
    wait_on_event s right_occ
    wait_on_event s output_occ

    f()

    e.Record s.Stream
    output_occ.Add e

/// Can be used to add matrices or for (4D)matrix-vector broadcast addition.
/// The output dimensions are based on the left argument.
/// Those dimenions the size of 1 of the right argument are broadcasted.
let inline tensor_add' add_to_left alpha (left : d4MUnion) beta (right : d4MUnion) =
    let left_nchw = left.nchw
    let right_nchw = right.nchw
    let leftDesc = ObjectPool.getTensorDescriptor left_nchw
    let rightDesc = ObjectPool.getTensorDescriptor right_nchw

    let output = 
        if add_to_left = false 
        then 
            ObjectPool.getd4M (false, left_nchw) 
            |> fun output -> 
                geam nT nT 1.0f left.P' 0.0f output.P' output.P'
                output
        else 
            left

    if left_nchw <> right_nchw then
        cudnn_unary_stream_function right.primal_occupied output.primal_occupied 
        <| fun _ -> cudnn.AddTensor(beta,rightDesc,right.P,alpha,leftDesc,output.P) // Add right to output.
    else geam nT nT beta right.P' alpha output.P' output.P'

    if right.A.IsSome then 
        let tensor_add_right_backwards () =
            deadness_check output right
            <| fun _ ->
                if left_nchw = right_nchw then
                    saxpy beta output.A' right.A'
                else
                    cudnn_unary_stream_function output.adjoint_occupied right.adjoint_occupied
                    <| fun _ -> cudnn.ConvolutionBackwardBias(beta,leftDesc,output.A.Value,1.0f,rightDesc,right.A.Value)
        tape.Push(tensor_add_right_backwards)

    if add_to_left = false && left.A.IsSome then // No point in adding the adjoint to itself.
        let tensor_add_left_backwards () = 
            deadness_check output left <| fun _ -> saxpy alpha output.A' left.A'
        tape.Push(tensor_add_left_backwards)
    output

let tensor_add = tensor_add' false

let linear_layer_matmult (mm: (d4MUnion*d4MUnion) []) (bias: d4MUnion option) =
    mm
    |> Array.fold matmult' None
    |> fun (output) ->
        let left = output.Value
        match bias with
        | None -> left
        | Some right -> tensor_add' true 1.0f left 1.0f right

/// The activation function. Zeroes out the target primal during the call.
let activation_forward mode (input : d4MUnion)  =
    let input_sizes = input.nchw
    let srcTensorDesc = ObjectPool.getTensorDescriptor input_sizes

    let output = ObjectPool.getd4M (false, input_sizes)

    cudnn_unary_stream_function input.primal_occupied output.primal_occupied
    <| fun _ -> cudnn.ActivationForward(mode,1.0f,srcTensorDesc,input.P,0.0f,srcTensorDesc,output.P)

    if input.A.IsSome then 
        let activation_backward () =
            deadness_check output input 
            <| fun _ -> 
                cudnn_unary_stream_function output.adjoint_occupied input.adjoint_occupied
                <| fun _ -> cudnn.ActivationBackward(mode,1.0f,srcTensorDesc,output.P,srcTensorDesc,output.A.Value,srcTensorDesc,input.P,1.0f,srcTensorDesc,input.A.Value)
        tape.Push(activation_backward)
    output

/// The pooling function. Zeroes out the target primal during the call.
let pooling_forward p (input : d4MUnion) =
    let poolingDescriptor = ObjectPool.getPoolingDescriptor p
    let input_sizes = input.nchw
    let srcTensorDesc = ObjectPool.getTensorDescriptor input_sizes
    let dest_sizes = poolingDescriptor.GetPooling2dForwardOutputDim srcTensorDesc

    let output = ObjectPool.getd4M(false, dest_sizes)

    let dstTensorDesc = ObjectPool.getTensorDescriptor dest_sizes

    cudnn_unary_stream_function input.primal_occupied output.primal_occupied
    <| fun _ -> cudnn.PoolingForward(poolingDescriptor,1.0f,srcTensorDesc,input.P,0.0f,dstTensorDesc,output.P)

    if input.A.IsSome then 
        let pooling_backward () =
            deadness_check output input 
            <| fun _ ->
                cudnn_unary_stream_function output.adjoint_occupied input.adjoint_occupied
                <| fun _ -> cudnn.PoolingBackward(poolingDescriptor,1.0f,srcTensorDesc,output.P,srcTensorDesc,
                                      output.A.Value,dstTensorDesc,input.P,1.0f,dstTensorDesc,input.A.Value)
        tape.Push(pooling_backward)
    output


let inline private convolutional_forward' (prev_output: ((int*int*int*int)*d4MUnion) option) (convPar, data : d4MUnion, filter : d4MUnion) =
    let data_sizes = data.nchw
    let filter_sizes = filter.nchw

    let srcTensorDesc = ObjectPool.getTensorDescriptor data_sizes
    
    let filterDesc = ObjectPool.getFilterDescriptor filter_sizes
    let convDesc = ObjectPool.getConvolutionDescriptor convPar

    let dims, output = 
        let dims = convDesc.GetConvolution2dForwardOutputDim(srcTensorDesc,filterDesc)
        match prev_output with
        | Some (prev_dims, prev_output) ->
            if dims <> prev_dims then failwith "dims <> prev_dims in linear_layer_conv"
            prev_dims, prev_output
        | None ->
            dims, ObjectPool.getd4M (false, dims)

    let dstTensorDesc = ObjectPool.getTensorDescriptor dims

    let algo = 
        cudnn.GetConvolutionForwardAlgorithm(srcTensorDesc,filterDesc,convDesc,dstTensorDesc,cudnnConvolutionFwdPreference.SpecifyWorkspaceLimit,SizeT 30000)
    let workspace = 
        cudnn.GetConvolutionForwardWorkspaceSize(srcTensorDesc, filterDesc, convDesc, dstTensorDesc, algo) |> int
        |> ObjectPool.getWorkspace

    let beta = 
        match prev_output with
        | None -> 0.0f
        | Some _ -> 1.0f
    cudnn_binary_stream_function data.primal_occupied filter.primal_occupied output.primal_occupied
    <| fun _ -> cudnn.ConvolutionForward(1.0f,srcTensorDesc,data.P,filterDesc,filter.P,convDesc,algo,workspace,beta,dstTensorDesc,output.P) // Don't zero out the previous output.

    if filter.A.IsSome then 
        let convolution_backwards_filter () =
            let algo = 
                cudnn.GetConvolutionBackwardFilterAlgorithm(srcTensorDesc,dstTensorDesc,convDesc,filterDesc,cudnnConvolutionBwdFilterPreference.SpecifyWorkspaceLimit,SizeT 30000)
            let workspace =
                cudnn.GetConvolutionBackwardFilterWorkspaceSize(srcTensorDesc,dstTensorDesc,convDesc,filterDesc,algo) |> int
                |> ObjectPool.getWorkspace
            deadness_check output filter 
            <| fun _ ->
                cudnn_unary_stream_function output.adjoint_occupied filter.adjoint_occupied
                <| fun _ -> cudnn.ConvolutionBackwardFilter(1.0f,srcTensorDesc,data.P,dstTensorDesc,output.A.Value,convDesc,algo,workspace,1.0f,filterDesc,filter.A.Value)
        tape.Push(convolution_backwards_filter)

    if data.A.IsSome then 
        let convolution_backwards_data () =
            let algo = 
                cudnn.GetConvolutionBackwardDataAlgorithm(filterDesc,dstTensorDesc,convDesc,srcTensorDesc,cudnnConvolutionBwdDataPreference.SpecifyWorkspaceLimit,SizeT 30000)
            let workspace =
                cudnn.GetConvolutionBackwardDataWorkspaceSize(filterDesc,dstTensorDesc,convDesc,srcTensorDesc,algo) |> int
                |> ObjectPool.getWorkspace

            deadness_check output data 
            <| fun _ ->
                cudnn_unary_stream_function output.adjoint_occupied data.adjoint_occupied
                <| fun _ -> cudnn.ConvolutionBackwardData(1.0f,filterDesc,filter.P,dstTensorDesc,output.A.Value,convDesc,1.0f,algo,workspace,srcTensorDesc,data.A.Value)
        tape.Push(convolution_backwards_data)

    (dims,output) |> Some

/// The convolutional function. Zeroes out the target primal during the call.
let convolution_forward convPar (data : d4MUnion) (filter : d4MUnion) = 
    convolutional_forward' None (convPar,data,filter)
    |> fun x -> x.Value |> snd

let batch_normalization_forward bnMode (bnScale : d4MUnion) (bnBias : d4MUnion) (bnRunningMean : d4MUnion) (bnRunningVariance : d4MUnion) exponentialAverageFactor do_inference (input : d4MUnion) =
    let input_sizes = input.nchw
    let bias_sizes = bnBias.nchw
    let srcTensorDesc = ObjectPool.getTensorDescriptor input_sizes

    let bnDesc = 
        ObjectPool.getBNDescriptor (input_sizes, bnMode, srcTensorDesc)

    let _ =
        let mutable d,n,c,h,w,sn,sc,sh,sw = cudnnDataType.Double,0,0,0,0,0,0,0,0
        bnDesc.GetTensor4dDescriptor(&d,&n,&c,&h,&w,&sn,&sc,&sh,&sw)
        let bn_nchw = n,c,h,w
        if bn_nchw <> bnScale.nchw then failwith "Tensor dimensions for bnScale are incorrect."
        if bn_nchw <> bnBias.nchw then failwith "Tensor dimensions for bnBias are incorrect."
        if bn_nchw <> bnRunningMean.nchw then failwith "Tensor dimensions for bnRunningMean are incorrect."
        if bn_nchw <> bnRunningVariance.nchw then failwith "Tensor dimensions for bnRunningVariance are incorrect."

    let alpha, beta = 1.0f, 0.0f
    let epsilon = 1e-5
    let bnSavedMean = ObjectPool.getd4M (true,bias_sizes)
    let bnSavedVariance = ObjectPool.getd4M (true,bias_sizes)
    let output = ObjectPool.getd4M (false,input_sizes)

    if do_inference = false then
        cudnn_unary_stream_function input.primal_occupied output.primal_occupied
        <| fun _ -> cudnn.BatchNormalizationForwardTraining(bnMode,alpha,beta,srcTensorDesc,input.P,srcTensorDesc,output.P,bnDesc,bnScale.P,bnBias.P,exponentialAverageFactor,bnRunningMean.P,bnRunningVariance.P,epsilon,bnSavedMean.P,bnSavedVariance.P)
        if input.A.IsSome then 
            let batch_normalization_backward () =
                let dx_alpha, dx_beta = 1.0f, 1.0f
                let param_alpha, param_beta = 1.0f, 1.0f

                deadness_check output input 
                <| fun _ ->
                    cudnn_unary_stream_function output.adjoint_occupied input.adjoint_occupied
                    <| fun _ -> cudnn.BatchNormalizationBackward(bnMode,dx_alpha,dx_beta,param_alpha,param_beta,srcTensorDesc,input.P,srcTensorDesc,output.A.Value,srcTensorDesc,input.A.Value,bnDesc,bnScale.P,bnScale.A.Value,bnBias.A.Value,epsilon,bnSavedMean.P,bnSavedVariance.P)
                
            tape.Push batch_normalization_backward
    else
            cudnn_unary_stream_function input.primal_occupied output.primal_occupied
            <| fun _ -> cudnn.BatchNormalizationForwardInference(bnMode,alpha,beta,srcTensorDesc,input.P,srcTensorDesc,output.P,bnDesc,bnScale.P,bnBias.P,bnRunningMean.P,bnRunningVariance.P, epsilon)
        
    output
    
let linear_layer_conv (convs: (ConvolutionParameters*d4MUnion*d4MUnion) []) (bias: d4MUnion option) =
    convs
    |> Array.fold convolutional_forward' None
    |> fun (output) ->
        let _, left = output.Value
        match bias with
        | None -> left
        | Some right -> tensor_add' true 1.0f left 1.0f right

let hadamaradMultiplicationModule = lazy new DeviceBinaryTransformModule("x*y;", "HadMult")
let hadamaradMultiplicationErrorModule = lazy new DeviceTrinaryTransformModule("x*y+z;", "HadMultError")
/// Hadamarad (elementwise) multiplication function.
let inline private hadmult' (prev_output : d4MUnion option) ((a,b): d4MUnion*d4MUnion) =
    let c = 
        match prev_output with
        | Some c -> 
            hadamaradMultiplicationErrorModule.Value.A(a.P', b.P', c.P', c.P'); c
        | None -> 
            ObjectPool.getd4M (false, a.nchw)
            |> fun c -> hadamaradMultiplicationModule.Value.A(a.P', b.P', c.P'); c

    if a.A.IsSome then 
        let hadmult_backward_left () = 
            deadness_check c a 
            <| fun _ ->
                hadamaradMultiplicationErrorModule.Value.A(b.P',c.A',a.A',a.A')
        tape.Push hadmult_backward_left
    if b.A.IsSome then 
        let hadmult_backward_right () = 
            deadness_check c b 
            <| fun _ ->
                hadamaradMultiplicationErrorModule.Value.A(a.P',c.A',b.A',b.A')
        tape.Push hadmult_backward_right
    Some c

let hadmult (a: d4MUnion) (b: d4MUnion) = hadmult' None (a, b) |> fun x -> x.Value
let linear_layer_hadmult (hads: (d4MUnion*d4MUnion)[]) = hads |> Array.fold hadmult' None |> fun x -> x.Value

let squareModule = lazy new DeviceUnaryTransformModule("x*x;","Square")
//y = error
//z = previous adjoint value
let squareErrorModule = lazy new DeviceTrinaryTransformModule("2.0f*x*y + z;","SquareError")
let square (a:d4MUnion) =
    let c = ObjectPool.getd4M (false,a.nchw)
    squareModule.Value.A(a.P',c.P')

    if a.A.IsSome then 
        let square_backward () = 
            deadness_check c a 
            <| fun _ -> squareErrorModule.Value.A(a.P',c.A',a.A',a.A')
        tape.Push square_backward
    c

/// This one is for debugging currently
let squareSumModule = lazy new DeviceUnaryMapSumModule("x*x;", "SquareSum")

let sumModule = lazy new DeviceUnaryMapSumModule("x;", "Sum")
let sumErrorModule = lazy new DeviceUnaryCoefTransformModule("coef_x + x;", "SumError")
let sum (a:d4MUnion) =
    let c = Df.create 0.0f
    c.P := sumModule.Value.A(a.P')

    if a.A.IsSome then 
        let sum_backward () = 
            if !c.A <> 0.0f then 
                a.is_dead <- Alive
                sumErrorModule.Value.A(!c.A,a.A',a.A')
            else a.is_dead <- Dead
        tape.Push sum_backward
    c

let scale (alpha: float32) (a:Df) =
    let c = Df.create 0.0f
    c.P := lazy (alpha * a.P.Value.Value)

    let scale_backward () = a.A := alpha * !c.A + !a.A
    tape.Push scale_backward
    c

let sum_scalars (a:Df[]) =
    let c = Df.create 0.0f

    for l in a do c.P := lazy (c.P.Value.Value + l.P.Value.Value)
    
    let sum_scalars_backwards () = for l in a do l.A := !c.A + !l.A
    tape.Push sum_scalars_backwards
    c

let logModule = lazy new DeviceUnaryTransformModule("logf(x);","Log")
//y=error
//z=previous adjoint
let logErrorModule = lazy new DeviceTrinaryTransformModule("y / x + z;","LogError")
let log_ (a:d4MUnion) =
    let c = ObjectPool.getd4M (false, a.nchw)

    logModule.Value.A(a.P',c.P')

    if a.A.IsSome then
        let log_backward () = 
            deadness_check c a 
            <| fun _ -> logErrorModule.Value.A(a.P',c.A', a.A', a.A')
        tape.Push log_backward
    c

//coef_x = scalar
//coef_y = coef
let scalarMatrixAddModule = lazy new DeviceBinaryCoefTransformModule("coef_x + coef_y*x;","ScalarMatrixAdd")
/// o <- scalar + coef*a
let scalar_matrix_add scalar coef (a:d4MUnion) =
    let c = ObjectPool.getd4M (false, a.nchw)

    scalarMatrixAddModule.Value.A(scalar,a.P',coef,a.P',c.P')

    if a.A.IsSome then
        let scalar_matrix_add_backward () = 
            deadness_check c a
            <| fun _ -> saxpy coef c.A' a.A'
        tape.Push scalar_matrix_add_backward
    c


let add alpha (a: d4MUnion) beta (b: d4MUnion) =
    let c = ObjectPool.getd4M (false, a.nchw)

    geam nT nT alpha a.P' beta b.P' c.P'

    if a.A.IsSome then
        let add_backward_left () = saxpy alpha c.A' a.A'
        tape.Push add_backward_left
    if b.A.IsSome then
        let add_backward_right () =  saxpy beta c.A' b.A'
        tape.Push add_backward_right
    c

let softmax_instance_forward (data : d4MUnion) =
    let data_sizes = data.nchw

    let srcTensorDesc = ObjectPool.getTensorDescriptor data_sizes
    let output =  ObjectPool.getd4M (false, data_sizes)

    let algo = cudnnSoftmaxAlgorithm.Accurate // Log mode forgets to re-exponentiate at the end.
    let mode = cudnnSoftmaxMode.Instance

    cudnn.SoftmaxForward(algo,mode,1.0f,srcTensorDesc,data.P,0.0f,srcTensorDesc,output.P)

    if data.A.IsSome then
        let softmax_channel_backward () =
            cudnn.SoftmaxBackward(algo,mode,1.0f,srcTensorDesc,output.P,srcTensorDesc,output.A.Value,1.0f,srcTensorDesc,data.A.Value)
        tape.Push softmax_channel_backward
    output

let inline softmax x = softmax_instance_forward x

let clipModule = lazy new DeviceTrinaryCoefTransformModule("((x < coef_x) ? coef_x : (x > coef_y ? coef_y : x))+coef_z;","Clip")
let clipErrorModule = lazy new DeviceTrinaryCoefTransformModule("y*((x < coef_x) ? 0.0f : (x > coef_y ? 0.0f : 1.0f))+z;","ClipError")
/// o <- clip(min,max,a)+scalar
/// The clip function. Can be used as Relu by setting max to positive infinity. 
/// Can be used to make linear clipped sigmoid by setting min,max,scalar to -0.5f,0.5f,0.5f.
let clip min max (a : d4MUnion) scalar =
    let c = ObjectPool.getd4M (false, a.nchw)

    clipModule.Value.A(min,a.P',max,a.P',scalar,a.P',c.P')

    if a.A.IsSome then
        let clip_backward () = 
            deadness_check c a
            <| fun _ -> clipErrorModule.Value.A(min,a.P',max,c.A',max,a.A',a.A')
        tape.Push clip_backward
    c

let inline relu x = 
    let t = ObjectPool.getActivationDescriptor (cudnnActivationMode.Relu, defaultReluNanOption, 0.0)
    activation_forward t x
let inline sigmoid x = 
    let t = ObjectPool.getActivationDescriptor (cudnnActivationMode.Sigmoid, defaultReluNanOption, 0.0)
    activation_forward t x
let inline tanh_ x = 
    let t = ObjectPool.getActivationDescriptor (cudnnActivationMode.Tanh, defaultReluNanOption, 0.0)
    activation_forward t x
let inline clipped_sigmoid x = clip 0.0001f 0.9999f (sigmoid x) 0.0f
let inline clipped_softmax x = clip 0.0001f 0.9999f (softmax x) 0.0f

let squared_error_cost target activations =
    add 1.0f target -1.0f activations // TODO: tensor_add is ungodly slow in v3. Make v4 wrapper.
    |> square
    |> sum
    |> scale (0.5f/ float32 (target.nchw |> fun (num_images,_,_,_) -> num_images))

let cross_entropy_cost target activations =
    linear_layer_hadmult [|target,log_ activations;scalar_matrix_add 1.0f -1.0f target, log_ (scalar_matrix_add 1.0f -1.0f activations)|]
    |> sum
    |> scale (-1.0f/float32 (target.nchw |> fun (num_images,_,_,_) -> num_images))


let maxColumnActivationModule = lazy new DeviceMaxColumnActivationModule()
let accuracyModule = lazy new DeviceBinaryMapSumModule("(x*y == 0.0f) ? 0.0f : 1.0f;","Accuracy")
let get_accuracy (targets : d4MUnion) (activations : d4MUnion) =
    let o = ObjectPool.getd4M (true, targets.nchw)
    maxColumnActivationModule.Value.A(activations.P',o.P')
    accuracyModule.Value.A(targets.P',o.P')

let find_max_index (action_values : float32[]) =
    let mutable max = Single.NegativeInfinity
    let mutable index = -1
    for i=0 to action_values.Length-1 do
        let x = action_values.[i]
        if max < x then max <- x; index <- i
    index

type d4MUnion with
    static member makeUniformRandomNode (n,c,h,w as nchw) =
        let scale = (1.0f / sqrt(add_nchw nchw |> float32))
        let p = d4MUnion.create((n,c,h,w))
        fillRandomUniformMatrix p.P' scale 0.0f
        p

// A convolutional feedforward layer of neurons
type ConvolutionalFeedforwardLayer =
    {
    W : d4MUnion  // Input weight matrix
    b : d4MUnion  // Bias vector
    a : d4MUnion -> d4MUnion // Activation function
    }      
     
    static member fromArray (a : d4MUnion[]) act =
        {
         W = a.[0]
         b = a.[1]
         a = act
        }

    static member createRandomLayer (n,c,h,w as nchw) act =
        {
         W = d4MUnion.makeUniformRandomNode nchw
         b = d4MUnion.makeUniformRandomNode (1,n,1,1)
         a = act
        } 

    member l.runLayer (convPars,x:d4MUnion) =
        linear_layer_conv [|convPars,x,l.W|] (Some l.b)
        |> l.a

    member l.ToArray = [|l.W;l.b|]
    member t.ResetAdjoints () = t.W.setZeroAdjoint(); t.b.setZeroAdjoint()
    member t.SGD learning_rate = saxpy -learning_rate t.W.A' t.W.P'; saxpy -learning_rate t.b.A' t.b.P'


type INNet =
      abstract member ResetAdjoints : unit -> unit
      abstract member SGD : learning_rate:float32 -> unit
      abstract member ToArray : d4MUnion []
      abstract member inference : x:d4MUnion -> d4MUnion
      abstract member runLayer : x:d4MUnion -> d4MUnion
      abstract member train : x:d4MUnion -> (unit -> float) -> d4MUnion

// A fully connected feedforward layer of neurons
//type FullyConnectedLayer = FeedforwardLayer
type FeedforwardLayer =
    {
    W : d4MUnion  // Input weight matrix
    b : d4MUnion  // Bias vector
    a : d4MUnion -> d4MUnion
    } with     // Activation function

    static member fromArray (a : d4MUnion[]) act =
        {
         W = a.[0]
         b = a.[1]
         a = act
        }

    static member createRandomLayer (n,c,h,w as nchw) act =
        {
         W = d4MUnion.makeUniformRandomNode nchw
         b = d4MUnion.makeUniformRandomNode (1,c,1,1)
         a = act
        } 

    static member inline create = FeedforwardLayer.createRandomLayer

    interface INNet with
        member l.runLayer (x:d4MUnion) =
            linear_layer_matmult [|l.W,x|] (Some l.b) |> l.a

        /// This second attribute is supposed to be the exponential factor from the BN layer, but it is not used here.
        member l.train (x: d4MUnion) _ = (l :> INNet).runLayer x

        member l.inference (x: d4MUnion) = (l :> INNet).runLayer x

        member l.ToArray = [|l.W;l.b|]
        member t.ResetAdjoints () = t.W.setZeroAdjoint(); t.b.setZeroAdjoint()
        member t.SGD learning_rate = saxpy -learning_rate t.W.A' t.W.P'; saxpy -learning_rate t.b.A' t.b.P'

type ResidualFeedforwardLayer =
    {
    W1 : d4MUnion  // Input weight matrix
    b1 : d4MUnion  // Bias vector
    a1 : d4MUnion -> d4MUnion
    W2 : d4MUnion  // Input weight matrix
    b2 : d4MUnion  // Bias vector
    a2 : d4MUnion -> d4MUnion
    } with     // Activation function
     
    static member fromArray (a : d4MUnion[]) act1 act2 =
        {
         W1 = a.[0]
         b1 = a.[1]
         a1 = act1
         W2 = a.[2]
         b2 = a.[3]
         a2 = act2
        }

    static member createRandomLayer (n,c,h,w as nchw) act1 act2 =
        {
         W1 = d4MUnion.makeUniformRandomNode nchw
         b1 = d4MUnion.makeUniformRandomNode (1,c,1,1)
         a1 = act1
         W2 = d4MUnion.makeUniformRandomNode nchw
         b2 = d4MUnion.makeUniformRandomNode (1,c,1,1)
         a2 = act2
        } 

    static member inline create = ResidualFeedforwardLayer.createRandomLayer

    interface INNet with
        member l.runLayer (x:d4MUnion) =
            linear_layer_matmult [|l.W1,x|] (Some l.b1) |> l.a1
            |> fun p -> linear_layer_matmult [|l.W2,p|] (Some l.b2)
            |> fun p -> add 1.0f p 1.0f x |> l.a2
        
        /// This second attribute is supposed to be the exponential factor from the BN layer, but it is not used here.
        member l.train (x: d4MUnion) _ = (l :> INNet).runLayer x

        member l.inference (x: d4MUnion) = (l :> INNet).runLayer x

        member l.ToArray = [|l.W1;l.b1;l.W2;l.b2|]
        member t.ResetAdjoints () = t.W1.setZeroAdjoint(); t.b1.setZeroAdjoint(); t.W2.setZeroAdjoint(); t.b2.setZeroAdjoint()
        member t.SGD learning_rate = 
            saxpy -learning_rate t.W1.A' t.W1.P'; saxpy -learning_rate t.b1.A' t.b1.P'
            saxpy -learning_rate t.W2.A' t.W2.P'; saxpy -learning_rate t.b2.A' t.b2.P'


/// The initialization parameter for this is based off the weights and not the constructor.
/// Can be used for feedforward nets assuming the last two dimensions are 1. Uses the spatial normalization mode.
type BNConvolutionalLayer =
    {
    W : d4MUnion  // Input weight tensor
    bnScale : d4MUnion  // Scale tensor
    bnBias : d4MUnion  // Bias tensor
    bnRunningMean : d4MUnion  // Mean tensor
    bnRunningVariance : d4MUnion  // Variance tensor
    a : d4MUnion -> d4MUnion // Activation function
    }      

    static member create weight_nchw a =
        let W = d4MUnion.makeUniformRandomNode weight_nchw
        let bias_sizes = weight_nchw |> fun (n,c,h,w) -> (1,n,1,1)

        let bnScale = bias_sizes |> d4MUnion.create 
        bnScale.setPrimal 0.1f // Initial scale value based on the Recurrent Batch Normalization paper by Cooijmans et al.
        bnScale.setZeroAdjoint()
        let bnBias = bias_sizes |> d4MUnion.create
        bnBias.setZeroPrimal()
        bnBias.setZeroAdjoint()
        let bnRunningMean = bias_sizes |> d4MUnion.createConstant
        let bnRunningVariance = bias_sizes |> d4MUnion.createConstant

        { W = W; bnScale = bnScale; bnBias = bnBias; bnRunningMean = bnRunningMean; bnRunningVariance = bnRunningVariance; a=a  }

    member t.train (convPars,input:d4MUnion) exponentialAverageFactor = 
        let bnMode = cudnnBatchNormMode.BatchNormSpatial
        convolution_forward convPars input t.W
        |> batch_normalization_forward bnMode t.bnScale t.bnBias t.bnRunningMean t.bnRunningVariance exponentialAverageFactor false
        |> t.a
    member t.inference (convPars,input:d4MUnion) = 
        let bnMode = cudnnBatchNormMode.BatchNormSpatial
        convolution_forward convPars input t.W
        |> batch_normalization_forward bnMode t.bnScale t.bnBias t.bnRunningMean t.bnRunningVariance 1.0 true
        |> t.a

    member l.ToArray = [|l.W;l.bnScale;l.bnBias;l.bnRunningMean;l.bnRunningVariance|]
    member l.ResetAdjoints () = 
        l.W.setZeroAdjoint();l.bnScale.setZeroAdjoint();
        l.bnBias.setZeroAdjoint()
    member t.SGD learning_rate = 
        saxpy -learning_rate t.W.A' t.W.P'
        saxpy -learning_rate t.bnScale.A' t.bnScale.P'
        saxpy -learning_rate t.bnBias.A' t.bnBias.P'

type BNFullyConnectedLayer =
    {
    W : d4MUnion  // Input weight tensor
    bnScale : d4MUnion  // Scale tensor
    bnBias : d4MUnion  // Bias tensor
    bnRunningMean : d4MUnion  // Mean tensor
    bnRunningVariance : d4MUnion  // Variance tensor
    a : d4MUnion -> d4MUnion // Activation function
    }      

    /// Creates a layer with random weights.
    static member create weight_nchw a =
        let W = d4MUnion.makeUniformRandomNode weight_nchw
        let bias_sizes = weight_nchw |> fun (n,c,h,w) -> (1,c,1,1)

        let bnScale = bias_sizes |> d4MUnion.create 
        bnScale.setPrimal 0.1f // Initial scale value based on the Recurrent Batch Normalization paper by Cooijmans et al.
        bnScale.setZeroAdjoint()
        let bnBias = bias_sizes |> d4MUnion.create
        bnBias.setZeroPrimal()
        bnBias.setZeroAdjoint()
        let bnRunningMean = bias_sizes |> d4MUnion.createConstant
        let bnRunningVariance = bias_sizes |> d4MUnion.createConstant

        { W = W; bnScale = bnScale; bnBias = bnBias; bnRunningMean = bnRunningMean; bnRunningVariance = bnRunningVariance; a=a  }

    interface INNet with
        member t.train input exponentialAverageFactor = 
            let bnMode = cudnnBatchNormMode.BatchNormSpatial
            matmult t.W input
            |> batch_normalization_forward bnMode t.bnScale t.bnBias t.bnRunningMean t.bnRunningVariance (exponentialAverageFactor()) false
            |> t.a
        member t.inference input = 
            let bnMode = cudnnBatchNormMode.BatchNormSpatial
            matmult t.W input
            |> batch_normalization_forward bnMode t.bnScale t.bnBias t.bnRunningMean t.bnRunningVariance 1.0 true
            |> t.a

        member t.runLayer x = (t :> INNet).inference x

        member l.ToArray = [|l.W;l.bnScale;l.bnBias;l.bnRunningMean;l.bnRunningVariance|]
        member l.ResetAdjoints () = 
            l.W.setZeroAdjoint();l.bnScale.setZeroAdjoint();
            l.bnBias.setZeroAdjoint()
        member t.SGD learning_rate = 
            saxpy -learning_rate t.W.A' t.W.P'
            saxpy -learning_rate t.bnScale.A' t.bnScale.P'
            saxpy -learning_rate t.bnBias.A' t.bnBias.P'

type BNResidualFullyConnectedLayer =
    {
    W1 : d4MUnion  // Input weight tensor
    bnScale1 : d4MUnion  // Scale tensor
    bnBias1 : d4MUnion  // Bias tensor
    bnRunningMean1 : d4MUnion  // Mean tensor
    bnRunningVariance1 : d4MUnion  // Variance tensor
    a1 : d4MUnion -> d4MUnion // Activation function

    W2 : d4MUnion  // Input weight tensor
    bnScale2 : d4MUnion  // Scale tensor
    bnBias2 : d4MUnion  // Bias tensor
    bnRunningMean2 : d4MUnion  // Mean tensor
    bnRunningVariance2 : d4MUnion  // Variance tensor
    a2 : d4MUnion -> d4MUnion // Activation function
    }      

    /// Creates a layer with random weights.
    static member create weight_nchw a1 a2 =
        let bias_sizes = weight_nchw |> fun (n,c,h,w) -> (1,c,1,1)
        
        let W1 = d4MUnion.makeUniformRandomNode weight_nchw
        let bnScale1 = bias_sizes |> d4MUnion.create 
        bnScale1.setPrimal 0.1f // Initial scale value based on the Recurrent Batch Normalization paper by Cooijmans et al.
        bnScale1.setZeroAdjoint()
        let bnBias1 = bias_sizes |> d4MUnion.create
        bnBias1.setZeroPrimal()
        bnBias1.setZeroAdjoint()
        let bnRunningMean1 = bias_sizes |> d4MUnion.createConstant
        let bnRunningVariance1 = bias_sizes |> d4MUnion.createConstant

        let W2 = d4MUnion.makeUniformRandomNode weight_nchw
        let bnScale2 = bias_sizes |> d4MUnion.create 
        bnScale2.setPrimal 0.1f // Initial scale value based on the Recurrent Batch Normalization paper by Cooijmans et al.
        bnScale2.setZeroAdjoint()
        let bnBias2 = bias_sizes |> d4MUnion.create
        bnBias2.setZeroPrimal()
        bnBias2.setZeroAdjoint()
        let bnRunningMean2 = bias_sizes |> d4MUnion.createConstant
        let bnRunningVariance2 = bias_sizes |> d4MUnion.createConstant

        { W1 = W1; bnScale1 = bnScale1; bnBias1 = bnBias1; bnRunningMean1 = bnRunningMean1; bnRunningVariance1 = bnRunningVariance1; a1=a1;
          W2 = W2; bnScale2 = bnScale2; bnBias2 = bnBias2; bnRunningMean2 = bnRunningMean2; bnRunningVariance2 = bnRunningVariance2; a2=a2  }


    interface INNet with
        member t.train input exponentialAverageFactor = 
            let bnMode = cudnnBatchNormMode.BatchNormPerActivation
            matmult t.W1 input
            |> batch_normalization_forward bnMode t.bnScale1 t.bnBias1 t.bnRunningMean1 t.bnRunningVariance1 (exponentialAverageFactor()) false
            |> t.a1
            |> matmult t.W2
            |> batch_normalization_forward bnMode t.bnScale2 t.bnBias2 t.bnRunningMean2 t.bnRunningVariance2 (exponentialAverageFactor()) false
            |> fun p -> add 1.0f p 1.0f input
            |> t.a2

        member t.inference input = 
            let bnMode = cudnnBatchNormMode.BatchNormPerActivation
            matmult t.W1 input
            |> batch_normalization_forward bnMode t.bnScale1 t.bnBias1 t.bnRunningMean1 t.bnRunningVariance1 1.0 true
            |> t.a1
            |> matmult t.W2
            |> batch_normalization_forward bnMode t.bnScale2 t.bnBias2 t.bnRunningMean2 t.bnRunningVariance2 1.0 true
            |> fun p -> add 1.0f p 1.0f input
            |> t.a2

        member t.runLayer x = (t :> INNet).inference x

        member l.ToArray = [|l.W1;l.bnScale1;l.bnBias1;l.bnRunningMean1;l.bnRunningVariance1;l.W2;l.bnScale2;l.bnBias2;l.bnRunningMean2;l.bnRunningVariance2|]
        member l.ResetAdjoints () = 
            l.W1.setZeroAdjoint();l.bnScale1.setZeroAdjoint();l.bnBias1.setZeroAdjoint()
            l.W2.setZeroAdjoint();l.bnScale2.setZeroAdjoint();l.bnBias2.setZeroAdjoint()
        member t.SGD learning_rate = 
            saxpy -learning_rate t.W1.A' t.W1.P'
            saxpy -learning_rate t.bnScale1.A' t.bnScale1.P'
            saxpy -learning_rate t.bnBias1.A' t.bnBias1.P'

            saxpy -learning_rate t.W2.A' t.W2.P'
            saxpy -learning_rate t.bnScale2.A' t.bnScale2.P'
            saxpy -learning_rate t.bnBias2.A' t.bnBias2.P'

/// Adapted from the previous version of the library.
/// An optimized implementation will be done in the future along with union types and streams.
type LSTMLayer =
    {W_z:d4MUnion  // Input weight matrix for the block input
     U_z:d4MUnion  // Recurrent weight matrix for the block input
     b_z:d4MUnion  // Bias vector for the block input

     W_i:d4MUnion  // Input weight matrix for the input gate
     U_i:d4MUnion  // Recurrent weight matrix for the input gate
     b_i:d4MUnion  // Bias vector for the input gate
     P_i:d4MUnion  // Peephole weight matrix for the input gate

     W_f:d4MUnion  // Input weight matrix for the forget gate
     U_f:d4MUnion  // Recurrent weight matrix for the forget gate
     b_f:d4MUnion  // Bias vector for the forget gate
     P_f:d4MUnion  // Peephole weight matrix for the forget gate

     W_o:d4MUnion  // Input weight matrix for the output gate
     U_o:d4MUnion  // Recurrent weight matrix for the output gate
     b_o:d4MUnion  // Bias vector for the output gate
     P_o:d4MUnion  // Peephole weight matrix for the output gate

     block_input_a : d4MUnion -> d4MUnion
     block_output_a : d4MUnion -> d4MUnion
     } 
    
    /// Returns all the weights in an array.
    member l.ToArray = [|l.W_z;l.U_z;l.b_z;l.W_i;l.U_i;l.b_i;l.P_i;l.W_f;l.U_f;l.b_f;l.P_f;l.W_o;l.U_o;l.b_o;l.P_o|]
    static member fromArray (a: d4MUnion[]) block_input_a block_output_a =
        {
         W_z = a.[0]
         U_z = a.[1]
         b_z = a.[2]

         W_i = a.[3]
         U_i = a.[4]
         b_i = a.[5]
         P_i = a.[6]

         W_f = a.[7]
         U_f = a.[8]
         b_f = a.[9]
         P_f = a.[10]

         W_o = a.[11]
         U_o = a.[12]
         b_o = a.[13]
         P_o = a.[14]

         block_input_a = block_input_a
         block_output_a = block_output_a
        }

    static member createRandomLSTMLayer input_size hidden_size block_input_a block_output_a =
        {
        W_z = d4MUnion.makeUniformRandomNode (input_size, hidden_size, 1, 1)
        U_z = d4MUnion.makeUniformRandomNode (hidden_size, hidden_size, 1, 1)
        b_z = d4MUnion.makeUniformRandomNode (1, hidden_size, 1, 1)
        W_i = d4MUnion.makeUniformRandomNode (input_size, hidden_size, 1, 1)
        U_i = d4MUnion.makeUniformRandomNode (hidden_size, hidden_size, 1, 1)
        b_i = d4MUnion.makeUniformRandomNode (1, hidden_size, 1, 1)
        P_i = d4MUnion.makeUniformRandomNode (hidden_size, hidden_size, 1, 1)
        W_f = d4MUnion.makeUniformRandomNode (input_size, hidden_size, 1, 1)
        U_f = d4MUnion.makeUniformRandomNode (hidden_size, hidden_size, 1, 1)
        b_f = d4MUnion.makeUniformRandomNode (1, hidden_size, 1, 1)
        P_f = d4MUnion.makeUniformRandomNode (hidden_size, hidden_size, 1, 1)
        W_o = d4MUnion.makeUniformRandomNode (input_size, hidden_size, 1, 1)
        U_o = d4MUnion.makeUniformRandomNode (hidden_size, hidden_size, 1, 1)
        b_o = d4MUnion.makeUniformRandomNode (1, hidden_size, 1, 1)
        P_o = d4MUnion.makeUniformRandomNode (hidden_size, hidden_size, 1, 1)

        block_input_a = block_input_a
        block_output_a = block_output_a
        }

    member l.runLayer (x:d4MUnion) (y:d4MUnion) (c:d4MUnion) =
        let block_input = linear_layer_matmult [|l.W_z,x;l.U_z,y|] (Some l.b_z) |> l.block_input_a
        let input_gate = linear_layer_matmult [|l.W_i,x;l.U_i,y;l.P_i,c|] (Some l.b_i) |> sigmoid
        let forget_gate = linear_layer_matmult [|l.W_f,x;l.U_f,y;l.P_f,c|] (Some l.b_f) |> sigmoid
        let c' = linear_layer_hadmult [|block_input,input_gate;c,forget_gate|]
        let output_gate = linear_layer_matmult [|l.W_o,x;l.U_o,y;l.P_o,c'|] (Some l.b_o) |> sigmoid
        hadmult (l.block_output_a c') output_gate, c'

    member l.runLayerNoH (x:d4MUnion) =
        let block_input = linear_layer_matmult [|l.W_z,x|] (Some l.b_z) |> l.block_input_a
        let input_gate = linear_layer_matmult [|l.W_i,x|] (Some l.b_i) |> sigmoid
        let forget_gate = linear_layer_matmult [|l.W_f,x|] (Some l.b_f) |> sigmoid
        let c' = hadmult block_input input_gate
        let output_gate = linear_layer_matmult [|l.W_o,x;l.P_o,c'|] (Some l.b_o) |> sigmoid
        hadmult (l.block_output_a c') output_gate, c'

    member l.runLayerNoI (y:d4MUnion) (c:d4MUnion) =
        let block_input = linear_layer_matmult [|l.U_z,y|] (Some l.b_z) |> l.block_input_a
        let input_gate = linear_layer_matmult [|l.U_i,y;l.P_i,c|] (Some l.b_i) |> sigmoid
        let forget_gate = linear_layer_matmult [|l.U_f,y;l.P_f,c|] (Some l.b_f) |> sigmoid
        let c' = linear_layer_hadmult [|block_input,input_gate;c,forget_gate|]
        let output_gate = linear_layer_matmult [|l.U_o,y;l.P_o,c'|] (Some l.b_o) |> sigmoid
        hadmult (l.block_output_a c') output_gate, c'


let load_data file_name is_constant =
    use stream_data = IO.File.OpenRead(file_name)
    use reader_data = new IO.BinaryReader(stream_data)

    let m = reader_data.ReadInt32()
    if m <> 929856 then failwith "Wrong file type in load_weights"

    let l = reader_data.ReadInt32()
    let weights = [|
        for i=1 to l do
            let num_rows = reader_data.ReadInt32()
            let num_cols = reader_data.ReadInt32()
            let ar = [|for x=1 to num_rows*num_cols do yield reader_data.ReadSingle()|]
            match is_constant with
            | true -> yield d4MUnion.createConstant(((num_cols,num_rows,1,1),ar))
            | false -> yield d4MUnion.create(((num_cols,num_rows,1,1),ar))
        |]

    weights

let sgd learning_rate (node : d4MUnion) = saxpy -learning_rate node.A' node.P'