// As the concurrent aspect of the library is giving me quite a lot of trouble, 
// I need some feedback on the order the functions are executing. The visual profiler is crap.

// I'll try doing a callback here.

// Edit: Was succesful. 

#if INTERACTIVE
#load "spiral_q_v0.fsx"
#endif
open SpiralV3
open ManagedCuda
open ManagedCuda.BasicTypes
open ManagedCuda.VectorTypes
open ManagedCuda.CudaBlas
open ManagedCuda.CudaRand
open ManagedCuda.NVRTC
open ManagedCuda.CudaDNNv5

open System
open System.Runtime
open System.Runtime.InteropServices

let str = new CudaStream()

type unit_to_unit_delegate = delegate of unit -> unit
let add_callback_to_stream (str : CudaStream) (callback : unit -> unit) =
    let callb (str : CUstream) (res : CUResult) (p : nativeint) =
        let f = p.ToPointer()
        let t : unit_to_unit_delegate = Runtime.InteropServices.Marshal.GetDelegateForFunctionPointer(p)
        t.Invoke()

    let aux = new unit_to_unit_delegate (callback)
    let ptr_to_aux = Marshal.GetFunctionPointerForDelegate aux

    let res = CUResult.Success
    let cuda_callback = CUstreamCallback(callb)
    str.AddCallback(cuda_callback,ptr_to_aux,CUStreamAddCallbackFlags.None)

add_callback_to_stream str (fun _ -> printfn "Hello, World!")
