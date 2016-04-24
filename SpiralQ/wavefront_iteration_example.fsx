// https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/
// The last optimization from the above post. This is a sorting based implementation.
// I can't think of anything more elegant that a state machine at the moment

let x_len = 4
let y_len = 1
let z_len = 1

let wavefront_order =
    [|
    for x=0 to x_len do
        for y=0 to y_len do
            for z=0 to z_len do
                yield (x,y,z)
    |]
    |> Array.sortBy (fun (x,y,z) -> x+y+z)

// http://stackoverflow.com/questions/36823316/does-there-exist-an-efficient-way-of-doing-a-wavefront-iterator-not-physics-re
// John Palmer's idea.
// It works quite beatifully. Exactly what I wanted.

let presort =
    [|
    for sum=0 to x_len+y_len+z_len do
        for x=0 to min sum x_len do
            for y=0 to min (sum-x) y_len do
                let z = sum-x-y
                if z <= z_len then yield (x,y,z)
                |]

presort.Length 
wavefront_order.Length
presort |> Array.map (fun (x,y,z) -> x+y+z)
wavefront_order |> Array.map (fun (x,y,z) -> x+y+z)