# Spiral V3

The third version of the library with the union type and automatic concurrency. Hopefully this will be sufficient for me to use Deep Q Learning.

UPDATE 4/18/2016: Halfway done with rewriting the library in terms of LOC.
UPDATE 4/19/2016: Done rewriting the library. Now it will automatically execute concurrent kernels.

I still have not run it even once though, so I am sure I am in for a painful debugging experience. But after that is done, I will be able to write the optimized linear layers and the LSTM implementation. The one that come in new cuDNN v5 is not suitable as it requires prepared inputs, while for my use case, I need to do action selection at every step.

Before I continue, let me do a short list of added features since v1:
-4d tensor type (replaced with the union type in this version)
-Cuda module caching
-Automatic concurrency
-Union 4d tensor type

Asynchronous memory copies are currently lacking but they are hardly a priority. I might need them to copy pointers for batched gemm by that is not currently on my agenda. I also added a inference_only_flag variable to prevent the adjoints being zeroed out automatically during the forward pass. For the automatic concurrency to work, the occupancy arrays need to be cleaned out after every pass. The backprop_tape function does this at start of the backward pass and the ResetOccupancy needs to be called at the end of the pass as well. The layer classes still do not clean the occupancy arrays of the base nodes automatically yet.

The primal of the Df type has been made lazy as getting the result of the sum module blocks the whole device, so that should be only done once the whole loop has been executed.

In terms of design, I quite like the library as it is now. I've always felt that way about it. As for the previous versions, building something significant and internalizing every part of it is definitely a satisfying experience. Doing the first version of Spiral and then the port of the GVGAI library really awoke me to the joys and benefits of functional programming. The more I do it, the more I love it. I've heard it said that it takes 3 years for a programmer to reach his peak. 1.5 years left to go then.

UPDATE 4/21/2016: As expected, concurrent programming does not spare in the pain department unfortunately. I am now completely stuck trying to track down that out of order execution bug. I cannot tell from the visual profiler what is wrong and trying to get a sense of when things are getting executed with callbacks "fixes" the error. For all I know this might be an error with Cuda. I'll have to take some time to do research and ask around.

UPDATE 4/22/2016: Finally figured out the bug that made me spend an entire day yesterday. It turns out that I was inadvertedly sharing the large workspace. The thing with FFT and Winograd functions is that they require workspaces in the 100-300Mb range so I made in the past version for the large workspace to be reused. Then in this version that got inadvertedly shared.

Well, at any rate, I still have to do some more tests regarding the functioning of events and there is an optimization on the backwards pass for the occupancy array and then I will be ready to start working on the new layers and LSTM implementation.

UPDATE 4/23/2016: Done with the second rewrite. Thanks to the experience of the last few days, I have a much better idea how Cuda concurrency works. Cuda Events are generally described as boolean variables, but it seems they are more like pointers to the stream queue where the `record` was last called. The thing they are pointing at is like a mailbox (of the first stream) that collects callbacks from the other streams that are waiting on the queue. When the first stream finishes its work, it sends those callbacks back allowing the other streams to stop waiting. The `query` function is just there to confuse users. There is no overwritting of variables in the stream and the streams do not erroneously switch to waiting on subseqent calls of record of the event like I thought they might.

At least, that is how I imagine it right now. Hopefully what I wrote here is correct otherwise I am going to have to rewrite the library again. I'd very much like to get back into RL, but working on the library does take precedence. I've also decided to go over CNTK's source as well. That library strikes me like a pyramid though - tall, humoungous and brittle. But as a ML student my notable weakness is that I do not know any of the libraries, so I might as well see where that path leads.

First though, comes testing.

Edit: Feedforward, convolutional and convolutional with BN nets work correctly. _v1 is ready for operation.

I also tested the library on the unpotimized LSTM and the result have been disappointing. It seems the scheduler is not good enough to extract much concurrency on its own, though I did observe a 10% speedup with 32 streams. Such bad luck. It does seem like I will get more mileage out the union type at any rate.

I can confirm that my idea of how Cuda event synchronization works is pretty much correct. I am sure of it now.

UPDATE 4/24/2016: I have to do yet another rewrite. The Cuda scheduler is quite poor and in the linear_layer_test.fsx, I've verified that concurrency works much better when the streams are reused. Tsk.

If I had to make a wish in cuDNN I would like if they could make optimized linear layers. Currently one optimization I could not do myself would be to have gemm do atomic adds to the output arrays. That would be greatly helpful. I really do not feel like allocating temporary arrays for the linear layer.

Hopefully this will be the last rewrite. I think I have a full idea of how to do automatic concurrency efficiently now, though this is hardly the end of optimization. The trouble with concurrent (and parallel) programming is that for a given problem, changing the inputs changes the structure of the problem. An AI working at 1000x speed of meat brain would have a strong advantage in getting the most juice out of available hardware. It is annoying to be the biggest bottleneck in programming.

With any luck, when I scale up the RNNs in depth and multiple dimensions, the benefits of what I am doing now will become clear. Out of all the optimizations, [the wavefront iteration](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/) from the linked post might be the easiest one to do. A simple and general, if relatively inneficient way of doing it would be to store all the (i,j) tuples in an array and sort them by Manhattan distance from (0,0).
