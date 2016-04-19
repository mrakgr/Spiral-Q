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