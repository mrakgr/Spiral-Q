# Spiral V3

The third version of the library with the union type and automatic concurrency. Hopefully this will be sufficient for me to use Deep Q Learning.

UPDATE 4/18/2016: Halfway done with rewriting the library in terms of LOC.
UPDATE 4/19/2016: Done rewriting the library. Now it will automatically execute concurrent kernels.

I still have not run it even once though, so I am sure I am in for a painful debugging experience. But after that is done, I will be able to write the optimized linear layers and the LSTM implementation. The one that come in new cuDNN v5 is not suitable as it requires prepared inputs, while for my use case, I need to do action selection at every step.

Before I continue, let me do a short list of added features since v1:
-4d tensor type ~~(replaced with the union type in this version)~~
-Cuda module caching
-Automatic concurrency
~~-Union 4d tensor type~~

Asynchronous memory copies are currently lacking but they are hardly a priority. I might need them to copy pointers for batched gemm by that is not currently on my agenda. I also added a inference_only_flag variable to prevent the adjoints being zeroed out automatically during the forward pass. For the automatic concurrency to work, the occupancy arrays need to be cleaned out after every pass. The backprop_tape function does this at start of the backward pass and the ResetOccupancy needs to be called at the end of the pass as well. The layer classes still do not clean the occupancy arrays of the base nodes automatically yet.

The primal of the Df type has been made lazy as getting the result of the sum module blocks the whole device, so that should be only done once the whole loop has been executed.

In terms of design, I quite like the library as it is now. I've always felt that way about it. As for the previous versions, building something significant and internalizing every part of it is definitely a satisfying experience. Doing the first version of Spiral and then the port of the GVGAI library really awoke me to the joys and benefits of functional programming. The more I do it, the more I love it. I've heard it said that it takes 3 years for a programmer to reach his peak. 1.5 years left to go then.

UPDATE 4/21/2016: As expected, concurrent programming does not spare in the pain department unfortunately. I am now completely stuck trying to track down that out of order execution bug. I cannot tell from the visual profiler what is wrong and trying to get a sense of when things are getting executed with callbacks "fixes" the error. For all I know this might be an error with Cuda. I'll have to take some time to do research and ask around.

UPDATE 4/22/2016: Finally figured out the bug that made me spend an entire day yesterday. It turns out that I was inadvertedly sharing the large workspace. The thing with FFT and Winograd functions is that they require workspaces in the 100-300Mb range so I made in the past version for the large workspace to be reused. Then in this version that got inadvertedly shared.

Well, at any rate, I still have to do some more tests regarding the functioning of events and there is an optimization on the backwards pass for the occupancy array and then I will be ready to start working on the new layers and LSTM implementation.

UPDATE 4/23/2016: Done with the second rewrite. Thanks to the experience of the last few days, I have a much better idea how Cuda concurrency works. Cuda Events are generally described as boolean variables, but it seems they are more like pointers to the stream queue where the `record` was last called. The thing they are pointing at is like a mailbox (of the first stream) that collects callbacks from the other streams that are waiting on the queue. When the first stream finishes its work, it sends those callbacks back allowing the other streams to stop waiting. The `query` function is just there to confuse users. There is no overwritting of variables in the stream and the streams do not erroneously switch to waiting on subseqent calls of record of the event like I thought they might.

At least, that is how I imagine it right now. Hopefully what I wrote here is correct otherwise I am going to have to rewrite the library again. I'd very much like to get back into RL, but working on the library does take precedence. I've also decided to go over CNTK's source as well. That library strikes me like a pyramid though - tall, humoungous and brittle. But as a ML student my notable weakness is that I do not know any of the big libraries, so I might as well see where that path leads.

First though, comes testing.

Edit: Feedforward, convolutional and convolutional with BN nets work correctly. _v1 is ready for operation.

I also tested the library on the unpotimized LSTM and the result have been disappointing. It seems the scheduler is not good enough to extract much concurrency on its own, though I did observe a 10% speedup with 32 streams. Such bad luck. It does seem like I will get more mileage out the union type at any rate.

I can confirm that my idea of how Cuda event synchronization works is pretty much correct. I am sure of it now.

UPDATE 4/24/2016: I have to do yet another rewrite. The Cuda scheduler is quite poor and in the linear_layer_test.fsx, I've verified that concurrency works much better when the streams are reused. Tsk.

If I had to make a wish in cuDNN I would like if they could make optimized linear layers. Currently one optimization I could not do myself would be to have gemm do atomic adds to the output arrays. That would be greatly helpful. I really do not feel like allocating temporary arrays for the linear layer.

Hopefully this will be the last rewrite. I think I have a full idea of how to do automatic concurrency efficiently now, though this is hardly the end of optimization. The trouble with concurrent (and parallel) programming is that for a given problem, changing the inputs changes the structure of the problem. An AI working at 1000x speed of meat brain would have a strong advantage in getting the most juice out of available hardware. It is annoying to be the biggest bottleneck in programming.

With any luck, when I scale up the RNNs in depth and multiple dimensions, the benefits of what I am doing now will become clear. Out of all the optimizations, [the wavefront iteration](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/) from the linked post might be the easiest one to do. A simple and general, if relatively inneficient way of doing it would be to store all the (i,j) tuples in an array and sort them by Manhattan distance from (0,0).

UPDATE 4/25/2016: Done with the third rewrite. Time for testing.

I also figured out how to do the wavefront iteration in an elegant fashion without the need for sorting for an arbitrary number of dimensions. It might not be necessarily more efficient though when all is said and done. It is in the `wavefront_iteration_example.fsx`.

Edit: Done with testing. Found and fixed a bug in the `matmult'` function. Just like previously I haven't had any luck with extracting more from the unoptimized LSTM with this. It does bring out the full potential in the `linear_layer_test_v1.fsx` which is a synthetic benchmark. It terms of automating concurrency, I can't do much more than this for the library. I learned quite a bit about Cuda concurrency from this.

Next step - the optimized linear layer.

Edit2: The performance of the concurrent linear layer is so poor compared to the original that it is not worth the effort putting into the library.

Maybe if I had the batched gemm with atomic addition to the outputs it would make sense, but not like this. After testing, I've concluded that whether there is a benefit to concurrency depends on whether it is possible to saturate the GPU. And the reason the Reber RNN example does so poorly with extra concurrency is that it is simply too small. I need to expand in depth and width to get the full benefit from it.

An interesting idea that is a small step beyond what I just did in this library would be to sort each forward step by depth.

This would have the optimal scheduling benefit of the wavefront iteration, but for any arbitrary architecture. It is quite amazing now that I think about it. It would definitely be worth adding at some point. It would not be possible to do without what I did just now, but now that I come this far, the step is fairly obvious. It is strange that the cuDNN post said that the mainstream libraries are not using it.

At any rate, right now I am tired from working on the library for so long. I want to get back into reinforcement learning.

Let me wrap this up by doing the LSTM with the union type.

UPDATE 4/26/2016: I've worked out how to make the union type work, but the cuDNN library currently lacks the CHWN layout so it won't be possible to get it done with convolutional functions for example. To be honest, the entire thing is quite tedious in terms of book-keeping so much that I've considered leaving it out. Before I move on to it, let me go with the sort idea from yesterday. It occurs to me that I've made a serious mistake by not printing out the depths in which the nodes are executed. That would have told me quite a lot about where the bottlenecks lie.

Also if I end up going with GRUs instead of LSTMs I would have just wasted my time doing the union type.

Edit: I've concluded that I simply cannot do the union type. As a feature it is too expensive for me in terms of effort required to support it. And I cannot think of an elegant way to add it to the library. Starting from here I'll look into removing the stump and wrap this up. One last feature I will add is to redirect gemm to gemv if the matrix-matrix is in fact matrix-vector multiplication.

In terms of complexity, probably I am at my limit with Spiral here. There is only so much one person can do without making library maintenance a full time occupation. Past this, I'd be better off writting a .NET F# wrapper for Tensorflow when it comes to Windows, which it eventually will.

Currently, I am trying to find some benefit in the concurrency thing that I did. When scaling the RNN LSTM example so it uses multiple layers, I finally start to observe some benefit. Right now I am in the process of creating a wavefront iteration example.

...I need a break. This has definitely gotten tedious, but I'll keep marching on. Same as usual. Making those RL players was a lot more fun than this. After I am done with this iteration I will definitely look towards outsourcing my ML work to some library. I know the ins and outs of backprop now so I would not feel bad about that anymore.

Edit2: I've tested this to hell. I cannot get the benfits of concurrency for the LSTM class to go much above 30% in any realistic setting. Not really what I expected. I really got carried away by that cuDNN v5 post.

In the end, it depends on the architecture. The sparser and more divergent the architecture, the more benefit I can expect to accrue from concurrency. By sparsity I do not mean a sparse activation or dropout, or anything along that line, but having seperate loosely connected modules. That could be where the benefits of concurrency accrue.

UPDATE 4/27/2016: Cleaned up the thing a little, but still a bit more remains to be done. I also need to put in gemv rerouting and take out d4MUnion stump.

Learning is an endless series of reviews so let me do one here.

I first started working on this library last year. I started by making neural nets by hand for the purpose of internalizing them. Despite much effort I never got far with that approach and instead discovered the wonder tool that is automatic differentiation. Changing directions, I ended up learning it by studying another F# library called [Diffsharp](http://diffsharp.github.io/DiffSharp/). Compared to the AD libraries in other languages, its code is quite an elegant construct, in a league of its own.

The choice of language definitely has a hand in that. Given a set amount of effort, the result produced will be the outcome of the tools used. The tools - programming languages and libraries - are the mold and Spiral is the resin removed from the mold.

To get to the higher level, what would be required for me here would be better tools. At the current level, I think that for every 30 line of Cuda GPU code, I could represent the same thing in one line of F# CPU code. It is a massive bottleneck both in terms of time taken to make a program and in brainpower expended. If I could spare a thousand times more effort than I could today, I would invest it all into programming language research.

The things that gave me difficulty in making of V3 like the union type and the thing that I did not even mention like kernel fusing would be easy had I easy access to a higher level of abstraction. It would be equivalent to a free boost in skill across the board.

This is largely the reason for my attraction to F#. The C++ (and even Python) programmers are very much missing out. Worst, there will be better languages than F#, and by sticking to languages worse than it, the programming community is invariably delaying their arrival. It is a pity that there is truth to the adage that science marches from funeral to funeral.

When I first started this over a year ago, my feelings were that I had to have all the basics down and now I think that moment has come. I've spent a significant amount of effort on this library, but I've known that I won't be compete with large teams working on the mainstream libraries. And some of the things currently being done stray into programming language research, like [NervanaGPU](https://github.com/NervanaSystems/nervanagpu)'s automatic kernel fusing or Theano's and Tensorflow's symbolic differentiation which is much like a regular compiler's optimization pass. In regards to the later, I've thought at first that symbolic differentiation was really separate from AD, but it is more like orthogonal to it. It is not so much about symbolic differentiation or AD, but the fact that Cuda gets significantly more efficient when the kernels are fused together, so much that the concurrency thing I just did in the past week is kind of like putting a bandaid on bullet hole. Rather than AD and SD being opposed, it is better put that SD is competing with regular programming language features.

As an example, since I failed to do the union type due to the lack of appropriate tooling, I did some tests to verify whether there was any truth to the [cuDNN v5](https://devblogs.nvidia.com/parallelforall/optimizing-recurrent-neural-networks-cudnn-5/) post on fusing the weight matrix multiplications. For a 128x128 * 128*512 vs (4 fold) 128x128 * 128x128 matrix multiply with a single stream there is. As per post, it the first option is roughly 110%. With with concurrent streams that advantage dwindless down to 50% which is more manageable at first glance making it look like concurrency might be a viable measure versus parallelism.

But despite my initial misgivings I think the point stands. 128x128 * 128*1280 vs (10 fold) 128x128 * 128x128 is roughly 4 times faster vs a single stream and 3 times faster versus 32 concurrent streams and I would guess that the lead would only get bigger if I compared it to a 20 or 30 fold multiply and more. SD could do this optimization automatically by detecting that multiple matrices are being multiplied with the same input.

Backpropagation (or reverse AD) is really an amazing and a flexible algorithm. I actually have no idea what the limit of its potential is. In a different field of optimization, there was an [amazing degree of improvement](http://www.math.uwaterloo.ca/~hwolkowi//henry/teaching/f14/602.f14/602miscfiles/UF_Entrepreneurship_19November2014.pdf) from 1990-2014. With the correct tools and abstractions to bring out the most out of the hardware and with more algorithmic improvements on par with batch normalization, the machine learning field should be an interesting thing to keep abreast of.

For comparison, just ten years ago, training a fully connected net with not even a dozen layers by RBM pretraining was a singnificant achievement.

Today with stochastic depth, batch normalization and residual learning, training a thousand layer net without any pretraining is possible without the gradients disappearing or exploding.

2026? I'll leave that as an open question. But rather than to just ponder, the best would be to reach out for the answer oneself.