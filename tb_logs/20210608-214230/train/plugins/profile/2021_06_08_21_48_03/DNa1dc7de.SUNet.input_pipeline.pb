	X9???9@X9???9@!X9???9@	Dn?R???Dn?R???!Dn?R???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$X9???9@ˡE?????AL7?A`E9@Y#??~j???*	     ?Z@2U
Iterator::Model::ParallelMapV2/?$???!R????C@)/?$???1R????C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty?&1???!oe?Cj:@)?I+???1???!5?4@:Preprocessing2F
Iterator::Model?A`??"??!      I@)?I+???1???!5?$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??~j?t??!?????1@)?I+???1???!5?$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice????Mb??!??sHM0@)????Mb??1??sHM0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?A`??"??!      I@)?~j?t?x?1??V?9?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?x?!??V?9?@)?~j?t?x?1??V?9?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Dn?R???I?#oZ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ˡE?????ˡE?????!ˡE?????      ??!       "      ??!       *      ??!       2	L7?A`E9@L7?A`E9@!L7?A`E9@:      ??!       B      ??!       J	#??~j???#??~j???!#??~j???R      ??!       Z	#??~j???#??~j???!#??~j???b      ??!       JCPU_ONLYYDn?R???b q?#oZ?X@