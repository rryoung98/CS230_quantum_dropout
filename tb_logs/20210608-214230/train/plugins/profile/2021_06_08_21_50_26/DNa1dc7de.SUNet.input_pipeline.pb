	sh??|?4@sh??|?4@!sh??|?4@	?q?T????q?T???!?q?T???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$sh??|?4@?Q?????A????x?3@Yy?&1???*	     ?O@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?~j?t???!?0?0C@)Zd;?O???1AAB@:Preprocessing2U
Iterator::Model::ParallelMapV2??~j?t??!??(??(>@)??~j?t??1??(??(>@:Preprocessing2F
Iterator::Model???S㥛?!۶m۶mE@)????Mb??1Y?eY?e)@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?~j?t???!?0?03@)y?&1?|?1??8??8&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice{?G?zt?!??????@){?G?zt?1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?!Y?eY?e??)????MbP?1Y?eY?e??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?q?T???I??#?L?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Q??????Q?????!?Q?????      ??!       "      ??!       *      ??!       2	????x?3@????x?3@!????x?3@:      ??!       B      ??!       J	y?&1???y?&1???!y?&1???R      ??!       Z	y?&1???y?&1???!y?&1???b      ??!       JCPU_ONLYY?q?T???b q??#?L?X@