	?O??n?:@?O??n?:@!?O??n?:@	;J?q"8??;J?q"8??!;J?q"8??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?O??n?:@???Mb??A%??C+:@Y??~j?t??*	     ?i@2U
Iterator::Model::ParallelMapV2?~j?t???!??????G@)?~j?t???1??????G@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????Mb??!______?@)?v??/??1??????;@:Preprocessing2F
Iterator::Model?"??~j??!?????4K@)???Q???1jiiiii@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?~j?t???!??????@)?~j?t???1??????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+??η?!KKKKK?F@);?O??n??1??????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{?G?z??!??????#@)????Mb??1______@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1?|?!tsssss@)y?&1?|?1tsssss@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9;J?q"8??Il????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???Mb?????Mb??!???Mb??      ??!       "      ??!       *      ??!       2	%??C+:@%??C+:@!%??C+:@:      ??!       B      ??!       J	??~j?t????~j?t??!??~j?t??R      ??!       Z	??~j?t????~j?t??!??~j?t??b      ??!       JCPU_ONLYY;J?q"8??b ql????X@