	? ?rh>@? ?rh>@!? ?rh>@	??????????????!???????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$? ?rh>@      ??A?/?$?=@Y?I+???*	     ?P@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????????!??|?B@)??~j?t??1?&?l??<@:Preprocessing2U
Iterator::Model::ParallelMapV2????Mb??!>???>8@)????Mb??1>???>8@:Preprocessing2F
Iterator::ModelZd;?O???!m??&?lA@)y?&1?|?16?d?M6%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipj?t???!?&?l?IP@)?~j?t?x?1/?袋."@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?~j?t???!/?袋.2@)?~j?t?x?1/?袋."@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?~j?t?x?!/?袋."@)?~j?t?x?1/?袋."@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?x?!/?袋."@)?~j?t?x?1/?袋."@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???????I
?r??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	      ??      ??!      ??      ??!       "      ??!       *      ??!       2	?/?$?=@?/?$?=@!?/?$?=@:      ??!       B      ??!       J	?I+????I+???!?I+???R      ??!       Z	?I+????I+???!?I+???b      ??!       JCPU_ONLYY???????b q
?r??X@