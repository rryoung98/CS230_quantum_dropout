	B`??"?5@B`??"?5@!B`??"?5@	vI?]m??vI?]m??!vI?]m??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$B`??"?5@\???(\??A?~j?ts5@Y{?G?z??*	     ?O@2F
Iterator::Model?I+???!]?u]?uA@)?I+???1]?u]?uA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZd;?O???!AAB@)????Mb??1Y?eY?e9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1?|?!??8??8&@)y?&1?|?1??8??8&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipˡE?????!QEQEP@)?~j?t?x?1?0?0#@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?~j?t???!?0?03@)?~j?t?x?1?0?0#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?~j?t?x?!?0?0#@)?~j?t?x?1?0?0#@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9vI?]m??I?%?vI?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	\???(\??\???(\??!\???(\??      ??!       "      ??!       *      ??!       2	?~j?ts5@?~j?ts5@!?~j?ts5@:      ??!       B      ??!       J	{?G?z??{?G?z??!{?G?z??R      ??!       Z	{?G?z??{?G?z??!{?G?z??b      ??!       JCPU_ONLYYvI?]m??b q?%?vI?X@