	??C?l?6@??C?l?6@!??C?l?6@	V?ر???V?ر???!V?ر???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??C?l?6@?l??????AZd;?/6@YZd;?O???*	      N@2U
Iterator::Model::ParallelMapV2?? ?rh??!UUUUUU<@)?? ?rh??1UUUUUU<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??~j?t??!???????@)???Q???1      9@:Preprocessing2F
Iterator::Model9??v????!??????E@);?O??n??1      .@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?? ?rh??!UUUUUUL@){?G?zt?1?????? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap{?G?z??!??????0@){?G?zt?1?????? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice{?G?zt?!?????? @){?G?zt?1?????? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!??????@)????Mbp?1??????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9U?ر???Is?'?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?l???????l??????!?l??????      ??!       "      ??!       *      ??!       2	Zd;?/6@Zd;?/6@!Zd;?/6@:      ??!       B      ??!       J	Zd;?O???Zd;?O???!Zd;?O???R      ??!       Z	Zd;?O???Zd;?O???!Zd;?O???b      ??!       JCPU_ONLYYU?ر???b qs?'?X@