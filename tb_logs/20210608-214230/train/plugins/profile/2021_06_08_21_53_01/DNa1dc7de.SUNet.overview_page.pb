?	R????5@R????5@!R????5@	qG~84h??qG~84h??!qG~84h??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$R????5@^?I+??AX9??v5@YbX9?ȶ?*	      `@2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???Q???!??}A7@)???Q???1??}A7@:Preprocessing2U
Iterator::Model::ParallelMapV2y?&1???!mI[Җ?5@)y?&1???1mI[Җ?5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?A`??"??!???+??D@)Zd;?O???1uE]QW?1@:Preprocessing2F
Iterator::Model
ףp=
??!?w?qA@)?? ?rh??1??%mI[*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip/?$???!w?qGP@)y?&1???1mI[Җ?%@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?? ?rh??!??%mI[*@);?O??n??1????/?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice????Mb??!4??9c?@)????Mb??14??9c?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9sG~84h??I???˗?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	^?I+??^?I+??!^?I+??      ??!       "      ??!       *      ??!       2	X9??v5@X9??v5@!X9??v5@:      ??!       B      ??!       J	bX9?ȶ?bX9?ȶ?!bX9?ȶ?R      ??!       Z	bX9?ȶ?bX9?ȶ?!bX9?ȶ?b      ??!       JCPU_ONLYYsG~84h??b q???˗?X@Y      Y@qL?\??'@"?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 