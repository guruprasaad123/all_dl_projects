	O???#?@O???#?@!O???#?@	???J??????J???!???J???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$O???#?@?d?`TR??A
h"lx?@Y?uq`@*	???????@2F
Iterator::Modelv??????!?I%?W@)T㥛? ??1?/?k?W@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat$???~???!?]??@)0L?
F%??1?x?Y??@:Preprocessing2S
Iterator::Model::ParallelMap@?߾???!%BC?? @)@?߾???1%BC?? @:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenatez6?>W[??!"?ш??)??ͪ?Ֆ?1?.w?m??:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???????!<rA]6??)???????1<rA]6??:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor?+e?X??!????&???)?+e?X??1????&???:Preprocessing2X
!Iterator::Model::ParallelMap::Zip_?L?J??!
c??^^@)A??ǘ???1Q?|?TW??:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap	??g????!y?Y??? @){?G?zt?1=?-??l??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?d?`TR???d?`TR??!?d?`TR??      ??!       "      ??!       *      ??!       2	
h"lx?@
h"lx?@!
h"lx?@:      ??!       B      ??!       J	?uq`@?uq`@!?uq`@R      ??!       Z	?uq`@?uq`@!?uq`@JCPU_ONLY