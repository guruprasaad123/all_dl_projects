	L7?A ??@L7?A ??@!L7?A ??@	?w?6?w??w?6?w?!?w?6?w?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$L7?A ??@???S????A??v????@Y?E??????*	?????i@2S
Iterator::Model::ParallelMapffffff??!?J???E@)ffffff??1?J???E@:Preprocessing2F
Iterator::Model?p=
ף??!{?&]-P@)??(\?¥?1??1?'5@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat??e?c]??!ҧӻn?+@)c?ZB>???1?????/)@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenatevq?-??!`?!!\u/@)??@??ǘ?11s8Mr(@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip?QI??&??!?̳E?A@)Έ?????11Rs?D?@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???_vO~?!?@?O?w@)???_vO~?1?@?O?w@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!?LA.??)a2U0*?c?1?LA.??:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapz6?>W[??!?a????0@)HP?s?b?1??TLQ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???S???????S????!???S????      ??!       "      ??!       *      ??!       2	??v????@??v????@!??v????@:      ??!       B      ??!       J	?E???????E??????!?E??????R      ??!       Z	?E???????E??????!?E??????JCPU_ONLY