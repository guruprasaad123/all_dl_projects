	?*?%?@?*?%?@!?*?%?@	:?#*????:?#*????!:?#*????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?*?%?@?f??j+??AO??e?#?@Y0?'???*	    ?h@2F
Iterator::Model?W[?????!R?????N@)??ׁsF??1??'D@:Preprocessing2S
Iterator::Model::ParallelMap??_?L??!???ͤ+5@)??_?L??1???ͤ+5@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatQ?|a??!S9h`E2@)2??%䃞?1t^7??T.@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate?X?? ??!_~??f?-@)??ZӼ???1P{k?$@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??y?):??!?\b??@)??y?):??1?\b??@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip??N@a??!?TCC@)vq?-??1,ܾC?@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor?HP?x?!?Pd???@)?HP?x?1?Pd???@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap?|a2U??! ????;0@)??_?Le?1???ͤ+??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?f??j+???f??j+??!?f??j+??      ??!       "      ??!       *      ??!       2	O??e?#?@O??e?#?@!O??e?#?@:      ??!       B      ??!       J	0?'???0?'???!0?'???R      ??!       Z	0?'???0?'???!0?'???JCPU_ONLY