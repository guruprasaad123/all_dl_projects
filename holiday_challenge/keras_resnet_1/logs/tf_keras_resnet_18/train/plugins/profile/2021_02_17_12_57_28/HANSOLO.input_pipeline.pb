	?H??2?@?H??2?@!?H??2?@	??pK$?????pK$???!??pK$???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?H??2?@??y76@A'?W??@Y5^?Ib1@*	????9??@2F
Iterator::Model:??HO1@!??ϧ?X@)S??:A1@1?9q?k?X@:Preprocessing2S
Iterator::Model::ParallelMap?Q???!???0?;??)?Q???1???0?;??:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat?~j?t???!?"%ܰ??)??ͪ?Ֆ?1ɩo?p??:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate?HP???!??GW???)?5?;Nё?1ciq?ئ??:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!????????)y?&1?|?1????????:Preprocessing2X
!Iterator::Model::ParallelMap::Ziplxz?,C??!F0?]0X??)?g??s?u?1?3?c?@??:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor_?Q?[?!??&;???)_?Q?[?1??&;???:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap??ݓ????!X?4
e??)?~j?t?X?1?"%ܰ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??y76@??y76@!??y76@      ??!       "      ??!       *      ??!       2	'?W??@'?W??@!'?W??@:      ??!       B      ??!       J	5^?Ib1@5^?Ib1@!5^?Ib1@R      ??!       Z	5^?Ib1@5^?Ib1@!5^?Ib1@JCPU_ONLY