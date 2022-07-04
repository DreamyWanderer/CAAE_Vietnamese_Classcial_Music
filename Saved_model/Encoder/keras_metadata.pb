
�7root"_tf_keras_network*�7{"name": "Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Encoder_input"}, "name": "Encoder_input", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "Flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Flatten_layer", "inbound_nodes": [[["Encoder_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Hidden_layer_1", "trainable": true, "dtype": "float32", "units": 150, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden_layer_1", "inbound_nodes": [[["Flatten_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Hidden_layer_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden_layer_2", "inbound_nodes": [[["Hidden_layer_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Mean_layer", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Mean_layer", "inbound_nodes": [[["Hidden_layer_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Variance_layer", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Variance_layer", "inbound_nodes": [[["Hidden_layer_2", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "Encoder_output", "trainable": true, "dtype": "float32"}, "name": "Encoder_output", "inbound_nodes": [[["Mean_layer", 0, 0, {}], ["Variance_layer", 0, 0, {}]]]}], "input_layers": [["Encoder_input", 0, 0]], "output_layers": [["Mean_layer", 0, 0], ["Variance_layer", 0, 0], ["Encoder_output", 0, 0]]}, "shared_object_id": 15, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28, 28]}, "float32", "Encoder_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28, 28]}, "float32", "Encoder_input"]}, "keras_version": "2.9.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Encoder_input"}, "name": "Encoder_input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Flatten", "config": {"name": "Flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "Flatten_layer", "inbound_nodes": [[["Encoder_input", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Dense", "config": {"name": "Hidden_layer_1", "trainable": true, "dtype": "float32", "units": 150, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden_layer_1", "inbound_nodes": [[["Flatten_layer", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Dense", "config": {"name": "Hidden_layer_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Hidden_layer_2", "inbound_nodes": [[["Hidden_layer_1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "Mean_layer", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Mean_layer", "inbound_nodes": [[["Hidden_layer_2", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "Variance_layer", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Variance_layer", "inbound_nodes": [[["Hidden_layer_2", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Sampling", "config": {"name": "Encoder_output", "trainable": true, "dtype": "float32"}, "name": "Encoder_output", "inbound_nodes": [[["Mean_layer", 0, 0, {}], ["Variance_layer", 0, 0, {}]]], "shared_object_id": 14}], "input_layers": [["Encoder_input", 0, 0]], "output_layers": [["Mean_layer", 0, 0], ["Variance_layer", 0, 0], ["Encoder_output", 0, 0]]}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "Encoder_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Encoder_input"}}2
�root.layer-1"_tf_keras_layer*�{"name": "Flatten_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "Flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Encoder_input", 0, 0, {}]]], "shared_object_id": 1, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 17}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "Hidden_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Hidden_layer_1", "trainable": true, "dtype": "float32", "units": 150, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Flatten_layer", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "Hidden_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Hidden_layer_2", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Hidden_layer_1", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 150}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 150]}}2
�root.layer_with_weights-2"_tf_keras_layer*�{"name": "Mean_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Mean_layer", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Hidden_layer_2", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}2
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "Variance_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Variance_layer", "trainable": true, "dtype": "float32", "units": 32, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Hidden_layer_2", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}2
�root.layer-6"_tf_keras_layer*�{"name": "Encoder_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Sampling", "config": {"name": "Encoder_output", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["Mean_layer", 0, 0, {}], ["Variance_layer", 0, 0, {}]]], "shared_object_id": 14}2