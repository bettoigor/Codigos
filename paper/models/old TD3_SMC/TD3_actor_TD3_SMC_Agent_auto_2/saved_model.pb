в
А§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.0.0-beta12v2.0.0-beta0-16-g1d912138х
u
dense/kernelVarHandleOp*
shape:	*
shared_namedense/kernel*
_output_shapes
: *
dtype0

 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
_class
loc:@dense/kernel*
dtype0
m

dense/biasVarHandleOp*
shared_name
dense/bias*
_output_shapes
: *
dtype0*
shape:

dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0*
_class
loc:@dense/bias
z
dense_1/kernelVarHandleOp*
shape:
*
dtype0*
shared_namedense_1/kernel*
_output_shapes
: 

"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*!
_class
loc:@dense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
dtype0*
shared_namedense_1/bias*
_output_shapes
: *
shape:

 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:*
_class
loc:@dense_1/bias
y
dense_2/kernelVarHandleOp*
shape:	*
_output_shapes
: *
dtype0*
shared_namedense_2/kernel

"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	*
dtype0*!
_class
loc:@dense_2/kernel
p
dense_2/biasVarHandleOp*
dtype0*
shape:*
_output_shapes
: *
shared_namedense_2/bias

 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
_class
loc:@dense_2/bias*
dtype0

NoOpNoOp
н
ConstConst"/device:CPU:0*
valueB B
ф
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
R

	variables
trainable_variables
regularization_losses
	keras_api


kernel
bias
_callable_losses
_eager_losses
	variables
trainable_variables
regularization_losses
	keras_api


kernel
bias
_callable_losses
_eager_losses
	variables
trainable_variables
regularization_losses
	keras_api


kernel
bias
 _callable_losses
!_eager_losses
"	variables
#trainable_variables
$regularization_losses
%	keras_api
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
y

&layers
	variables
'metrics
(non_trainable_variables
trainable_variables
regularization_losses
 
 
 
 
y

)layers

	variables
*metrics
+non_trainable_variables
trainable_variables
regularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
 
y

,layers
	variables
-metrics
.non_trainable_variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
 
y

/layers
	variables
0metrics
1non_trainable_variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
 
y

2layers
"	variables
3metrics
4non_trainable_variables
#trainable_variables
$regularization_losses

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
dtype0*
_output_shapes
: 
z
serving_default_input_1Placeholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ
Ъ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ*
Tout
2*
Tin
	2*/
f*R(
&__inference_signature_wrapper_10944395
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
о
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpConst*
Tout
2*
Tin

2**
config_proto

GPU 

CPU2J 8*/
_gradient_op_typePartitionedCall-10944440**
f%R#
!__inference__traced_save_10944439*
_output_shapes
: 
с
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*/
_gradient_op_typePartitionedCall-10944471**
config_proto

GPU 

CPU2J 8*-
f(R&
$__inference__traced_restore_10944470*
Tout
2*
_output_shapes
: *
Tin
	2цы
к
Љ
(__inference_dense_layer_call_fn_10944252

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_10944241*/
_gradient_op_typePartitionedCall-10944247
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
е

!__inference__traced_save_10944439
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_c34c50e4fc044ec182d8873fde3568e8/part*
_output_shapes
: *
dtype0s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ж
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*п
valueеBвB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:y
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueBB B B B B B *
dtype0
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop"/device:CPU:0*
dtypes

2*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B У
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2Й
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*M
_input_shapes<
:: :	::
::	:: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints:+ '
%
_user_specified_namefile_prefix: : : : : : : 
е	
м
C__inference_dense_layer_call_and_return_conditional_losses_10944241

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0Q
ReluReluBiasAdd:output:0*(
_output_shapes
:џџџџџџџџџ*
T0
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
й	
о
E__inference_dense_1_layer_call_and_return_conditional_losses_10944269

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЄ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0Ё
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 

б
#__inference__wrapped_model_10944224
input_1.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource
identityЂ"model/dense/BiasAdd/ReadVariableOpЂ!model/dense/MatMul/ReadVariableOpЂ$model/dense_1/BiasAdd/ReadVariableOpЂ#model/dense_1/MatMul/ReadVariableOpЂ$model/dense_2/BiasAdd/ReadVariableOpЂ#model/dense_2/MatMul/ReadVariableOpЛ
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	*
dtype0
model/dense/MatMulMatMulinput_1)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЙ
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџi
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџР
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0Н
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ё
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџm
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџП
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџМ
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0 
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ*
T0Щ
IdentityIdentitymodel/dense_2/BiasAdd:output:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp: : : :' #
!
_user_specified_name	input_1: : : 

О
$__inference__traced_restore_10944470
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias

identity_7ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5Ђ	RestoreV2ЂRestoreV2_1Й
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*п
valueеBвB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0|
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueBB B B B B B *
dtype0М
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0y
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:}
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0*
_output_shapes
 *
dtype0
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueB
B *
_output_shapes
:Е
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ж

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: т

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2
	RestoreV2	RestoreV22(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_52
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : 
є
Ы
C__inference_model_layer_call_and_return_conditional_losses_10944314
input_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tout
2*
Tin
2*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_10944241*/
_gradient_op_typePartitionedCall-10944247*(
_output_shapes
:џџџџџџџџџЈ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_10944269*/
_gradient_op_typePartitionedCall-10944275*
Tin
2*
Tout
2Љ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*'
_output_shapes
:џџџџџџџџџ*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_10944296**
config_proto

GPU 

CPU2J 8*
Tout
2*/
_gradient_op_typePartitionedCall-10944302*
Tin
2д
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : 
ё
Ъ
C__inference_model_layer_call_and_return_conditional_losses_10944372

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-10944247*(
_output_shapes
:џџџџџџџџџ**
config_proto

GPU 

CPU2J 8*
Tin
2*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_10944241*
Tout
2Ј
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_10944269*/
_gradient_op_typePartitionedCall-10944275*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*(
_output_shapes
:џџџџџџџџџЉ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*/
_gradient_op_typePartitionedCall-10944302*
Tout
2*
Tin
2*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_10944296*'
_output_shapes
:џџџџџџџџџд
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : 
п
Ћ
*__inference_dense_1_layer_call_fn_10944280

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*(
_output_shapes
:џџџџџџџџџ**
config_proto

GPU 

CPU2J 8*/
_gradient_op_typePartitionedCall-10944275*
Tout
2*
Tin
2*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_10944269
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
	
К
(__inference_model_layer_call_fn_10944382
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*'
_output_shapes
:џџџџџџџџџ**
config_proto

GPU 

CPU2J 8*/
_gradient_op_typePartitionedCall-10944373*
Tin
	2*
Tout
2*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_10944372
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : 
є
Ы
C__inference_model_layer_call_and_return_conditional_losses_10944329
input_1(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_10944241*
Tin
2*/
_gradient_op_typePartitionedCall-10944247*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџЈ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_10944269*
Tin
2*/
_gradient_op_typePartitionedCall-10944275*(
_output_shapes
:џџџџџџџџџ*
Tout
2**
config_proto

GPU 

CPU2J 8Љ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tin
2*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_10944296*'
_output_shapes
:џџџџџџџџџ*
Tout
2**
config_proto

GPU 

CPU2J 8*/
_gradient_op_typePartitionedCall-10944302д
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : 
џ
о
E__inference_dense_2_layer_call_and_return_conditional_losses_10944296

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ё
Ъ
C__inference_model_layer_call_and_return_conditional_losses_10944345

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_10944241*(
_output_shapes
:џџџџџџџџџ*
Tout
2*/
_gradient_op_typePartitionedCall-10944247Ј
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_10944269**
config_proto

GPU 

CPU2J 8*/
_gradient_op_typePartitionedCall-10944275*(
_output_shapes
:џџџџџџџџџ*
Tout
2*
Tin
2Љ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*
Tout
2*/
_gradient_op_typePartitionedCall-10944302**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_10944296*
Tin
2*'
_output_shapes
:џџџџџџџџџд
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: : : 
	
К
(__inference_model_layer_call_fn_10944355
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-10944346*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_10944345*
Tout
2*
Tin
	2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : 
ѕ
И
&__inference_signature_wrapper_10944395
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identityЂStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6**
config_proto

GPU 

CPU2J 8*/
_gradient_op_typePartitionedCall-10944386*,
f'R%
#__inference__wrapped_model_10944224*
Tin
	2*'
_output_shapes
:џџџџџџџџџ*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : 
н
Ћ
*__inference_dense_2_layer_call_fn_10944307

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*/
_gradient_op_typePartitionedCall-10944302*
Tout
2*N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_10944296*'
_output_shapes
:џџџџџџџџџ*
Tin
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : "7L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Њ
serving_default
;
input_10
serving_default_input_1:0џџџџџџџџџ;
dense_20
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:зv
 
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
5_default_save_signature
*6&call_and_return_all_conditional_losses
7__call__"Ю
_tf_keras_modelД{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": null, "batch_input_shape": null, "config": {"name": "model", "layers": [{"name": "input_1", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 3], "dtype": "float32", "sparse": false, "name": "input_1"}, "inbound_nodes": []}, {"name": "dense", "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"name": "dense_1", "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"name": "dense_2", "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "input_spec": null, "activity_regularizer": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"name": "input_1", "class_name": "InputLayer", "config": {"batch_input_shape": [null, 3], "dtype": "float32", "sparse": false, "name": "input_1"}, "inbound_nodes": []}, {"name": "dense", "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"name": "dense_1", "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"name": "dense_2", "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
д

	variables
trainable_variables
regularization_losses
	keras_api
*8&call_and_return_all_conditional_losses
9__call__"Х
_tf_keras_layerЋ{"class_name": "InputLayer", "name": "input_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 3], "config": {"batch_input_shape": [null, 3], "dtype": "float32", "sparse": false, "name": "input_1"}, "input_spec": null, "activity_regularizer": null}
Д

kernel
bias
_callable_losses
_eager_losses
	variables
trainable_variables
regularization_losses
	keras_api
*:&call_and_return_all_conditional_losses
;__call__"ц
_tf_keras_layerЬ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "activity_regularizer": null}
К

kernel
bias
_callable_losses
_eager_losses
	variables
trainable_variables
regularization_losses
	keras_api
*<&call_and_return_all_conditional_losses
=__call__"ь
_tf_keras_layerв{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 400, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}, "activity_regularizer": null}
К

kernel
bias
 _callable_losses
!_eager_losses
"	variables
#trainable_variables
$regularization_losses
%	keras_api
*>&call_and_return_all_conditional_losses
?__call__"ь
_tf_keras_layerв{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}, "activity_regularizer": null}
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper


&layers
	variables
'metrics
(non_trainable_variables
trainable_variables
regularization_losses
7__call__
5_default_save_signature
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
,
@serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
љ

)layers

	variables
*metrics
+non_trainable_variables
trainable_variables
regularization_losses
9__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
:	2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
љ

,layers
	variables
-metrics
.non_trainable_variables
trainable_variables
regularization_losses
;__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
љ

/layers
	variables
0metrics
1non_trainable_variables
trainable_variables
regularization_losses
=__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
љ

2layers
"	variables
3metrics
4non_trainable_variables
#trainable_variables
$regularization_losses
?__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
с2о
#__inference__wrapped_model_10944224Ж
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *&Ђ#
!
input_1џџџџџџџџџ
У2Р
C__inference_model_layer_call_and_return_conditional_losses_10944314
C__inference_model_layer_call_and_return_conditional_losses_10944345
C__inference_model_layer_call_and_return_conditional_losses_10944372
C__inference_model_layer_call_and_return_conditional_losses_10944329Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
(__inference_model_layer_call_fn_10944382
(__inference_model_layer_call_fn_10944355Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
у2р
C__inference_dense_layer_call_and_return_conditional_losses_10944241
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш2Х
(__inference_dense_layer_call_fn_10944252
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
х2т
E__inference_dense_1_layer_call_and_return_conditional_losses_10944269
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ъ2Ч
*__inference_dense_1_layer_call_fn_10944280
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
х2т
E__inference_dense_2_layer_call_and_return_conditional_losses_10944296
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ъ2Ч
*__inference_dense_2_layer_call_fn_10944307
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5B3
&__inference_signature_wrapper_10944395input_1
(__inference_model_layer_call_fn_10944355X4Ђ1
*Ђ'
!
input_1џџџџџџџџџ
p 
Њ "џџџџџџџџџЄ
C__inference_dense_layer_call_and_return_conditional_losses_10944241]/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 Ќ
C__inference_model_layer_call_and_return_conditional_losses_10944314e4Ђ1
*Ђ'
!
input_1џџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 Ї
E__inference_dense_1_layer_call_and_return_conditional_losses_10944269^0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 І
E__inference_dense_2_layer_call_and_return_conditional_losses_10944296]0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 Ћ
C__inference_model_layer_call_and_return_conditional_losses_10944345d3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p 
Њ "%Ђ"

0џџџџџџџџџ
 
#__inference__wrapped_model_10944224m0Ђ-
&Ђ#
!
input_1џџџџџџџџџ
Њ "1Њ.
,
dense_2!
dense_2џџџџџџџџџ~
*__inference_dense_2_layer_call_fn_10944307P0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЋ
C__inference_model_layer_call_and_return_conditional_losses_10944372d3Ђ0
)Ђ&
 
inputsџџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 
(__inference_model_layer_call_fn_10944382X4Ђ1
*Ђ'
!
input_1џџџџџџџџџ
p
Њ "џџџџџџџџџ|
(__inference_dense_layer_call_fn_10944252P/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЌ
C__inference_model_layer_call_and_return_conditional_losses_10944329e4Ђ1
*Ђ'
!
input_1џџџџџџџџџ
p
Њ "%Ђ"

0џџџџџџџџџ
 Ђ
&__inference_signature_wrapper_10944395x;Ђ8
Ђ 
1Њ.
,
input_1!
input_1џџџџџџџџџ"1Њ.
,
dense_2!
dense_2џџџџџџџџџ
*__inference_dense_1_layer_call_fn_10944280Q0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ