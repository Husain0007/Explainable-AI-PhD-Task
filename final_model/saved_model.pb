
ŁØ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
®
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
¹
SparseTensorDenseMatMul
	a_indices"Tindices
a_values"T
a_shape	
b"T
product"T"	
Ttype"
Tindicestype0	:
2	"
	adjoint_abool( "
	adjoint_bbool( 
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Į
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
executor_typestring Ø
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58§§	
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
dtype0

Adam/graph_convolution_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/graph_convolution_1/bias/v

3Adam/graph_convolution_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/graph_convolution_1/bias/v*
_output_shapes
:*
dtype0

!Adam/graph_convolution_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/graph_convolution_1/kernel/v

5Adam/graph_convolution_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/graph_convolution_1/kernel/v*
_output_shapes

:*
dtype0

Adam/graph_convolution/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/graph_convolution/bias/v

1Adam/graph_convolution/bias/v/Read/ReadVariableOpReadVariableOpAdam/graph_convolution/bias/v*
_output_shapes
:*
dtype0

Adam/graph_convolution/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!Adam/graph_convolution/kernel/v

3Adam/graph_convolution/kernel/v/Read/ReadVariableOpReadVariableOpAdam/graph_convolution/kernel/v*
_output_shapes
:	*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0

Adam/graph_convolution_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/graph_convolution_1/bias/m

3Adam/graph_convolution_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/graph_convolution_1/bias/m*
_output_shapes
:*
dtype0

!Adam/graph_convolution_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/graph_convolution_1/kernel/m

5Adam/graph_convolution_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/graph_convolution_1/kernel/m*
_output_shapes

:*
dtype0

Adam/graph_convolution/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/graph_convolution/bias/m

1Adam/graph_convolution/bias/m/Read/ReadVariableOpReadVariableOpAdam/graph_convolution/bias/m*
_output_shapes
:*
dtype0

Adam/graph_convolution/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!Adam/graph_convolution/kernel/m

3Adam/graph_convolution/kernel/m/Read/ReadVariableOpReadVariableOpAdam/graph_convolution/kernel/m*
_output_shapes
:	*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0

graph_convolution_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namegraph_convolution_1/bias

,graph_convolution_1/bias/Read/ReadVariableOpReadVariableOpgraph_convolution_1/bias*
_output_shapes
:*
dtype0

graph_convolution_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_namegraph_convolution_1/kernel

.graph_convolution_1/kernel/Read/ReadVariableOpReadVariableOpgraph_convolution_1/kernel*
_output_shapes

:*
dtype0

graph_convolution/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namegraph_convolution/bias
}
*graph_convolution/bias/Read/ReadVariableOpReadVariableOpgraph_convolution/bias*
_output_shapes
:*
dtype0

graph_convolution/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_namegraph_convolution/kernel

,graph_convolution/kernel/Read/ReadVariableOpReadVariableOpgraph_convolution/kernel*
_output_shapes
:	*
dtype0
t
serving_default_input_1Placeholder*$
_output_shapes
:*
dtype0*
shape:
z
serving_default_input_2Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

serving_default_input_3Placeholder*+
_output_shapes
:’’’’’’’’’*
dtype0	* 
shape:’’’’’’’’’
z
serving_default_input_4Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3serving_default_input_4graph_convolution/kernelgraph_convolution/biasgraph_convolution_1/kernelgraph_convolution_1/biasdense/kernel
dense/bias*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_38667

NoOpNoOp
žA
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¹A
valueÆAB¬A B„A
Ŗ
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer-6
layer_with_weights-1
layer-7
	layer-8

layer-9
layer_with_weights-2
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
* 
* 
„
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 

	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses* 
¦
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias*
„
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_random_generator* 
¦
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias*
* 

9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses* 
¦
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias*
.
(0
)1
72
83
E4
F5*
.
(0
)1
72
83
E4
F5*
* 
°
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_3* 
6
Ptrace_0
Qtrace_1
Rtrace_2
Strace_3* 
* 
¼
Titer

Ubeta_1

Vbeta_2
	Wdecay
Xlearning_rate(m)m7m8mEmFm(v )v”7v¢8v£Ev¤Fv„*

Yserving_default* 
* 
* 
* 

Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

_trace_0
`trace_1* 

atrace_0
btrace_1* 
* 
* 
* 
* 

cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

htrace_0* 

itrace_0* 

(0
)1*

(0
)1*
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
hb
VARIABLE_VALUEgraph_convolution/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEgraph_convolution/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

vtrace_0
wtrace_1* 

xtrace_0
ytrace_1* 
* 

70
81*

70
81*
* 

znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

trace_0* 

trace_0* 
jd
VARIABLE_VALUEgraph_convolution_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEgraph_convolution_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

E0
F1*

E0
F1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

trace_0* 

trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUEAdam/graph_convolution/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/graph_convolution/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/graph_convolution_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/graph_convolution_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/graph_convolution/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/graph_convolution/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/graph_convolution_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/graph_convolution_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,graph_convolution/kernel/Read/ReadVariableOp*graph_convolution/bias/Read/ReadVariableOp.graph_convolution_1/kernel/Read/ReadVariableOp,graph_convolution_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp3Adam/graph_convolution/kernel/m/Read/ReadVariableOp1Adam/graph_convolution/bias/m/Read/ReadVariableOp5Adam/graph_convolution_1/kernel/m/Read/ReadVariableOp3Adam/graph_convolution_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp3Adam/graph_convolution/kernel/v/Read/ReadVariableOp1Adam/graph_convolution/bias/v/Read/ReadVariableOp5Adam/graph_convolution_1/kernel/v/Read/ReadVariableOp3Adam/graph_convolution_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_39241
ļ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegraph_convolution/kernelgraph_convolution/biasgraph_convolution_1/kernelgraph_convolution_1/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/graph_convolution/kernel/mAdam/graph_convolution/bias/m!Adam/graph_convolution_1/kernel/mAdam/graph_convolution_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/graph_convolution/kernel/vAdam/graph_convolution/bias/v!Adam/graph_convolution_1/kernel/vAdam/graph_convolution_1/bias/vAdam/dense/kernel/vAdam/dense/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_39332®
õt
š
@__inference_model_layer_call_and_return_conditional_losses_38911
inputs_0
inputs_1
inputs_2	
inputs_3D
1graph_convolution_shape_1_readvariableop_resource:	;
-graph_convolution_add_readvariableop_resource:E
3graph_convolution_1_shape_1_readvariableop_resource:=
/graph_convolution_1_add_readvariableop_resource:9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢$graph_convolution/add/ReadVariableOp¢*graph_convolution/transpose/ReadVariableOp¢&graph_convolution_1/add/ReadVariableOp¢,graph_convolution_1/transpose/ReadVariableOp
"squeezed_sparse_conversion/SqueezeSqueezeinputs_2*
T0	*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
 ~
$squeezed_sparse_conversion/Squeeze_1Squeezeinputs_3*
T0*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
 
3squeezed_sparse_conversion/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"
      
      Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @s
dropout/dropout/MulMulinputs_0dropout/dropout/Const:output:0*
T0*$
_output_shapes
:j
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
    
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*$
_output_shapes
:*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?»
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    °
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*$
_output_shapes
:
graph_convolution/SqueezeSqueeze!dropout/dropout/SelectV2:output:0*
T0* 
_output_shapes
:
*
squeeze_dims
 Å
Agraph_convolution/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0-squeezed_sparse_conversion/Squeeze_1:output:0<squeezed_sparse_conversion/SparseTensor/dense_shape:output:0"graph_convolution/Squeeze:output:0*
T0* 
_output_shapes
:
b
 graph_convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ń
graph_convolution/ExpandDims
ExpandDimsKgraph_convolution/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0)graph_convolution/ExpandDims/dim:output:0*
T0*$
_output_shapes
:l
graph_convolution/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
    u
graph_convolution/unstackUnpack graph_convolution/Shape:output:0*
T0*
_output_shapes
: : : *	
num
(graph_convolution/Shape_1/ReadVariableOpReadVariableOp1graph_convolution_shape_1_readvariableop_resource*
_output_shapes
:	*
dtype0j
graph_convolution/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"     w
graph_convolution/unstack_1Unpack"graph_convolution/Shape_1:output:0*
T0*
_output_shapes
: : *	
nump
graph_convolution/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
graph_convolution/ReshapeReshape%graph_convolution/ExpandDims:output:0(graph_convolution/Reshape/shape:output:0*
T0* 
_output_shapes
:

*graph_convolution/transpose/ReadVariableOpReadVariableOp1graph_convolution_shape_1_readvariableop_resource*
_output_shapes
:	*
dtype0q
 graph_convolution/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ±
graph_convolution/transpose	Transpose2graph_convolution/transpose/ReadVariableOp:value:0)graph_convolution/transpose/perm:output:0*
T0*
_output_shapes
:	r
!graph_convolution/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ’’’’
graph_convolution/Reshape_1Reshapegraph_convolution/transpose:y:0*graph_convolution/Reshape_1/shape:output:0*
T0*
_output_shapes
:	
graph_convolution/MatMulMatMul"graph_convolution/Reshape:output:0$graph_convolution/Reshape_1:output:0*
T0*
_output_shapes
:	v
!graph_convolution/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   
     ¤
graph_convolution/Reshape_2Reshape"graph_convolution/MatMul:product:0*graph_convolution/Reshape_2/shape:output:0*
T0*#
_output_shapes
:
$graph_convolution/add/ReadVariableOpReadVariableOp-graph_convolution_add_readvariableop_resource*
_output_shapes
:*
dtype0 
graph_convolution/addAddV2$graph_convolution/Reshape_2:output:0,graph_convolution/add/ReadVariableOp:value:0*
T0*#
_output_shapes
:g
graph_convolution/ReluRelugraph_convolution/add:z:0*
T0*#
_output_shapes
:\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_1/dropout/MulMul$graph_convolution/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*#
_output_shapes
:l
dropout_1/dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
     
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*#
_output_shapes
:*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ą
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*#
_output_shapes
:^
dropout_1/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ·
dropout_1/dropout/SelectV2SelectV2"dropout_1/dropout/GreaterEqual:z:0dropout_1/dropout/Mul:z:0"dropout_1/dropout/Const_1:output:0*
T0*#
_output_shapes
:
graph_convolution_1/SqueezeSqueeze#dropout_1/dropout/SelectV2:output:0*
T0*
_output_shapes
:	*
squeeze_dims
 Č
Cgraph_convolution_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0-squeezed_sparse_conversion/Squeeze_1:output:0<squeezed_sparse_conversion/SparseTensor/dense_shape:output:0$graph_convolution_1/Squeeze:output:0*
T0*
_output_shapes
:	d
"graph_convolution_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ö
graph_convolution_1/ExpandDims
ExpandDimsMgraph_convolution_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0+graph_convolution_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:n
graph_convolution_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
     y
graph_convolution_1/unstackUnpack"graph_convolution_1/Shape:output:0*
T0*
_output_shapes
: : : *	
num
*graph_convolution_1/Shape_1/ReadVariableOpReadVariableOp3graph_convolution_1_shape_1_readvariableop_resource*
_output_shapes

:*
dtype0l
graph_convolution_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      {
graph_convolution_1/unstack_1Unpack$graph_convolution_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
numr
!graph_convolution_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   „
graph_convolution_1/ReshapeReshape'graph_convolution_1/ExpandDims:output:0*graph_convolution_1/Reshape/shape:output:0*
T0*
_output_shapes
:	 
,graph_convolution_1/transpose/ReadVariableOpReadVariableOp3graph_convolution_1_shape_1_readvariableop_resource*
_output_shapes

:*
dtype0s
"graph_convolution_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ¶
graph_convolution_1/transpose	Transpose4graph_convolution_1/transpose/ReadVariableOp:value:0+graph_convolution_1/transpose/perm:output:0*
T0*
_output_shapes

:t
#graph_convolution_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ’’’’¢
graph_convolution_1/Reshape_1Reshape!graph_convolution_1/transpose:y:0,graph_convolution_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:
graph_convolution_1/MatMulMatMul$graph_convolution_1/Reshape:output:0&graph_convolution_1/Reshape_1:output:0*
T0*
_output_shapes
:	x
#graph_convolution_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   
     Ŗ
graph_convolution_1/Reshape_2Reshape$graph_convolution_1/MatMul:product:0,graph_convolution_1/Reshape_2/shape:output:0*
T0*#
_output_shapes
:
&graph_convolution_1/add/ReadVariableOpReadVariableOp/graph_convolution_1_add_readvariableop_resource*
_output_shapes
:*
dtype0¦
graph_convolution_1/addAddV2&graph_convolution_1/Reshape_2:output:0.graph_convolution_1/add/ReadVariableOp:value:0*
T0*#
_output_shapes
:k
graph_convolution_1/ReluRelugraph_convolution_1/add:z:0*
T0*#
_output_shapes
:^
gather_indices/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :ē
gather_indices/GatherV2GatherV2&graph_convolution_1/Relu:activations:0inputs_1%gather_indices/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:’’’’’’’’’*

batch_dims
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
dense/Tensordot/ShapeShape gather_indices/GatherV2:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : “
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transpose gather_indices/GatherV2:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : æ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’f
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’j
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’²
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp%^graph_convolution/add/ReadVariableOp+^graph_convolution/transpose/ReadVariableOp'^graph_convolution_1/add/ReadVariableOp-^graph_convolution_1/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2L
$graph_convolution/add/ReadVariableOp$graph_convolution/add/ReadVariableOp2X
*graph_convolution/transpose/ReadVariableOp*graph_convolution/transpose/ReadVariableOp2P
&graph_convolution_1/add/ReadVariableOp&graph_convolution_1/add/ReadVariableOp2\
,graph_convolution_1/transpose/ReadVariableOp,graph_convolution_1/transpose/ReadVariableOp:N J
$
_output_shapes
:
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1:UQ
+
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_3
¦

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_38445

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:’’’’’’’’’C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ŗ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:’’’’’’’’’T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
	

:__inference_squeezed_sparse_conversion_layer_call_fn_38948
inputs_0	
inputs_1
identity	

identity_1

identity_2	ä
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2	*
Tout
2		*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_38231`
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:’’’’’’’’’^

Identity_1IdentityPartitionedCall:output:1*
T0*#
_output_shapes
:’’’’’’’’’U

Identity_2IdentityPartitionedCall:output:2*
T0	*
_output_shapes
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’:U Q
+
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1
’
b
)__inference_dropout_1_layer_call_fn_39016

inputs
identity¢StatefulPartitionedCallĆ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_38445s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¤
÷
@__inference_dense_layer_call_and_return_conditional_losses_39134

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : æ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’Z
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’d
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_38481

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @a
dropout/MulMulinputsdropout/Const:output:0*
T0*$
_output_shapes
:b
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
    
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*$
_output_shapes
:*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?£
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*$
_output_shapes
:^
IdentityIdentitydropout/SelectV2:output:0*
T0*$
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
::L H
$
_output_shapes
:
 
_user_specified_nameinputs
¾
¤
L__inference_graph_convolution_layer_call_and_return_conditional_losses_38278

inputs
inputs_1	
inputs_2
inputs_3	2
shape_1_readvariableop_resource:	)
add_readvariableop_resource:
identity¢add/ReadVariableOp¢transpose/ReadVariableOp\
SqueezeSqueezeinputs*
T0* 
_output_shapes
:
*
squeeze_dims
 ­
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3Squeeze:output:0*
T0*(
_output_shapes
:’’’’’’’’’P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : £

ExpandDims
ExpandDims9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:’’’’’’’’’H
ShapeShapeExpandDims:output:0*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numw
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes
:	*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"     S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  r
ReshapeReshapeExpandDims:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’y
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes
:	*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       {
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes
:	`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ’’’’g
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_2/shapePackReshape_2/shape/0:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0r
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’K
ReluReluadd:z:0*
T0*+
_output_shapes
:’’’’’’’’’e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’v
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<::’’’’’’’’’:’’’’’’’’’:: : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:L H
$
_output_shapes
:
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
É
`
B__inference_dropout_layer_call_and_return_conditional_losses_38240

inputs

identity_1K
IdentityIdentityinputs*
T0*$
_output_shapes
:X

Identity_1IdentityIdentity:output:0*
T0*$
_output_shapes
:"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
::L H
$
_output_shapes
:
 
_user_specified_nameinputs
š
«
%__inference_model_layer_call_fn_38707
inputs_0
inputs_1
inputs_2	
inputs_3
unknown:	
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_38548s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
$
_output_shapes
:
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1:UQ
+
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_3
š
«
%__inference_model_layer_call_fn_38687
inputs_0
inputs_1
inputs_2	
inputs_3
unknown:	
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_38380s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
$
_output_shapes
:
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1:UQ
+
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_3
­
E
)__inference_dropout_1_layer_call_fn_39011

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_38289d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
½

Ź
3__inference_graph_convolution_1_layer_call_fn_39045
inputs_0

inputs	
inputs_1
inputs_2	
unknown:
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown	unknown_0*
Tin

2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_graph_convolution_1_layer_call_and_return_conditional_losses_38327s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ē
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_39021

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:’’’’’’’’’_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ē

U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_38231

inputs	
inputs_1
identity	

identity_1

identity_2	c
SqueezeSqueezeinputs*
T0	*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
 c
	Squeeze_1Squeezeinputs_1*
T0*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
 q
SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"
      
      X
IdentityIdentitySqueeze:output:0*
T0	*'
_output_shapes
:’’’’’’’’’X

Identity_1IdentitySqueeze_1:output:0*
T0*#
_output_shapes
:’’’’’’’’’^

Identity_2Identity!SparseTensor/dense_shape:output:0*
T0	*
_output_shapes
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ä
§
%__inference_model_layer_call_fn_38395
input_1
input_2
input_3	
input_4
unknown:	
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_38380s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
$
_output_shapes
:
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:TP
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_3:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_4
Ļ
„
N__inference_graph_convolution_1_layer_call_and_return_conditional_losses_39081
inputs_0

inputs	
inputs_1
inputs_2	1
shape_1_readvariableop_resource:)
add_readvariableop_resource:
identity¢add/ReadVariableOp¢transpose/ReadVariableOpe
SqueezeSqueezeinputs_0*
T0*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
 Ŗ
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2Squeeze:output:0*
T0*'
_output_shapes
:’’’’’’’’’P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ¢

ExpandDims
ExpandDims9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’H
ShapeShapeExpandDims:output:0*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   q
ReshapeReshapeExpandDims:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’x
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ’’’’f
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_2/shapePackReshape_2/shape/0:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0r
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’K
ReluReluadd:z:0*
T0*+
_output_shapes
:’’’’’’’’’e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’v
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:: : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:U Q
+
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ß
`
'__inference_dropout_layer_call_fn_38921

inputs
identity¢StatefulPartitionedCallŗ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_38481l
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*$
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall:L H
$
_output_shapes
:
 
_user_specified_nameinputs
’
s
I__inference_gather_indices_layer_call_and_return_conditional_losses_38340

inputs
inputs_1
identityO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :©
GatherV2GatherV2inputsinputs_1GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:’’’’’’’’’*

batch_dims]
IdentityIdentityGatherV2:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

u
I__inference_gather_indices_layer_call_and_return_conditional_losses_39094
inputs_0
inputs_1
identityO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :«
GatherV2GatherV2inputs_0inputs_1GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:’’’’’’’’’*

batch_dims]
IdentityIdentityGatherV2:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’:U Q
+
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1
Ķ
„
N__inference_graph_convolution_1_layer_call_and_return_conditional_losses_38327

inputs
inputs_1	
inputs_2
inputs_3	1
shape_1_readvariableop_resource:)
add_readvariableop_resource:
identity¢add/ReadVariableOp¢transpose/ReadVariableOpc
SqueezeSqueezeinputs*
T0*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
 ¬
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3Squeeze:output:0*
T0*'
_output_shapes
:’’’’’’’’’P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ¢

ExpandDims
ExpandDims9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:’’’’’’’’’H
ShapeShapeExpandDims:output:0*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numv
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   q
ReshapeReshapeExpandDims:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:’’’’’’’’’x
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes

:*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       z
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ’’’’f
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_2/shapePackReshape_2/shape/0:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0r
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’K
ReluReluadd:z:0*
T0*+
_output_shapes
:’’’’’’’’’e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’v
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*V
_input_shapesE
C:’’’’’’’’’:’’’’’’’’’:’’’’’’’’’:: : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ū=
ļ
__inference__traced_save_39241
file_prefix7
3savev2_graph_convolution_kernel_read_readvariableop5
1savev2_graph_convolution_bias_read_readvariableop9
5savev2_graph_convolution_1_kernel_read_readvariableop7
3savev2_graph_convolution_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop>
:savev2_adam_graph_convolution_kernel_m_read_readvariableop<
8savev2_adam_graph_convolution_bias_m_read_readvariableop@
<savev2_adam_graph_convolution_1_kernel_m_read_readvariableop>
:savev2_adam_graph_convolution_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop>
:savev2_adam_graph_convolution_kernel_v_read_readvariableop<
8savev2_adam_graph_convolution_bias_v_read_readvariableop@
<savev2_adam_graph_convolution_1_kernel_v_read_readvariableop>
:savev2_adam_graph_convolution_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: õ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH„
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_graph_convolution_kernel_read_readvariableop1savev2_graph_convolution_bias_read_readvariableop5savev2_graph_convolution_1_kernel_read_readvariableop3savev2_graph_convolution_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_adam_graph_convolution_kernel_m_read_readvariableop8savev2_adam_graph_convolution_bias_m_read_readvariableop<savev2_adam_graph_convolution_1_kernel_m_read_readvariableop:savev2_adam_graph_convolution_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop:savev2_adam_graph_convolution_kernel_v_read_readvariableop8savev2_adam_graph_convolution_bias_v_read_readvariableop<savev2_adam_graph_convolution_1_kernel_v_read_readvariableop:savev2_adam_graph_convolution_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 **
dtypes 
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:³
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*¾
_input_shapes¬
©: :	:::::: : : : : : : : : :	::::::	:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
¬

É
1__inference_graph_convolution_layer_call_fn_38970
inputs_0

inputs	
inputs_1
inputs_2	
unknown:	
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown	unknown_0*
Tin

2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_graph_convolution_layer_call_and_return_conditional_losses_38278s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<::’’’’’’’’’:’’’’’’’’’:: : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
$
_output_shapes
:
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
¦

c
D__inference_dropout_1_layer_call_and_return_conditional_losses_39033

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:’’’’’’’’’C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ŗ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:’’’’’’’’’T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’e
IdentityIdentitydropout/SelectV2:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_38938

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @a
dropout/MulMulinputsdropout/Const:output:0*
T0*$
_output_shapes
:b
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
    
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*$
_output_shapes
:*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?£
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*$
_output_shapes
:T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*$
_output_shapes
:^
IdentityIdentitydropout/SelectV2:output:0*
T0*$
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
::L H
$
_output_shapes
:
 
_user_specified_nameinputs

C
'__inference_dropout_layer_call_fn_38916

inputs
identityŖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_38240]
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
::L H
$
_output_shapes
:
 
_user_specified_nameinputs
Ą
¤
L__inference_graph_convolution_layer_call_and_return_conditional_losses_39006
inputs_0

inputs	
inputs_1
inputs_2	2
shape_1_readvariableop_resource:	)
add_readvariableop_resource:
identity¢add/ReadVariableOp¢transpose/ReadVariableOp^
SqueezeSqueezeinputs_0*
T0* 
_output_shapes
:
*
squeeze_dims
 «
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2Squeeze:output:0*
T0*(
_output_shapes
:’’’’’’’’’P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : £

ExpandDims
ExpandDims9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:’’’’’’’’’H
ShapeShapeExpandDims:output:0*
T0*
_output_shapes
:Q
unstackUnpackShape:output:0*
T0*
_output_shapes
: : : *	
numw
Shape_1/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes
:	*
dtype0X
Shape_1Const*
_output_shapes
:*
dtype0*
valueB"     S
	unstack_1UnpackShape_1:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  r
ReshapeReshapeExpandDims:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:’’’’’’’’’y
transpose/ReadVariableOpReadVariableOpshape_1_readvariableop_resource*
_output_shapes
:	*
dtype0_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       {
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes
:	`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ’’’’g
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes
:	h
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:’’’’’’’’’S
Reshape_2/shape/0Const*
_output_shapes
: *
dtype0*
value	B :S
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_2/shapePackReshape_2/shape/0:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:v
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:’’’’’’’’’j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype0r
addAddV2Reshape_2:output:0add/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’K
ReluReluadd:z:0*
T0*+
_output_shapes
:’’’’’’’’’e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’v
NoOpNoOp^add/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<::’’’’’’’’’:’’’’’’’’’:: : 2(
add/ReadVariableOpadd/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:N J
$
_output_shapes
:
"
_user_specified_name
inputs_0:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:KG
#
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
Ź

%__inference_dense_layer_call_fn_39103

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallŁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_38373s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¹n

 __inference__wrapped_model_38208
input_1
input_2
input_3	
input_4J
7model_graph_convolution_shape_1_readvariableop_resource:	A
3model_graph_convolution_add_readvariableop_resource:K
9model_graph_convolution_1_shape_1_readvariableop_resource:C
5model_graph_convolution_1_add_readvariableop_resource:?
-model_dense_tensordot_readvariableop_resource:9
+model_dense_biasadd_readvariableop_resource:
identity¢"model/dense/BiasAdd/ReadVariableOp¢$model/dense/Tensordot/ReadVariableOp¢*model/graph_convolution/add/ReadVariableOp¢0model/graph_convolution/transpose/ReadVariableOp¢,model/graph_convolution_1/add/ReadVariableOp¢2model/graph_convolution_1/transpose/ReadVariableOp
(model/squeezed_sparse_conversion/SqueezeSqueezeinput_3*
T0	*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
 
*model/squeezed_sparse_conversion/Squeeze_1Squeezeinput_4*
T0*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
 
9model/squeezed_sparse_conversion/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"
      
      Z
model/dropout/IdentityIdentityinput_1*
T0*$
_output_shapes
:
model/graph_convolution/SqueezeSqueezemodel/dropout/Identity:output:0*
T0* 
_output_shapes
:
*
squeeze_dims
 ć
Gmodel/graph_convolution/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul1model/squeezed_sparse_conversion/Squeeze:output:03model/squeezed_sparse_conversion/Squeeze_1:output:0Bmodel/squeezed_sparse_conversion/SparseTensor/dense_shape:output:0(model/graph_convolution/Squeeze:output:0*
T0* 
_output_shapes
:
h
&model/graph_convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ć
"model/graph_convolution/ExpandDims
ExpandDimsQmodel/graph_convolution/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0/model/graph_convolution/ExpandDims/dim:output:0*
T0*$
_output_shapes
:r
model/graph_convolution/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
    
model/graph_convolution/unstackUnpack&model/graph_convolution/Shape:output:0*
T0*
_output_shapes
: : : *	
num§
.model/graph_convolution/Shape_1/ReadVariableOpReadVariableOp7model_graph_convolution_shape_1_readvariableop_resource*
_output_shapes
:	*
dtype0p
model/graph_convolution/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"     
!model/graph_convolution/unstack_1Unpack(model/graph_convolution/Shape_1:output:0*
T0*
_output_shapes
: : *	
numv
%model/graph_convolution/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’  ²
model/graph_convolution/ReshapeReshape+model/graph_convolution/ExpandDims:output:0.model/graph_convolution/Reshape/shape:output:0*
T0* 
_output_shapes
:
©
0model/graph_convolution/transpose/ReadVariableOpReadVariableOp7model_graph_convolution_shape_1_readvariableop_resource*
_output_shapes
:	*
dtype0w
&model/graph_convolution/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Ć
!model/graph_convolution/transpose	Transpose8model/graph_convolution/transpose/ReadVariableOp:value:0/model/graph_convolution/transpose/perm:output:0*
T0*
_output_shapes
:	x
'model/graph_convolution/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ’’’’Æ
!model/graph_convolution/Reshape_1Reshape%model/graph_convolution/transpose:y:00model/graph_convolution/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Ø
model/graph_convolution/MatMulMatMul(model/graph_convolution/Reshape:output:0*model/graph_convolution/Reshape_1:output:0*
T0*
_output_shapes
:	|
'model/graph_convolution/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   
     ¶
!model/graph_convolution/Reshape_2Reshape(model/graph_convolution/MatMul:product:00model/graph_convolution/Reshape_2/shape:output:0*
T0*#
_output_shapes
:
*model/graph_convolution/add/ReadVariableOpReadVariableOp3model_graph_convolution_add_readvariableop_resource*
_output_shapes
:*
dtype0²
model/graph_convolution/addAddV2*model/graph_convolution/Reshape_2:output:02model/graph_convolution/add/ReadVariableOp:value:0*
T0*#
_output_shapes
:s
model/graph_convolution/ReluRelumodel/graph_convolution/add:z:0*
T0*#
_output_shapes
:~
model/dropout_1/IdentityIdentity*model/graph_convolution/Relu:activations:0*
T0*#
_output_shapes
:
!model/graph_convolution_1/SqueezeSqueeze!model/dropout_1/Identity:output:0*
T0*
_output_shapes
:	*
squeeze_dims
 ę
Imodel/graph_convolution_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul1model/squeezed_sparse_conversion/Squeeze:output:03model/squeezed_sparse_conversion/Squeeze_1:output:0Bmodel/squeezed_sparse_conversion/SparseTensor/dense_shape:output:0*model/graph_convolution_1/Squeeze:output:0*
T0*
_output_shapes
:	j
(model/graph_convolution_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : č
$model/graph_convolution_1/ExpandDims
ExpandDimsSmodel/graph_convolution_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:01model/graph_convolution_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:t
model/graph_convolution_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
     
!model/graph_convolution_1/unstackUnpack(model/graph_convolution_1/Shape:output:0*
T0*
_output_shapes
: : : *	
numŖ
0model/graph_convolution_1/Shape_1/ReadVariableOpReadVariableOp9model_graph_convolution_1_shape_1_readvariableop_resource*
_output_shapes

:*
dtype0r
!model/graph_convolution_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      
#model/graph_convolution_1/unstack_1Unpack*model/graph_convolution_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
numx
'model/graph_convolution_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   ·
!model/graph_convolution_1/ReshapeReshape-model/graph_convolution_1/ExpandDims:output:00model/graph_convolution_1/Reshape/shape:output:0*
T0*
_output_shapes
:	¬
2model/graph_convolution_1/transpose/ReadVariableOpReadVariableOp9model_graph_convolution_1_shape_1_readvariableop_resource*
_output_shapes

:*
dtype0y
(model/graph_convolution_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Č
#model/graph_convolution_1/transpose	Transpose:model/graph_convolution_1/transpose/ReadVariableOp:value:01model/graph_convolution_1/transpose/perm:output:0*
T0*
_output_shapes

:z
)model/graph_convolution_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ’’’’“
#model/graph_convolution_1/Reshape_1Reshape'model/graph_convolution_1/transpose:y:02model/graph_convolution_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:®
 model/graph_convolution_1/MatMulMatMul*model/graph_convolution_1/Reshape:output:0,model/graph_convolution_1/Reshape_1:output:0*
T0*
_output_shapes
:	~
)model/graph_convolution_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   
     ¼
#model/graph_convolution_1/Reshape_2Reshape*model/graph_convolution_1/MatMul:product:02model/graph_convolution_1/Reshape_2/shape:output:0*
T0*#
_output_shapes
:
,model/graph_convolution_1/add/ReadVariableOpReadVariableOp5model_graph_convolution_1_add_readvariableop_resource*
_output_shapes
:*
dtype0ø
model/graph_convolution_1/addAddV2,model/graph_convolution_1/Reshape_2:output:04model/graph_convolution_1/add/ReadVariableOp:value:0*
T0*#
_output_shapes
:w
model/graph_convolution_1/ReluRelu!model/graph_convolution_1/add:z:0*
T0*#
_output_shapes
:d
"model/gather_indices/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :ų
model/gather_indices/GatherV2GatherV2,model/graph_convolution_1/Relu:activations:0input_2+model/gather_indices/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:’’’’’’’’’*

batch_dims
$model/dense/Tensordot/ReadVariableOpReadVariableOp-model_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0d
model/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
model/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       q
model/dense/Tensordot/ShapeShape&model/gather_indices/GatherV2:output:0*
T0*
_output_shapes
:e
#model/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ė
model/dense/Tensordot/GatherV2GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/free:output:0,model/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
%model/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ļ
 model/dense/Tensordot/GatherV2_1GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/axes:output:0.model/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
model/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense/Tensordot/ProdProd'model/dense/Tensordot/GatherV2:output:0$model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: g
model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense/Tensordot/Prod_1Prod)model/dense/Tensordot/GatherV2_1:output:0&model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ģ
model/dense/Tensordot/concatConcatV2#model/dense/Tensordot/free:output:0#model/dense/Tensordot/axes:output:0*model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
model/dense/Tensordot/stackPack#model/dense/Tensordot/Prod:output:0%model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:±
model/dense/Tensordot/transpose	Transpose&model/gather_indices/GatherV2:output:0%model/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’®
model/dense/Tensordot/ReshapeReshape#model/dense/Tensordot/transpose:y:0$model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’®
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’g
model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:e
#model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
model/dense/Tensordot/concat_1ConcatV2'model/dense/Tensordot/GatherV2:output:0&model/dense/Tensordot/Const_2:output:0,model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:§
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0'model/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’r
model/dense/SoftmaxSoftmaxmodel/dense/BiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’p
IdentityIdentitymodel/dense/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’Ö
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOp+^model/graph_convolution/add/ReadVariableOp1^model/graph_convolution/transpose/ReadVariableOp-^model/graph_convolution_1/add/ReadVariableOp3^model/graph_convolution_1/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2X
*model/graph_convolution/add/ReadVariableOp*model/graph_convolution/add/ReadVariableOp2d
0model/graph_convolution/transpose/ReadVariableOp0model/graph_convolution/transpose/ReadVariableOp2\
,model/graph_convolution_1/add/ReadVariableOp,model/graph_convolution_1/add/ReadVariableOp2h
2model/graph_convolution_1/transpose/ReadVariableOp2model/graph_convolution_1/transpose/ReadVariableOp:M I
$
_output_shapes
:
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:TP
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_3:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_4
ā&
Ŗ
@__inference_model_layer_call_and_return_conditional_losses_38639
input_1
input_2
input_3	
input_4*
graph_convolution_38621:	%
graph_convolution_38623:+
graph_convolution_1_38627:'
graph_convolution_1_38629:
dense_38633:
dense_38635:
identity¢dense/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢)graph_convolution/StatefulPartitionedCall¢+graph_convolution_1/StatefulPartitionedCallż
*squeezed_sparse_conversion/PartitionedCallPartitionedCallinput_3input_4*
Tin
2	*
Tout
2		*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_38231Ć
dropout/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_38481Ł
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_convolution_38621graph_convolution_38623*
Tin

2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_graph_convolution_layer_call_and_return_conditional_losses_38278
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_38445ć
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_convolution_1_38627graph_convolution_1_38629*
Tin

2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_graph_convolution_1_layer_call_and_return_conditional_losses_38327’
gather_indices/PartitionedCallPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_gather_indices_layer_call_and_return_conditional_losses_38340
dense/StatefulPartitionedCallStatefulPartitionedCall'gather_indices/PartitionedCall:output:0dense_38633dense_38635*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_38373y
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’
NoOpNoOp^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall:M I
$
_output_shapes
:
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:TP
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_3:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_4
ų#
ä
@__inference_model_layer_call_and_return_conditional_losses_38611
input_1
input_2
input_3	
input_4*
graph_convolution_38593:	%
graph_convolution_38595:+
graph_convolution_1_38599:'
graph_convolution_1_38601:
dense_38605:
dense_38607:
identity¢dense/StatefulPartitionedCall¢)graph_convolution/StatefulPartitionedCall¢+graph_convolution_1/StatefulPartitionedCallż
*squeezed_sparse_conversion/PartitionedCallPartitionedCallinput_3input_4*
Tin
2	*
Tout
2		*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_38231³
dropout/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_38240Ń
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_convolution_38593graph_convolution_38595*
Tin

2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_graph_convolution_layer_call_and_return_conditional_losses_38278é
dropout_1/PartitionedCallPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_38289Ū
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_convolution_1_38599graph_convolution_1_38601*
Tin

2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_graph_convolution_1_layer_call_and_return_conditional_losses_38327’
gather_indices/PartitionedCallPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_gather_indices_layer_call_and_return_conditional_losses_38340
dense/StatefulPartitionedCallStatefulPartitionedCall'gather_indices/PartitionedCall:output:0dense_38605dense_38607*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_38373y
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’Ą
NoOpNoOp^dense/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall:M I
$
_output_shapes
:
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:TP
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_3:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_4
ā&
¬
@__inference_model_layer_call_and_return_conditional_losses_38548

inputs
inputs_1
inputs_2	
inputs_3*
graph_convolution_38530:	%
graph_convolution_38532:+
graph_convolution_1_38536:'
graph_convolution_1_38538:
dense_38542:
dense_38544:
identity¢dense/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢)graph_convolution/StatefulPartitionedCall¢+graph_convolution_1/StatefulPartitionedCall’
*squeezed_sparse_conversion/PartitionedCallPartitionedCallinputs_2inputs_3*
Tin
2	*
Tout
2		*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_38231Ā
dropout/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_38481Ł
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_convolution_38530graph_convolution_38532*
Tin

2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_graph_convolution_layer_call_and_return_conditional_losses_38278
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_38445ć
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_convolution_1_38536graph_convolution_1_38538*
Tin

2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_graph_convolution_1_layer_call_and_return_conditional_losses_38327
gather_indices/PartitionedCallPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_gather_indices_layer_call_and_return_conditional_losses_38340
dense/StatefulPartitionedCallStatefulPartitionedCall'gather_indices/PartitionedCall:output:0dense_38542dense_38544*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_38373y
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’
NoOpNoOp^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall:L H
$
_output_shapes
:
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:SO
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ų#
ę
@__inference_model_layer_call_and_return_conditional_losses_38380

inputs
inputs_1
inputs_2	
inputs_3*
graph_convolution_38279:	%
graph_convolution_38281:+
graph_convolution_1_38328:'
graph_convolution_1_38330:
dense_38374:
dense_38376:
identity¢dense/StatefulPartitionedCall¢)graph_convolution/StatefulPartitionedCall¢+graph_convolution_1/StatefulPartitionedCall’
*squeezed_sparse_conversion/PartitionedCallPartitionedCallinputs_2inputs_3*
Tin
2	*
Tout
2		*
_collective_manager_ids
 *<
_output_shapes*
(:’’’’’’’’’:’’’’’’’’’:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_38231²
dropout/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_38240Ń
)graph_convolution/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_convolution_38279graph_convolution_38281*
Tin

2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_graph_convolution_layer_call_and_return_conditional_losses_38278é
dropout_1/PartitionedCallPartitionedCall2graph_convolution/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_38289Ū
+graph_convolution_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:03squeezed_sparse_conversion/PartitionedCall:output:13squeezed_sparse_conversion/PartitionedCall:output:2graph_convolution_1_38328graph_convolution_1_38330*
Tin

2		*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_graph_convolution_1_layer_call_and_return_conditional_losses_38327
gather_indices/PartitionedCallPartitionedCall4graph_convolution_1/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_gather_indices_layer_call_and_return_conditional_losses_38340
dense/StatefulPartitionedCallStatefulPartitionedCall'gather_indices/PartitionedCall:output:0dense_38374dense_38376*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_38373y
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’Ą
NoOpNoOp^dense/StatefulPartitionedCall*^graph_convolution/StatefulPartitionedCall,^graph_convolution_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2V
)graph_convolution/StatefulPartitionedCall)graph_convolution/StatefulPartitionedCall2Z
+graph_convolution_1/StatefulPartitionedCall+graph_convolution_1/StatefulPartitionedCall:L H
$
_output_shapes
:
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:SO
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:OK
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
É
`
B__inference_dropout_layer_call_and_return_conditional_losses_38926

inputs

identity_1K
IdentityIdentityinputs*
T0*$
_output_shapes
:X

Identity_1IdentityIdentity:output:0*
T0*$
_output_shapes
:"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
::L H
$
_output_shapes
:
 
_user_specified_nameinputs
ļ
”
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_38958
inputs_0	
inputs_1
identity	

identity_1

identity_2	e
SqueezeSqueezeinputs_0*
T0	*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
 c
	Squeeze_1Squeezeinputs_1*
T0*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
 q
SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"
      
      X
IdentityIdentitySqueeze:output:0*
T0	*'
_output_shapes
:’’’’’’’’’X

Identity_1IdentitySqueeze_1:output:0*
T0*#
_output_shapes
:’’’’’’’’’^

Identity_2Identity!SparseTensor/dense_shape:output:0*
T0	*
_output_shapes
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’:U Q
+
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1
ße
š
@__inference_model_layer_call_and_return_conditional_losses_38802
inputs_0
inputs_1
inputs_2	
inputs_3D
1graph_convolution_shape_1_readvariableop_resource:	;
-graph_convolution_add_readvariableop_resource:E
3graph_convolution_1_shape_1_readvariableop_resource:=
/graph_convolution_1_add_readvariableop_resource:9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/Tensordot/ReadVariableOp¢$graph_convolution/add/ReadVariableOp¢*graph_convolution/transpose/ReadVariableOp¢&graph_convolution_1/add/ReadVariableOp¢,graph_convolution_1/transpose/ReadVariableOp
"squeezed_sparse_conversion/SqueezeSqueezeinputs_2*
T0	*'
_output_shapes
:’’’’’’’’’*
squeeze_dims
 ~
$squeezed_sparse_conversion/Squeeze_1Squeezeinputs_3*
T0*#
_output_shapes
:’’’’’’’’’*
squeeze_dims
 
3squeezed_sparse_conversion/SparseTensor/dense_shapeConst*
_output_shapes
:*
dtype0	*%
valueB	"
      
      U
dropout/IdentityIdentityinputs_0*
T0*$
_output_shapes
:
graph_convolution/SqueezeSqueezedropout/Identity:output:0*
T0* 
_output_shapes
:
*
squeeze_dims
 Å
Agraph_convolution/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0-squeezed_sparse_conversion/Squeeze_1:output:0<squeezed_sparse_conversion/SparseTensor/dense_shape:output:0"graph_convolution/Squeeze:output:0*
T0* 
_output_shapes
:
b
 graph_convolution/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ń
graph_convolution/ExpandDims
ExpandDimsKgraph_convolution/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0)graph_convolution/ExpandDims/dim:output:0*
T0*$
_output_shapes
:l
graph_convolution/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
    u
graph_convolution/unstackUnpack graph_convolution/Shape:output:0*
T0*
_output_shapes
: : : *	
num
(graph_convolution/Shape_1/ReadVariableOpReadVariableOp1graph_convolution_shape_1_readvariableop_resource*
_output_shapes
:	*
dtype0j
graph_convolution/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"     w
graph_convolution/unstack_1Unpack"graph_convolution/Shape_1:output:0*
T0*
_output_shapes
: : *	
nump
graph_convolution/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   
graph_convolution/ReshapeReshape%graph_convolution/ExpandDims:output:0(graph_convolution/Reshape/shape:output:0*
T0* 
_output_shapes
:

*graph_convolution/transpose/ReadVariableOpReadVariableOp1graph_convolution_shape_1_readvariableop_resource*
_output_shapes
:	*
dtype0q
 graph_convolution/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ±
graph_convolution/transpose	Transpose2graph_convolution/transpose/ReadVariableOp:value:0)graph_convolution/transpose/perm:output:0*
T0*
_output_shapes
:	r
!graph_convolution/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"  ’’’’
graph_convolution/Reshape_1Reshapegraph_convolution/transpose:y:0*graph_convolution/Reshape_1/shape:output:0*
T0*
_output_shapes
:	
graph_convolution/MatMulMatMul"graph_convolution/Reshape:output:0$graph_convolution/Reshape_1:output:0*
T0*
_output_shapes
:	v
!graph_convolution/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   
     ¤
graph_convolution/Reshape_2Reshape"graph_convolution/MatMul:product:0*graph_convolution/Reshape_2/shape:output:0*
T0*#
_output_shapes
:
$graph_convolution/add/ReadVariableOpReadVariableOp-graph_convolution_add_readvariableop_resource*
_output_shapes
:*
dtype0 
graph_convolution/addAddV2$graph_convolution/Reshape_2:output:0,graph_convolution/add/ReadVariableOp:value:0*
T0*#
_output_shapes
:g
graph_convolution/ReluRelugraph_convolution/add:z:0*
T0*#
_output_shapes
:r
dropout_1/IdentityIdentity$graph_convolution/Relu:activations:0*
T0*#
_output_shapes
:
graph_convolution_1/SqueezeSqueezedropout_1/Identity:output:0*
T0*
_output_shapes
:	*
squeeze_dims
 Č
Cgraph_convolution_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMul+squeezed_sparse_conversion/Squeeze:output:0-squeezed_sparse_conversion/Squeeze_1:output:0<squeezed_sparse_conversion/SparseTensor/dense_shape:output:0$graph_convolution_1/Squeeze:output:0*
T0*
_output_shapes
:	d
"graph_convolution_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ö
graph_convolution_1/ExpandDims
ExpandDimsMgraph_convolution_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0+graph_convolution_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:n
graph_convolution_1/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"   
     y
graph_convolution_1/unstackUnpack"graph_convolution_1/Shape:output:0*
T0*
_output_shapes
: : : *	
num
*graph_convolution_1/Shape_1/ReadVariableOpReadVariableOp3graph_convolution_1_shape_1_readvariableop_resource*
_output_shapes

:*
dtype0l
graph_convolution_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      {
graph_convolution_1/unstack_1Unpack$graph_convolution_1/Shape_1:output:0*
T0*
_output_shapes
: : *	
numr
!graph_convolution_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"’’’’   „
graph_convolution_1/ReshapeReshape'graph_convolution_1/ExpandDims:output:0*graph_convolution_1/Reshape/shape:output:0*
T0*
_output_shapes
:	 
,graph_convolution_1/transpose/ReadVariableOpReadVariableOp3graph_convolution_1_shape_1_readvariableop_resource*
_output_shapes

:*
dtype0s
"graph_convolution_1/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ¶
graph_convolution_1/transpose	Transpose4graph_convolution_1/transpose/ReadVariableOp:value:0+graph_convolution_1/transpose/perm:output:0*
T0*
_output_shapes

:t
#graph_convolution_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ’’’’¢
graph_convolution_1/Reshape_1Reshape!graph_convolution_1/transpose:y:0,graph_convolution_1/Reshape_1/shape:output:0*
T0*
_output_shapes

:
graph_convolution_1/MatMulMatMul$graph_convolution_1/Reshape:output:0&graph_convolution_1/Reshape_1:output:0*
T0*
_output_shapes
:	x
#graph_convolution_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   
     Ŗ
graph_convolution_1/Reshape_2Reshape$graph_convolution_1/MatMul:product:0,graph_convolution_1/Reshape_2/shape:output:0*
T0*#
_output_shapes
:
&graph_convolution_1/add/ReadVariableOpReadVariableOp/graph_convolution_1_add_readvariableop_resource*
_output_shapes
:*
dtype0¦
graph_convolution_1/addAddV2&graph_convolution_1/Reshape_2:output:0.graph_convolution_1/add/ReadVariableOp:value:0*
T0*#
_output_shapes
:k
graph_convolution_1/ReluRelugraph_convolution_1/add:z:0*
T0*#
_output_shapes
:^
gather_indices/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :ē
gather_indices/GatherV2GatherV2&graph_convolution_1/Relu:activations:0inputs_1%gather_indices/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:’’’’’’’’’*

batch_dims
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       e
dense/Tensordot/ShapeShape gather_indices/GatherV2:output:0*
T0*
_output_shapes
:_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : “
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transpose gather_indices/GatherV2:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’a
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : æ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’f
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’j
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’²
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp%^graph_convolution/add/ReadVariableOp+^graph_convolution/transpose/ReadVariableOp'^graph_convolution_1/add/ReadVariableOp-^graph_convolution_1/transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2L
$graph_convolution/add/ReadVariableOp$graph_convolution/add/ReadVariableOp2X
*graph_convolution/transpose/ReadVariableOp*graph_convolution/transpose/ReadVariableOp2P
&graph_convolution_1/add/ReadVariableOp&graph_convolution_1/add/ReadVariableOp2\
,graph_convolution_1/transpose/ReadVariableOp,graph_convolution_1/transpose/ReadVariableOp:N J
$
_output_shapes
:
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1:UQ
+
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_3
÷v
Ś
!__inference__traced_restore_39332
file_prefix<
)assignvariableop_graph_convolution_kernel:	7
)assignvariableop_1_graph_convolution_bias:?
-assignvariableop_2_graph_convolution_1_kernel:9
+assignvariableop_3_graph_convolution_1_bias:1
assignvariableop_4_dense_kernel:+
assignvariableop_5_dense_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: #
assignvariableop_13_total: #
assignvariableop_14_count: F
3assignvariableop_15_adam_graph_convolution_kernel_m:	?
1assignvariableop_16_adam_graph_convolution_bias_m:G
5assignvariableop_17_adam_graph_convolution_1_kernel_m:A
3assignvariableop_18_adam_graph_convolution_1_bias_m:9
'assignvariableop_19_adam_dense_kernel_m:3
%assignvariableop_20_adam_dense_bias_m:F
3assignvariableop_21_adam_graph_convolution_kernel_v:	?
1assignvariableop_22_adam_graph_convolution_bias_v:G
5assignvariableop_23_adam_graph_convolution_1_kernel_v:A
3assignvariableop_24_adam_graph_convolution_1_bias_v:9
'assignvariableop_25_adam_dense_kernel_v:3
%assignvariableop_26_adam_dense_bias_v:
identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ų
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHØ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B «
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOpAssignVariableOp)assignvariableop_graph_convolution_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ą
AssignVariableOp_1AssignVariableOp)assignvariableop_1_graph_convolution_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ä
AssignVariableOp_2AssignVariableOp-assignvariableop_2_graph_convolution_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ā
AssignVariableOp_3AssignVariableOp+assignvariableop_3_graph_convolution_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:³
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:æ
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:“
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ģ
AssignVariableOp_15AssignVariableOp3assignvariableop_15_adam_graph_convolution_kernel_mIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ź
AssignVariableOp_16AssignVariableOp1assignvariableop_16_adam_graph_convolution_bias_mIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ī
AssignVariableOp_17AssignVariableOp5assignvariableop_17_adam_graph_convolution_1_kernel_mIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ģ
AssignVariableOp_18AssignVariableOp3assignvariableop_18_adam_graph_convolution_1_bias_mIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ą
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_mIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_mIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ģ
AssignVariableOp_21AssignVariableOp3assignvariableop_21_adam_graph_convolution_kernel_vIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ź
AssignVariableOp_22AssignVariableOp1assignvariableop_22_adam_graph_convolution_bias_vIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ī
AssignVariableOp_23AssignVariableOp5assignvariableop_23_adam_graph_convolution_1_kernel_vIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ģ
AssignVariableOp_24AssignVariableOp3assignvariableop_24_adam_graph_convolution_1_bias_vIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ą
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_vIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_vIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ”
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ä
§
%__inference_model_layer_call_fn_38583
input_1
input_2
input_3	
input_4
unknown:	
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_38548s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
$
_output_shapes
:
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:TP
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_3:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_4
¼
Z
.__inference_gather_indices_layer_call_fn_39087
inputs_0
inputs_1
identityÅ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_gather_indices_layer_call_and_return_conditional_losses_38340d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:’’’’’’’’’:’’’’’’’’’:U Q
+
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1
¤
÷
@__inference_dense_layer_call_and_return_conditional_losses_38373

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : æ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:’’’’’’’’’
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:’’’’’’’’’r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:’’’’’’’’’Z
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:’’’’’’’’’d
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:’’’’’’’’’: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ā
„
#__inference_signature_wrapper_38667
input_1
input_2
input_3	
input_4
unknown:	
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2
	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:’’’’’’’’’*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_38208s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:’’’’’’’’’`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y::’’’’’’’’’:’’’’’’’’’:’’’’’’’’’: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
$
_output_shapes
:
!
_user_specified_name	input_1:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_2:TP
+
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_3:PL
'
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_4
ē
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_38289

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:’’’’’’’’’_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:’’’’’’’’’"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:’’’’’’’’’:S O
+
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ä
serving_defaultŠ
8
input_1-
serving_default_input_1:0
;
input_20
serving_default_input_2:0’’’’’’’’’
?
input_34
serving_default_input_3:0	’’’’’’’’’
;
input_40
serving_default_input_4:0’’’’’’’’’=
dense4
StatefulPartitionedCall:0’’’’’’’’’tensorflow/serving/predict:ņŌ
Į
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer-6
layer_with_weights-1
layer-7
	layer-8

layer-9
layer_with_weights-2
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
¼
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
„
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
»
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias"
_tf_keras_layer
¼
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses
0_random_generator"
_tf_keras_layer
»
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
"
_tf_keras_input_layer
„
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
»
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias"
_tf_keras_layer
J
(0
)1
72
83
E4
F5"
trackable_list_wrapper
J
(0
)1
72
83
E4
F5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ź
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
É
Ltrace_0
Mtrace_1
Ntrace_2
Otrace_32Ž
%__inference_model_layer_call_fn_38395
%__inference_model_layer_call_fn_38687
%__inference_model_layer_call_fn_38707
%__inference_model_layer_call_fn_38583æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zLtrace_0zMtrace_1zNtrace_2zOtrace_3
µ
Ptrace_0
Qtrace_1
Rtrace_2
Strace_32Ź
@__inference_model_layer_call_and_return_conditional_losses_38802
@__inference_model_layer_call_and_return_conditional_losses_38911
@__inference_model_layer_call_and_return_conditional_losses_38611
@__inference_model_layer_call_and_return_conditional_losses_38639æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zPtrace_0zQtrace_1zRtrace_2zStrace_3
ęBć
 __inference__wrapped_model_38208input_1input_2input_3input_4"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ė
Titer

Ubeta_1

Vbeta_2
	Wdecay
Xlearning_rate(m)m7m8mEmFm(v )v”7v¢8v£Ev¤Fv„"
	optimizer
,
Yserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
æ
_trace_0
`trace_12
'__inference_dropout_layer_call_fn_38916
'__inference_dropout_layer_call_fn_38921³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 z_trace_0z`trace_1
õ
atrace_0
btrace_12¾
B__inference_dropout_layer_call_and_return_conditional_losses_38926
B__inference_dropout_layer_call_and_return_conditional_losses_38938³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zatrace_0zbtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
ž
htrace_02į
:__inference_squeezed_sparse_conversion_layer_call_fn_38948¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 zhtrace_0

itrace_02ü
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_38958¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 zitrace_0
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
õ
otrace_02Ų
1__inference_graph_convolution_layer_call_fn_38970¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 zotrace_0

ptrace_02ó
L__inference_graph_convolution_layer_call_and_return_conditional_losses_39006¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 zptrace_0
+:)	2graph_convolution/kernel
$:"2graph_convolution/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
Ć
vtrace_0
wtrace_12
)__inference_dropout_1_layer_call_fn_39011
)__inference_dropout_1_layer_call_fn_39016³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zvtrace_0zwtrace_1
ł
xtrace_0
ytrace_12Ā
D__inference_dropout_1_layer_call_and_return_conditional_losses_39021
D__inference_dropout_1_layer_call_and_return_conditional_losses_39033³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 zxtrace_0zytrace_1
"
_generic_user_object
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
­
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
÷
trace_02Ś
3__inference_graph_convolution_1_layer_call_fn_39045¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 ztrace_0

trace_02õ
N__inference_graph_convolution_1_layer_call_and_return_conditional_losses_39081¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 ztrace_0
,:*2graph_convolution_1/kernel
&:$2graph_convolution_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
ō
trace_02Õ
.__inference_gather_indices_layer_call_fn_39087¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 ztrace_0

trace_02š
I__inference_gather_indices_layer_call_and_return_conditional_losses_39094¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 ztrace_0
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
ė
trace_02Ģ
%__inference_dense_layer_call_fn_39103¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 ztrace_0

trace_02ē
@__inference_dense_layer_call_and_return_conditional_losses_39134¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 ztrace_0
:2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
%__inference_model_layer_call_fn_38395input_1input_2input_3input_4"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
%__inference_model_layer_call_fn_38687inputs_0inputs_1inputs_2inputs_3"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
%__inference_model_layer_call_fn_38707inputs_0inputs_1inputs_2inputs_3"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
%__inference_model_layer_call_fn_38583input_1input_2input_3input_4"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
±B®
@__inference_model_layer_call_and_return_conditional_losses_38802inputs_0inputs_1inputs_2inputs_3"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
±B®
@__inference_model_layer_call_and_return_conditional_losses_38911inputs_0inputs_1inputs_2inputs_3"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
­BŖ
@__inference_model_layer_call_and_return_conditional_losses_38611input_1input_2input_3input_4"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
­BŖ
@__inference_model_layer_call_and_return_conditional_losses_38639input_1input_2input_3input_4"æ
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ćBą
#__inference_signature_wrapper_38667input_1input_2input_3input_4"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ģBé
'__inference_dropout_layer_call_fn_38916inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ģBé
'__inference_dropout_layer_call_fn_38921inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_38926inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_38938inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
śB÷
:__inference_squeezed_sparse_conversion_layer_call_fn_38948inputs_0inputs_1"¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 
B
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_38958inputs_0inputs_1"¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
1__inference_graph_convolution_layer_call_fn_38970inputs_0inputsinputs_1inputs_2"¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 
B
L__inference_graph_convolution_layer_call_and_return_conditional_losses_39006inputs_0inputsinputs_1inputs_2"¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
īBė
)__inference_dropout_1_layer_call_fn_39011inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
īBė
)__inference_dropout_1_layer_call_fn_39016inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_39021inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_39033inputs"³
Ŗ²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_graph_convolution_1_layer_call_fn_39045inputs_0inputsinputs_1inputs_2"¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 
 B
N__inference_graph_convolution_1_layer_call_and_return_conditional_losses_39081inputs_0inputsinputs_1inputs_2"¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
īBė
.__inference_gather_indices_layer_call_fn_39087inputs_0inputs_1"¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 
B
I__inference_gather_indices_layer_call_and_return_conditional_losses_39094inputs_0inputs_1"¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŁBÖ
%__inference_dense_layer_call_fn_39103inputs"¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 
ōBń
@__inference_dense_layer_call_and_return_conditional_losses_39134inputs"¢
²
FullArgSpec
args
jself
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
annotationsŖ *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0:.	2Adam/graph_convolution/kernel/m
):'2Adam/graph_convolution/bias/m
1:/2!Adam/graph_convolution_1/kernel/m
+:)2Adam/graph_convolution_1/bias/m
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
0:.	2Adam/graph_convolution/kernel/v
):'2Adam/graph_convolution/bias/v
1:/2!Adam/graph_convolution_1/kernel/v
+:)2Adam/graph_convolution_1/bias/v
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
 __inference__wrapped_model_38208ā()78EF¤¢ 
¢


input_1
!
input_2’’’’’’’’’
%"
input_3’’’’’’’’’	
!
input_4’’’’’’’’’
Ŗ "1Ŗ.
,
dense# 
dense’’’’’’’’’Æ
@__inference_dense_layer_call_and_return_conditional_losses_39134kEF3¢0
)¢&
$!
inputs’’’’’’’’’
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 
%__inference_dense_layer_call_fn_39103`EF3¢0
)¢&
$!
inputs’’’’’’’’’
Ŗ "%"
unknown’’’’’’’’’³
D__inference_dropout_1_layer_call_and_return_conditional_losses_39021k7¢4
-¢*
$!
inputs’’’’’’’’’
p 
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 ³
D__inference_dropout_1_layer_call_and_return_conditional_losses_39033k7¢4
-¢*
$!
inputs’’’’’’’’’
p
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 
)__inference_dropout_1_layer_call_fn_39011`7¢4
-¢*
$!
inputs’’’’’’’’’
p 
Ŗ "%"
unknown’’’’’’’’’
)__inference_dropout_1_layer_call_fn_39016`7¢4
-¢*
$!
inputs’’’’’’’’’
p
Ŗ "%"
unknown’’’’’’’’’£
B__inference_dropout_layer_call_and_return_conditional_losses_38926]0¢-
&¢#

inputs
p 
Ŗ ")¢&

tensor_0
 £
B__inference_dropout_layer_call_and_return_conditional_losses_38938]0¢-
&¢#

inputs
p
Ŗ ")¢&

tensor_0
 }
'__inference_dropout_layer_call_fn_38916R0¢-
&¢#

inputs
p 
Ŗ "
unknown}
'__inference_dropout_layer_call_fn_38921R0¢-
&¢#

inputs
p
Ŗ "
unknowną
I__inference_gather_indices_layer_call_and_return_conditional_losses_39094^¢[
T¢Q
OL
&#
inputs_0’’’’’’’’’
"
inputs_1’’’’’’’’’
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 ŗ
.__inference_gather_indices_layer_call_fn_39087^¢[
T¢Q
OL
&#
inputs_0’’’’’’’’’
"
inputs_1’’’’’’’’’
Ŗ "%"
unknown’’’’’’’’’
N__inference_graph_convolution_1_layer_call_and_return_conditional_losses_39081¶78~¢{
t¢q
ol
&#
inputs_0’’’’’’’’’
B?'¢$
ś’’’’’’’’’’’’’’’’’’
SparseTensorSpec 
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 ć
3__inference_graph_convolution_1_layer_call_fn_39045«78~¢{
t¢q
ol
&#
inputs_0’’’’’’’’’
B?'¢$
ś’’’’’’’’’’’’’’’’’’
SparseTensorSpec 
Ŗ "%"
unknown’’’’’’’’’
L__inference_graph_convolution_layer_call_and_return_conditional_losses_39006Æ()w¢t
m¢j
he

inputs_0
B?'¢$
ś’’’’’’’’’’’’’’’’’’
SparseTensorSpec 
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 Ś
1__inference_graph_convolution_layer_call_fn_38970¤()w¢t
m¢j
he

inputs_0
B?'¢$
ś’’’’’’’’’’’’’’’’’’
SparseTensorSpec 
Ŗ "%"
unknown’’’’’’’’’®
@__inference_model_layer_call_and_return_conditional_losses_38611é()78EF¬¢Ø
 ¢


input_1
!
input_2’’’’’’’’’
%"
input_3’’’’’’’’’	
!
input_4’’’’’’’’’
p 

 
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 ®
@__inference_model_layer_call_and_return_conditional_losses_38639é()78EF¬¢Ø
 ¢


input_1
!
input_2’’’’’’’’’
%"
input_3’’’’’’’’’	
!
input_4’’’’’’’’’
p

 
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 ²
@__inference_model_layer_call_and_return_conditional_losses_38802ķ()78EF°¢¬
¤¢ 


inputs_0
"
inputs_1’’’’’’’’’
&#
inputs_2’’’’’’’’’	
"
inputs_3’’’’’’’’’
p 

 
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 ²
@__inference_model_layer_call_and_return_conditional_losses_38911ķ()78EF°¢¬
¤¢ 


inputs_0
"
inputs_1’’’’’’’’’
&#
inputs_2’’’’’’’’’	
"
inputs_3’’’’’’’’’
p

 
Ŗ "0¢-
&#
tensor_0’’’’’’’’’
 
%__inference_model_layer_call_fn_38395Ž()78EF¬¢Ø
 ¢


input_1
!
input_2’’’’’’’’’
%"
input_3’’’’’’’’’	
!
input_4’’’’’’’’’
p 

 
Ŗ "%"
unknown’’’’’’’’’
%__inference_model_layer_call_fn_38583Ž()78EF¬¢Ø
 ¢


input_1
!
input_2’’’’’’’’’
%"
input_3’’’’’’’’’	
!
input_4’’’’’’’’’
p

 
Ŗ "%"
unknown’’’’’’’’’
%__inference_model_layer_call_fn_38687ā()78EF°¢¬
¤¢ 


inputs_0
"
inputs_1’’’’’’’’’
&#
inputs_2’’’’’’’’’	
"
inputs_3’’’’’’’’’
p 

 
Ŗ "%"
unknown’’’’’’’’’
%__inference_model_layer_call_fn_38707ā()78EF°¢¬
¤¢ 


inputs_0
"
inputs_1’’’’’’’’’
&#
inputs_2’’’’’’’’’	
"
inputs_3’’’’’’’’’
p

 
Ŗ "%"
unknown’’’’’’’’’Æ
#__inference_signature_wrapper_38667()78EFÉ¢Å
¢ 
½Ŗ¹
)
input_1
input_1
,
input_2!
input_2’’’’’’’’’
0
input_3%"
input_3’’’’’’’’’	
,
input_4!
input_4’’’’’’’’’"1Ŗ.
,
dense# 
dense’’’’’’’’’ų
U__inference_squeezed_sparse_conversion_layer_call_and_return_conditional_losses_38958^¢[
T¢Q
OL
&#
inputs_0’’’’’’’’’	
"
inputs_1’’’’’’’’’
Ŗ "<¢9
2/¢
ś

SparseTensorSpec 
 ć
:__inference_squeezed_sparse_conversion_layer_call_fn_38948¤^¢[
T¢Q
OL
&#
inputs_0’’’’’’’’’	
"
inputs_1’’’’’’’’’
Ŗ "B?'¢$
ś’’’’’’’’’’’’’’’’’’
SparseTensorSpec 