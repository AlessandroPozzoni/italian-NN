??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
@
ReadVariableOp
resource
value"dtype"
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
?
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
executor_typestring ??
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
Conv64_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameConv64_1/kernel
{
#Conv64_1/kernel/Read/ReadVariableOpReadVariableOpConv64_1/kernel*&
_output_shapes
:@*
dtype0
r
Conv64_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameConv64_1/bias
k
!Conv64_1/bias/Read/ReadVariableOpReadVariableOpConv64_1/bias*
_output_shapes
:@*
dtype0
?
Conv64_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameConv64_2/kernel
{
#Conv64_2/kernel/Read/ReadVariableOpReadVariableOpConv64_2/kernel*&
_output_shapes
:@@*
dtype0
r
Conv64_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameConv64_2/bias
k
!Conv64_2/bias/Read/ReadVariableOpReadVariableOpConv64_2/bias*
_output_shapes
:@*
dtype0
?
Conv32_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameConv32_1/kernel
{
#Conv32_1/kernel/Read/ReadVariableOpReadVariableOpConv32_1/kernel*&
_output_shapes
:@ *
dtype0
r
Conv32_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv32_1/bias
k
!Conv32_1/bias/Read/ReadVariableOpReadVariableOpConv32_1/bias*
_output_shapes
: *
dtype0
?
Conv32_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameConv32_2/kernel
{
#Conv32_2/kernel/Read/ReadVariableOpReadVariableOpConv32_2/kernel*&
_output_shapes
:  *
dtype0
r
Conv32_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv32_2/bias
k
!Conv32_2/bias/Read/ReadVariableOpReadVariableOpConv32_2/bias*
_output_shapes
: *
dtype0
?
Conv16_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameConv16_1/kernel
{
#Conv16_1/kernel/Read/ReadVariableOpReadVariableOpConv16_1/kernel*&
_output_shapes
: *
dtype0
r
Conv16_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv16_1/bias
k
!Conv16_1/bias/Read/ReadVariableOpReadVariableOpConv16_1/bias*
_output_shapes
:*
dtype0
?
Conv16_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameConv16_2/kernel
{
#Conv16_2/kernel/Read/ReadVariableOpReadVariableOpConv16_2/kernel*&
_output_shapes
:*
dtype0
r
Conv16_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv16_2/bias
k
!Conv16_2/bias/Read/ReadVariableOpReadVariableOpConv16_2/bias*
_output_shapes
:*
dtype0
w
Dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*
shared_nameDense/kernel
p
 Dense/kernel/Read/ReadVariableOpReadVariableOpDense/kernel*!
_output_shapes
:???*
dtype0
m

Dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
Dense/bias
f
Dense/bias/Read/ReadVariableOpReadVariableOp
Dense/bias*
_output_shapes	
:?*
dtype0

Classifier/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*"
shared_nameClassifier/kernel
x
%Classifier/kernel/Read/ReadVariableOpReadVariableOpClassifier/kernel*
_output_shapes
:	?*
dtype0
v
Classifier/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameClassifier/bias
o
#Classifier/bias/Read/ReadVariableOpReadVariableOpClassifier/bias*
_output_shapes
:*
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
?
Adam/Conv64_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/Conv64_1/kernel/m
?
*Adam/Conv64_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv64_1/kernel/m*&
_output_shapes
:@*
dtype0
?
Adam/Conv64_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Conv64_1/bias/m
y
(Adam/Conv64_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv64_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/Conv64_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/Conv64_2/kernel/m
?
*Adam/Conv64_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv64_2/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/Conv64_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Conv64_2/bias/m
y
(Adam/Conv64_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv64_2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/Conv32_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/Conv32_1/kernel/m
?
*Adam/Conv32_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv32_1/kernel/m*&
_output_shapes
:@ *
dtype0
?
Adam/Conv32_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv32_1/bias/m
y
(Adam/Conv32_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv32_1/bias/m*
_output_shapes
: *
dtype0
?
Adam/Conv32_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/Conv32_2/kernel/m
?
*Adam/Conv32_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv32_2/kernel/m*&
_output_shapes
:  *
dtype0
?
Adam/Conv32_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv32_2/bias/m
y
(Adam/Conv32_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv32_2/bias/m*
_output_shapes
: *
dtype0
?
Adam/Conv16_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Conv16_1/kernel/m
?
*Adam/Conv16_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv16_1/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/Conv16_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv16_1/bias/m
y
(Adam/Conv16_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv16_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/Conv16_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Conv16_2/kernel/m
?
*Adam/Conv16_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv16_2/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/Conv16_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv16_2/bias/m
y
(Adam/Conv16_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv16_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/Dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/Dense/kernel/m
~
'Adam/Dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense/kernel/m*!
_output_shapes
:???*
dtype0
{
Adam/Dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/Dense/bias/m
t
%Adam/Dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/Classifier/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameAdam/Classifier/kernel/m
?
,Adam/Classifier/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Classifier/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/Classifier/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Classifier/bias/m
}
*Adam/Classifier/bias/m/Read/ReadVariableOpReadVariableOpAdam/Classifier/bias/m*
_output_shapes
:*
dtype0
?
Adam/Conv64_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/Conv64_1/kernel/v
?
*Adam/Conv64_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv64_1/kernel/v*&
_output_shapes
:@*
dtype0
?
Adam/Conv64_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Conv64_1/bias/v
y
(Adam/Conv64_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv64_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/Conv64_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/Conv64_2/kernel/v
?
*Adam/Conv64_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv64_2/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/Conv64_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/Conv64_2/bias/v
y
(Adam/Conv64_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv64_2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/Conv32_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *'
shared_nameAdam/Conv32_1/kernel/v
?
*Adam/Conv32_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv32_1/kernel/v*&
_output_shapes
:@ *
dtype0
?
Adam/Conv32_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv32_1/bias/v
y
(Adam/Conv32_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv32_1/bias/v*
_output_shapes
: *
dtype0
?
Adam/Conv32_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/Conv32_2/kernel/v
?
*Adam/Conv32_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv32_2/kernel/v*&
_output_shapes
:  *
dtype0
?
Adam/Conv32_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv32_2/bias/v
y
(Adam/Conv32_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv32_2/bias/v*
_output_shapes
: *
dtype0
?
Adam/Conv16_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Conv16_1/kernel/v
?
*Adam/Conv16_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv16_1/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/Conv16_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv16_1/bias/v
y
(Adam/Conv16_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv16_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/Conv16_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Conv16_2/kernel/v
?
*Adam/Conv16_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv16_2/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/Conv16_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv16_2/bias/v
y
(Adam/Conv16_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv16_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/Dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:???*$
shared_nameAdam/Dense/kernel/v
~
'Adam/Dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense/kernel/v*!
_output_shapes
:???*
dtype0
{
Adam/Dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/Dense/bias/v
t
%Adam/Dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/Classifier/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameAdam/Classifier/kernel/v
?
,Adam/Classifier/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Classifier/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/Classifier/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Classifier/bias/v
}
*Adam/Classifier/bias/v/Read/ReadVariableOpReadVariableOpAdam/Classifier/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?b
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?a
value?aB?a B?a
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
R
'	variables
(trainable_variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
R
7	variables
8trainable_variables
9regularization_losses
:	keras_api
h

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
h

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
R
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
R
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
R
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
h

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
R
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
h

]kernel
^bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
?
citer

dbeta_1

ebeta_2
	fdecay
glearning_ratem?m?!m?"m?+m?,m?1m?2m?;m?<m?Am?Bm?Sm?Tm?]m?^m?v?v?!v?"v?+v?,v?1v?2v?;v?<v?Av?Bv?Sv?Tv?]v?^v?
v
0
1
!2
"3
+4
,5
16
27
;8
<9
A10
B11
S12
T13
]14
^15
v
0
1
!2
"3
+4
,5
16
27
;8
<9
A10
B11
S12
T13
]14
^15
 
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEConv64_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv64_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEConv64_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv64_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
#	variables
$trainable_variables
%regularization_losses
 
 
 
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
[Y
VARIABLE_VALUEConv32_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv32_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
[Y
VARIABLE_VALUEConv32_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv32_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
[Y
VARIABLE_VALUEConv16_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv16_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

;0
<1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
[Y
VARIABLE_VALUEConv16_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv16_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

A0
B1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
XV
VARIABLE_VALUEDense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
Dense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
][
VARIABLE_VALUEClassifier/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEClassifier/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

]0
^1

]0
^1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
v
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
10
11
12
13
14
15

?0
?1
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
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/Conv64_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv64_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv64_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv64_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv32_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv32_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv32_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv32_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv16_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv16_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv16_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv16_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Dense/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Dense/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Classifier/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Classifier/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv64_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv64_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv64_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv64_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv32_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv32_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv32_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv32_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv16_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv16_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv16_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv16_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/Dense/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/Dense/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/Classifier/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/Classifier/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_InputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_InputConv64_1/kernelConv64_1/biasConv64_2/kernelConv64_2/biasConv32_1/kernelConv32_1/biasConv32_2/kernelConv32_2/biasConv16_1/kernelConv16_1/biasConv16_2/kernelConv16_2/biasDense/kernel
Dense/biasClassifier/kernelClassifier/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_6134
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#Conv64_1/kernel/Read/ReadVariableOp!Conv64_1/bias/Read/ReadVariableOp#Conv64_2/kernel/Read/ReadVariableOp!Conv64_2/bias/Read/ReadVariableOp#Conv32_1/kernel/Read/ReadVariableOp!Conv32_1/bias/Read/ReadVariableOp#Conv32_2/kernel/Read/ReadVariableOp!Conv32_2/bias/Read/ReadVariableOp#Conv16_1/kernel/Read/ReadVariableOp!Conv16_1/bias/Read/ReadVariableOp#Conv16_2/kernel/Read/ReadVariableOp!Conv16_2/bias/Read/ReadVariableOp Dense/kernel/Read/ReadVariableOpDense/bias/Read/ReadVariableOp%Classifier/kernel/Read/ReadVariableOp#Classifier/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/Conv64_1/kernel/m/Read/ReadVariableOp(Adam/Conv64_1/bias/m/Read/ReadVariableOp*Adam/Conv64_2/kernel/m/Read/ReadVariableOp(Adam/Conv64_2/bias/m/Read/ReadVariableOp*Adam/Conv32_1/kernel/m/Read/ReadVariableOp(Adam/Conv32_1/bias/m/Read/ReadVariableOp*Adam/Conv32_2/kernel/m/Read/ReadVariableOp(Adam/Conv32_2/bias/m/Read/ReadVariableOp*Adam/Conv16_1/kernel/m/Read/ReadVariableOp(Adam/Conv16_1/bias/m/Read/ReadVariableOp*Adam/Conv16_2/kernel/m/Read/ReadVariableOp(Adam/Conv16_2/bias/m/Read/ReadVariableOp'Adam/Dense/kernel/m/Read/ReadVariableOp%Adam/Dense/bias/m/Read/ReadVariableOp,Adam/Classifier/kernel/m/Read/ReadVariableOp*Adam/Classifier/bias/m/Read/ReadVariableOp*Adam/Conv64_1/kernel/v/Read/ReadVariableOp(Adam/Conv64_1/bias/v/Read/ReadVariableOp*Adam/Conv64_2/kernel/v/Read/ReadVariableOp(Adam/Conv64_2/bias/v/Read/ReadVariableOp*Adam/Conv32_1/kernel/v/Read/ReadVariableOp(Adam/Conv32_1/bias/v/Read/ReadVariableOp*Adam/Conv32_2/kernel/v/Read/ReadVariableOp(Adam/Conv32_2/bias/v/Read/ReadVariableOp*Adam/Conv16_1/kernel/v/Read/ReadVariableOp(Adam/Conv16_1/bias/v/Read/ReadVariableOp*Adam/Conv16_2/kernel/v/Read/ReadVariableOp(Adam/Conv16_2/bias/v/Read/ReadVariableOp'Adam/Dense/kernel/v/Read/ReadVariableOp%Adam/Dense/bias/v/Read/ReadVariableOp,Adam/Classifier/kernel/v/Read/ReadVariableOp*Adam/Classifier/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
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
GPU 2J 8? *&
f!R
__inference__traced_save_6856
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv64_1/kernelConv64_1/biasConv64_2/kernelConv64_2/biasConv32_1/kernelConv32_1/biasConv32_2/kernelConv32_2/biasConv16_1/kernelConv16_1/biasConv16_2/kernelConv16_2/biasDense/kernel
Dense/biasClassifier/kernelClassifier/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Conv64_1/kernel/mAdam/Conv64_1/bias/mAdam/Conv64_2/kernel/mAdam/Conv64_2/bias/mAdam/Conv32_1/kernel/mAdam/Conv32_1/bias/mAdam/Conv32_2/kernel/mAdam/Conv32_2/bias/mAdam/Conv16_1/kernel/mAdam/Conv16_1/bias/mAdam/Conv16_2/kernel/mAdam/Conv16_2/bias/mAdam/Dense/kernel/mAdam/Dense/bias/mAdam/Classifier/kernel/mAdam/Classifier/bias/mAdam/Conv64_1/kernel/vAdam/Conv64_1/bias/vAdam/Conv64_2/kernel/vAdam/Conv64_2/bias/vAdam/Conv32_1/kernel/vAdam/Conv32_1/bias/vAdam/Conv32_2/kernel/vAdam/Conv32_2/bias/vAdam/Conv16_1/kernel/vAdam/Conv16_1/bias/vAdam/Conv16_2/kernel/vAdam/Conv16_2/bias/vAdam/Dense/kernel/vAdam/Dense/bias/vAdam/Classifier/kernel/vAdam/Classifier/bias/v*E
Tin>
<2:*
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_7037??

?
_
C__inference_Rescaling_layer_call_and_return_conditional_losses_6377

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:???????????d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:???????????Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
]
A__inference_Flatten_layer_call_and_return_conditional_losses_6568

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? @  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
B__inference_Conv64_1_layer_call_and_return_conditional_losses_6397

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
?__inference_Dense_layer_call_and_return_conditional_losses_6615

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_Conv32_2_layer_call_fn_6466

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv32_2_layer_call_and_return_conditional_losses_5528y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
D
(__inference_dropout_1_layer_call_fn_6620

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5617a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_Rescaling_layer_call_and_return_conditional_losses_5458

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:???????????d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:???????????Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?]
?
__inference__wrapped_model_5407	
inputG
-model_conv64_1_conv2d_readvariableop_resource:@<
.model_conv64_1_biasadd_readvariableop_resource:@G
-model_conv64_2_conv2d_readvariableop_resource:@@<
.model_conv64_2_biasadd_readvariableop_resource:@G
-model_conv32_1_conv2d_readvariableop_resource:@ <
.model_conv32_1_biasadd_readvariableop_resource: G
-model_conv32_2_conv2d_readvariableop_resource:  <
.model_conv32_2_biasadd_readvariableop_resource: G
-model_conv16_1_conv2d_readvariableop_resource: <
.model_conv16_1_biasadd_readvariableop_resource:G
-model_conv16_2_conv2d_readvariableop_resource:<
.model_conv16_2_biasadd_readvariableop_resource:?
*model_dense_matmul_readvariableop_resource:???:
+model_dense_biasadd_readvariableop_resource:	?B
/model_classifier_matmul_readvariableop_resource:	?>
0model_classifier_biasadd_readvariableop_resource:
identity??'model/Classifier/BiasAdd/ReadVariableOp?&model/Classifier/MatMul/ReadVariableOp?%model/Conv16_1/BiasAdd/ReadVariableOp?$model/Conv16_1/Conv2D/ReadVariableOp?%model/Conv16_2/BiasAdd/ReadVariableOp?$model/Conv16_2/Conv2D/ReadVariableOp?%model/Conv32_1/BiasAdd/ReadVariableOp?$model/Conv32_1/Conv2D/ReadVariableOp?%model/Conv32_2/BiasAdd/ReadVariableOp?$model/Conv32_2/Conv2D/ReadVariableOp?%model/Conv64_1/BiasAdd/ReadVariableOp?$model/Conv64_1/Conv2D/ReadVariableOp?%model/Conv64_2/BiasAdd/ReadVariableOp?$model/Conv64_2/Conv2D/ReadVariableOp?"model/Dense/BiasAdd/ReadVariableOp?!model/Dense/MatMul/ReadVariableOp[
model/Rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;]
model/Rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
model/Rescaling/mulMulinputmodel/Rescaling/Cast/x:output:0*
T0*1
_output_shapes
:????????????
model/Rescaling/addAddV2model/Rescaling/mul:z:0!model/Rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:????????????
$model/Conv64_1/Conv2D/ReadVariableOpReadVariableOp-model_conv64_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
model/Conv64_1/Conv2DConv2Dmodel/Rescaling/add:z:0,model/Conv64_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
%model/Conv64_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv64_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/Conv64_1/BiasAddBiasAddmodel/Conv64_1/Conv2D:output:0-model/Conv64_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@x
model/Conv64_1/ReluRelumodel/Conv64_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
$model/Conv64_2/Conv2D/ReadVariableOpReadVariableOp-model_conv64_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
model/Conv64_2/Conv2DConv2D!model/Conv64_1/Relu:activations:0,model/Conv64_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
%model/Conv64_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv64_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/Conv64_2/BiasAddBiasAddmodel/Conv64_2/Conv2D:output:0-model/Conv64_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@x
model/Conv64_2/ReluRelumodel/Conv64_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
model/Pool_1/MaxPoolMaxPool!model/Conv64_2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
?
$model/Conv32_1/Conv2D/ReadVariableOpReadVariableOp-model_conv32_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
model/Conv32_1/Conv2DConv2Dmodel/Pool_1/MaxPool:output:0,model/Conv32_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
%model/Conv32_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv32_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model/Conv32_1/BiasAddBiasAddmodel/Conv32_1/Conv2D:output:0-model/Conv32_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? x
model/Conv32_1/ReluRelumodel/Conv32_1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? ?
$model/Conv32_2/Conv2D/ReadVariableOpReadVariableOp-model_conv32_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
model/Conv32_2/Conv2DConv2D!model/Conv32_1/Relu:activations:0,model/Conv32_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
%model/Conv32_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv32_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model/Conv32_2/BiasAddBiasAddmodel/Conv32_2/Conv2D:output:0-model/Conv32_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? x
model/Conv32_2/ReluRelumodel/Conv32_2/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? ?
model/Pool_2/MaxPoolMaxPool!model/Conv32_2/Relu:activations:0*/
_output_shapes
:?????????@@ *
ksize
*
paddingVALID*
strides
?
$model/Conv16_1/Conv2D/ReadVariableOpReadVariableOp-model_conv16_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
model/Conv16_1/Conv2DConv2Dmodel/Pool_2/MaxPool:output:0,model/Conv16_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
%model/Conv16_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv16_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/Conv16_1/BiasAddBiasAddmodel/Conv16_1/Conv2D:output:0-model/Conv16_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@v
model/Conv16_1/ReluRelumodel/Conv16_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
$model/Conv16_2/Conv2D/ReadVariableOpReadVariableOp-model_conv16_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model/Conv16_2/Conv2DConv2D!model/Conv16_1/Relu:activations:0,model/Conv16_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
%model/Conv16_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv16_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/Conv16_2/BiasAddBiasAddmodel/Conv16_2/Conv2D:output:0-model/Conv16_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@v
model/Conv16_2/ReluRelumodel/Conv16_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
model/Pool_3/MaxPoolMaxPool!model/Conv16_2/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
d
model/Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? @  ?
model/Flatten/ReshapeReshapemodel/Pool_3/MaxPool:output:0model/Flatten/Const:output:0*
T0*)
_output_shapes
:???????????v
model/dropout/IdentityIdentitymodel/Flatten/Reshape:output:0*
T0*)
_output_shapes
:????????????
!model/Dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype0?
model/Dense/MatMulMatMulmodel/dropout/Identity:output:0)model/Dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
"model/Dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model/Dense/BiasAddBiasAddmodel/Dense/MatMul:product:0*model/Dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????i
model/Dense/ReluRelumodel/Dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????w
model/dropout_1/IdentityIdentitymodel/Dense/Relu:activations:0*
T0*(
_output_shapes
:???????????
&model/Classifier/MatMul/ReadVariableOpReadVariableOp/model_classifier_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model/Classifier/MatMulMatMul!model/dropout_1/Identity:output:0.model/Classifier/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'model/Classifier/BiasAdd/ReadVariableOpReadVariableOp0model_classifier_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/Classifier/BiasAddBiasAdd!model/Classifier/MatMul:product:0/model/Classifier/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
model/Classifier/SoftmaxSoftmax!model/Classifier/BiasAdd:output:0*
T0*'
_output_shapes
:?????????q
IdentityIdentity"model/Classifier/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^model/Classifier/BiasAdd/ReadVariableOp'^model/Classifier/MatMul/ReadVariableOp&^model/Conv16_1/BiasAdd/ReadVariableOp%^model/Conv16_1/Conv2D/ReadVariableOp&^model/Conv16_2/BiasAdd/ReadVariableOp%^model/Conv16_2/Conv2D/ReadVariableOp&^model/Conv32_1/BiasAdd/ReadVariableOp%^model/Conv32_1/Conv2D/ReadVariableOp&^model/Conv32_2/BiasAdd/ReadVariableOp%^model/Conv32_2/Conv2D/ReadVariableOp&^model/Conv64_1/BiasAdd/ReadVariableOp%^model/Conv64_1/Conv2D/ReadVariableOp&^model/Conv64_2/BiasAdd/ReadVariableOp%^model/Conv64_2/Conv2D/ReadVariableOp#^model/Dense/BiasAdd/ReadVariableOp"^model/Dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2R
'model/Classifier/BiasAdd/ReadVariableOp'model/Classifier/BiasAdd/ReadVariableOp2P
&model/Classifier/MatMul/ReadVariableOp&model/Classifier/MatMul/ReadVariableOp2N
%model/Conv16_1/BiasAdd/ReadVariableOp%model/Conv16_1/BiasAdd/ReadVariableOp2L
$model/Conv16_1/Conv2D/ReadVariableOp$model/Conv16_1/Conv2D/ReadVariableOp2N
%model/Conv16_2/BiasAdd/ReadVariableOp%model/Conv16_2/BiasAdd/ReadVariableOp2L
$model/Conv16_2/Conv2D/ReadVariableOp$model/Conv16_2/Conv2D/ReadVariableOp2N
%model/Conv32_1/BiasAdd/ReadVariableOp%model/Conv32_1/BiasAdd/ReadVariableOp2L
$model/Conv32_1/Conv2D/ReadVariableOp$model/Conv32_1/Conv2D/ReadVariableOp2N
%model/Conv32_2/BiasAdd/ReadVariableOp%model/Conv32_2/BiasAdd/ReadVariableOp2L
$model/Conv32_2/Conv2D/ReadVariableOp$model/Conv32_2/Conv2D/ReadVariableOp2N
%model/Conv64_1/BiasAdd/ReadVariableOp%model/Conv64_1/BiasAdd/ReadVariableOp2L
$model/Conv64_1/Conv2D/ReadVariableOp$model/Conv64_1/Conv2D/ReadVariableOp2N
%model/Conv64_2/BiasAdd/ReadVariableOp%model/Conv64_2/BiasAdd/ReadVariableOp2L
$model/Conv64_2/Conv2D/ReadVariableOp$model/Conv64_2/Conv2D/ReadVariableOp2H
"model/Dense/BiasAdd/ReadVariableOp"model/Dense/BiasAdd/ReadVariableOp2F
!model/Dense/MatMul/ReadVariableOp!model/Dense/MatMul/ReadVariableOp:X T
1
_output_shapes
:???????????

_user_specified_nameInput
?
\
@__inference_Pool_3_layer_call_and_return_conditional_losses_5578

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?T
?
?__inference_model_layer_call_and_return_conditional_losses_6279

inputsA
'conv64_1_conv2d_readvariableop_resource:@6
(conv64_1_biasadd_readvariableop_resource:@A
'conv64_2_conv2d_readvariableop_resource:@@6
(conv64_2_biasadd_readvariableop_resource:@A
'conv32_1_conv2d_readvariableop_resource:@ 6
(conv32_1_biasadd_readvariableop_resource: A
'conv32_2_conv2d_readvariableop_resource:  6
(conv32_2_biasadd_readvariableop_resource: A
'conv16_1_conv2d_readvariableop_resource: 6
(conv16_1_biasadd_readvariableop_resource:A
'conv16_2_conv2d_readvariableop_resource:6
(conv16_2_biasadd_readvariableop_resource:9
$dense_matmul_readvariableop_resource:???4
%dense_biasadd_readvariableop_resource:	?<
)classifier_matmul_readvariableop_resource:	?8
*classifier_biasadd_readvariableop_resource:
identity??!Classifier/BiasAdd/ReadVariableOp? Classifier/MatMul/ReadVariableOp?Conv16_1/BiasAdd/ReadVariableOp?Conv16_1/Conv2D/ReadVariableOp?Conv16_2/BiasAdd/ReadVariableOp?Conv16_2/Conv2D/ReadVariableOp?Conv32_1/BiasAdd/ReadVariableOp?Conv32_1/Conv2D/ReadVariableOp?Conv32_2/BiasAdd/ReadVariableOp?Conv32_2/Conv2D/ReadVariableOp?Conv64_1/BiasAdd/ReadVariableOp?Conv64_1/Conv2D/ReadVariableOp?Conv64_2/BiasAdd/ReadVariableOp?Conv64_2/Conv2D/ReadVariableOp?Dense/BiasAdd/ReadVariableOp?Dense/MatMul/ReadVariableOpU
Rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;W
Rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    s
Rescaling/mulMulinputsRescaling/Cast/x:output:0*
T0*1
_output_shapes
:????????????
Rescaling/addAddV2Rescaling/mul:z:0Rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:????????????
Conv64_1/Conv2D/ReadVariableOpReadVariableOp'conv64_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv64_1/Conv2DConv2DRescaling/add:z:0&Conv64_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
Conv64_1/BiasAdd/ReadVariableOpReadVariableOp(conv64_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Conv64_1/BiasAddBiasAddConv64_1/Conv2D:output:0'Conv64_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@l
Conv64_1/ReluReluConv64_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
Conv64_2/Conv2D/ReadVariableOpReadVariableOp'conv64_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv64_2/Conv2DConv2DConv64_1/Relu:activations:0&Conv64_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
Conv64_2/BiasAdd/ReadVariableOpReadVariableOp(conv64_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Conv64_2/BiasAddBiasAddConv64_2/Conv2D:output:0'Conv64_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@l
Conv64_2/ReluReluConv64_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
Pool_1/MaxPoolMaxPoolConv64_2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
?
Conv32_1/Conv2D/ReadVariableOpReadVariableOp'conv32_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv32_1/Conv2DConv2DPool_1/MaxPool:output:0&Conv32_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
Conv32_1/BiasAdd/ReadVariableOpReadVariableOp(conv32_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv32_1/BiasAddBiasAddConv32_1/Conv2D:output:0'Conv32_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? l
Conv32_1/ReluReluConv32_1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? ?
Conv32_2/Conv2D/ReadVariableOpReadVariableOp'conv32_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv32_2/Conv2DConv2DConv32_1/Relu:activations:0&Conv32_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
Conv32_2/BiasAdd/ReadVariableOpReadVariableOp(conv32_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv32_2/BiasAddBiasAddConv32_2/Conv2D:output:0'Conv32_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? l
Conv32_2/ReluReluConv32_2/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? ?
Pool_2/MaxPoolMaxPoolConv32_2/Relu:activations:0*/
_output_shapes
:?????????@@ *
ksize
*
paddingVALID*
strides
?
Conv16_1/Conv2D/ReadVariableOpReadVariableOp'conv16_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv16_1/Conv2DConv2DPool_2/MaxPool:output:0&Conv16_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
Conv16_1/BiasAdd/ReadVariableOpReadVariableOp(conv16_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Conv16_1/BiasAddBiasAddConv16_1/Conv2D:output:0'Conv16_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@j
Conv16_1/ReluReluConv16_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
Conv16_2/Conv2D/ReadVariableOpReadVariableOp'conv16_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv16_2/Conv2DConv2DConv16_1/Relu:activations:0&Conv16_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
Conv16_2/BiasAdd/ReadVariableOpReadVariableOp(conv16_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Conv16_2/BiasAddBiasAddConv16_2/Conv2D:output:0'Conv16_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@j
Conv16_2/ReluReluConv16_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
Pool_3/MaxPoolMaxPoolConv16_2/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
^
Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? @  
Flatten/ReshapeReshapePool_3/MaxPool:output:0Flatten/Const:output:0*
T0*)
_output_shapes
:???????????j
dropout/IdentityIdentityFlatten/Reshape:output:0*
T0*)
_output_shapes
:????????????
Dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype0?
Dense/MatMulMatMuldropout/Identity:output:0#Dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Dense/BiasAddBiasAddDense/MatMul:product:0$Dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

Dense/ReluReluDense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????k
dropout_1/IdentityIdentityDense/Relu:activations:0*
T0*(
_output_shapes
:???????????
 Classifier/MatMul/ReadVariableOpReadVariableOp)classifier_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Classifier/MatMulMatMuldropout_1/Identity:output:0(Classifier/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!Classifier/BiasAdd/ReadVariableOpReadVariableOp*classifier_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Classifier/BiasAddBiasAddClassifier/MatMul:product:0)Classifier/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????l
Classifier/SoftmaxSoftmaxClassifier/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
IdentityIdentityClassifier/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^Classifier/BiasAdd/ReadVariableOp!^Classifier/MatMul/ReadVariableOp ^Conv16_1/BiasAdd/ReadVariableOp^Conv16_1/Conv2D/ReadVariableOp ^Conv16_2/BiasAdd/ReadVariableOp^Conv16_2/Conv2D/ReadVariableOp ^Conv32_1/BiasAdd/ReadVariableOp^Conv32_1/Conv2D/ReadVariableOp ^Conv32_2/BiasAdd/ReadVariableOp^Conv32_2/Conv2D/ReadVariableOp ^Conv64_1/BiasAdd/ReadVariableOp^Conv64_1/Conv2D/ReadVariableOp ^Conv64_2/BiasAdd/ReadVariableOp^Conv64_2/Conv2D/ReadVariableOp^Dense/BiasAdd/ReadVariableOp^Dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2F
!Classifier/BiasAdd/ReadVariableOp!Classifier/BiasAdd/ReadVariableOp2D
 Classifier/MatMul/ReadVariableOp Classifier/MatMul/ReadVariableOp2B
Conv16_1/BiasAdd/ReadVariableOpConv16_1/BiasAdd/ReadVariableOp2@
Conv16_1/Conv2D/ReadVariableOpConv16_1/Conv2D/ReadVariableOp2B
Conv16_2/BiasAdd/ReadVariableOpConv16_2/BiasAdd/ReadVariableOp2@
Conv16_2/Conv2D/ReadVariableOpConv16_2/Conv2D/ReadVariableOp2B
Conv32_1/BiasAdd/ReadVariableOpConv32_1/BiasAdd/ReadVariableOp2@
Conv32_1/Conv2D/ReadVariableOpConv32_1/Conv2D/ReadVariableOp2B
Conv32_2/BiasAdd/ReadVariableOpConv32_2/BiasAdd/ReadVariableOp2@
Conv32_2/Conv2D/ReadVariableOpConv32_2/Conv2D/ReadVariableOp2B
Conv64_1/BiasAdd/ReadVariableOpConv64_1/BiasAdd/ReadVariableOp2@
Conv64_1/Conv2D/ReadVariableOpConv64_1/Conv2D/ReadVariableOp2B
Conv64_2/BiasAdd/ReadVariableOpConv64_2/BiasAdd/ReadVariableOp2@
Conv64_2/Conv2D/ReadVariableOpConv64_2/Conv2D/ReadVariableOp2<
Dense/BiasAdd/ReadVariableOpDense/BiasAdd/ReadVariableOp2:
Dense/MatMul/ReadVariableOpDense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_Classifier_layer_call_and_return_conditional_losses_5630

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
A
%__inference_Pool_2_layer_call_fn_6487

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_2_layer_call_and_return_conditional_losses_5538h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
\
@__inference_Pool_2_layer_call_and_return_conditional_losses_6492

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
\
@__inference_Pool_1_layer_call_and_return_conditional_losses_5498

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
b
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
B
&__inference_Flatten_layer_call_fn_6562

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Flatten_layer_call_and_return_conditional_losses_5586b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
)__inference_Classifier_layer_call_fn_6651

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Classifier_layer_call_and_return_conditional_losses_5630o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
A
%__inference_Pool_1_layer_call_fn_6427

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_1_layer_call_and_return_conditional_losses_5498j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
B__inference_Conv32_2_layer_call_and_return_conditional_losses_6477

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?=
?
?__inference_model_layer_call_and_return_conditional_losses_6038	
input'
conv64_1_5991:@
conv64_1_5993:@'
conv64_2_5996:@@
conv64_2_5998:@'
conv32_1_6002:@ 
conv32_1_6004: '
conv32_2_6007:  
conv32_2_6009: '
conv16_1_6013: 
conv16_1_6015:'
conv16_2_6018:
conv16_2_6020:

dense_6026:???

dense_6028:	?"
classifier_6032:	?
classifier_6034:
identity??"Classifier/StatefulPartitionedCall? Conv16_1/StatefulPartitionedCall? Conv16_2/StatefulPartitionedCall? Conv32_1/StatefulPartitionedCall? Conv32_2/StatefulPartitionedCall? Conv64_1/StatefulPartitionedCall? Conv64_2/StatefulPartitionedCall?Dense/StatefulPartitionedCall?
Rescaling/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Rescaling_layer_call_and_return_conditional_losses_5458?
 Conv64_1/StatefulPartitionedCallStatefulPartitionedCall"Rescaling/PartitionedCall:output:0conv64_1_5991conv64_1_5993*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv64_1_layer_call_and_return_conditional_losses_5471?
 Conv64_2/StatefulPartitionedCallStatefulPartitionedCall)Conv64_1/StatefulPartitionedCall:output:0conv64_2_5996conv64_2_5998*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv64_2_layer_call_and_return_conditional_losses_5488?
Pool_1/PartitionedCallPartitionedCall)Conv64_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_1_layer_call_and_return_conditional_losses_5498?
 Conv32_1/StatefulPartitionedCallStatefulPartitionedCallPool_1/PartitionedCall:output:0conv32_1_6002conv32_1_6004*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv32_1_layer_call_and_return_conditional_losses_5511?
 Conv32_2/StatefulPartitionedCallStatefulPartitionedCall)Conv32_1/StatefulPartitionedCall:output:0conv32_2_6007conv32_2_6009*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv32_2_layer_call_and_return_conditional_losses_5528?
Pool_2/PartitionedCallPartitionedCall)Conv32_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_2_layer_call_and_return_conditional_losses_5538?
 Conv16_1/StatefulPartitionedCallStatefulPartitionedCallPool_2/PartitionedCall:output:0conv16_1_6013conv16_1_6015*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv16_1_layer_call_and_return_conditional_losses_5551?
 Conv16_2/StatefulPartitionedCallStatefulPartitionedCall)Conv16_1/StatefulPartitionedCall:output:0conv16_2_6018conv16_2_6020*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv16_2_layer_call_and_return_conditional_losses_5568?
Pool_3/PartitionedCallPartitionedCall)Conv16_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_3_layer_call_and_return_conditional_losses_5578?
Flatten/PartitionedCallPartitionedCallPool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Flatten_layer_call_and_return_conditional_losses_5586?
dropout/PartitionedCallPartitionedCall Flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5593?
Dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
dense_6026
dense_6028*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense_layer_call_and_return_conditional_losses_5606?
dropout_1/PartitionedCallPartitionedCall&Dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5617?
"Classifier/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0classifier_6032classifier_6034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Classifier_layer_call_and_return_conditional_losses_5630z
IdentityIdentity+Classifier/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^Classifier/StatefulPartitionedCall!^Conv16_1/StatefulPartitionedCall!^Conv16_2/StatefulPartitionedCall!^Conv32_1/StatefulPartitionedCall!^Conv32_2/StatefulPartitionedCall!^Conv64_1/StatefulPartitionedCall!^Conv64_2/StatefulPartitionedCall^Dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2H
"Classifier/StatefulPartitionedCall"Classifier/StatefulPartitionedCall2D
 Conv16_1/StatefulPartitionedCall Conv16_1/StatefulPartitionedCall2D
 Conv16_2/StatefulPartitionedCall Conv16_2/StatefulPartitionedCall2D
 Conv32_1/StatefulPartitionedCall Conv32_1/StatefulPartitionedCall2D
 Conv32_2/StatefulPartitionedCall Conv32_2/StatefulPartitionedCall2D
 Conv64_1/StatefulPartitionedCall Conv64_1/StatefulPartitionedCall2D
 Conv64_2/StatefulPartitionedCall Conv64_2/StatefulPartitionedCall2>
Dense/StatefulPartitionedCallDense/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameInput
??
?
?__inference_model_layer_call_and_return_conditional_losses_6089	
input'
conv64_1_6042:@
conv64_1_6044:@'
conv64_2_6047:@@
conv64_2_6049:@'
conv32_1_6053:@ 
conv32_1_6055: '
conv32_2_6058:  
conv32_2_6060: '
conv16_1_6064: 
conv16_1_6066:'
conv16_2_6069:
conv16_2_6071:

dense_6077:???

dense_6079:	?"
classifier_6083:	?
classifier_6085:
identity??"Classifier/StatefulPartitionedCall? Conv16_1/StatefulPartitionedCall? Conv16_2/StatefulPartitionedCall? Conv32_1/StatefulPartitionedCall? Conv32_2/StatefulPartitionedCall? Conv64_1/StatefulPartitionedCall? Conv64_2/StatefulPartitionedCall?Dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
Rescaling/PartitionedCallPartitionedCallinput*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Rescaling_layer_call_and_return_conditional_losses_5458?
 Conv64_1/StatefulPartitionedCallStatefulPartitionedCall"Rescaling/PartitionedCall:output:0conv64_1_6042conv64_1_6044*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv64_1_layer_call_and_return_conditional_losses_5471?
 Conv64_2/StatefulPartitionedCallStatefulPartitionedCall)Conv64_1/StatefulPartitionedCall:output:0conv64_2_6047conv64_2_6049*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv64_2_layer_call_and_return_conditional_losses_5488?
Pool_1/PartitionedCallPartitionedCall)Conv64_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_1_layer_call_and_return_conditional_losses_5498?
 Conv32_1/StatefulPartitionedCallStatefulPartitionedCallPool_1/PartitionedCall:output:0conv32_1_6053conv32_1_6055*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv32_1_layer_call_and_return_conditional_losses_5511?
 Conv32_2/StatefulPartitionedCallStatefulPartitionedCall)Conv32_1/StatefulPartitionedCall:output:0conv32_2_6058conv32_2_6060*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv32_2_layer_call_and_return_conditional_losses_5528?
Pool_2/PartitionedCallPartitionedCall)Conv32_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_2_layer_call_and_return_conditional_losses_5538?
 Conv16_1/StatefulPartitionedCallStatefulPartitionedCallPool_2/PartitionedCall:output:0conv16_1_6064conv16_1_6066*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv16_1_layer_call_and_return_conditional_losses_5551?
 Conv16_2/StatefulPartitionedCallStatefulPartitionedCall)Conv16_1/StatefulPartitionedCall:output:0conv16_2_6069conv16_2_6071*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv16_2_layer_call_and_return_conditional_losses_5568?
Pool_3/PartitionedCallPartitionedCall)Conv16_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_3_layer_call_and_return_conditional_losses_5578?
Flatten/PartitionedCallPartitionedCallPool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Flatten_layer_call_and_return_conditional_losses_5586?
dropout/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5735?
Dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
dense_6077
dense_6079*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense_layer_call_and_return_conditional_losses_5606?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&Dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5702?
"Classifier/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0classifier_6083classifier_6085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Classifier_layer_call_and_return_conditional_losses_5630z
IdentityIdentity+Classifier/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^Classifier/StatefulPartitionedCall!^Conv16_1/StatefulPartitionedCall!^Conv16_2/StatefulPartitionedCall!^Conv32_1/StatefulPartitionedCall!^Conv32_2/StatefulPartitionedCall!^Conv64_1/StatefulPartitionedCall!^Conv64_2/StatefulPartitionedCall^Dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2H
"Classifier/StatefulPartitionedCall"Classifier/StatefulPartitionedCall2D
 Conv16_1/StatefulPartitionedCall Conv16_1/StatefulPartitionedCall2D
 Conv16_2/StatefulPartitionedCall Conv16_2/StatefulPartitionedCall2D
 Conv32_1/StatefulPartitionedCall Conv32_1/StatefulPartitionedCall2D
 Conv32_2/StatefulPartitionedCall Conv32_2/StatefulPartitionedCall2D
 Conv64_1/StatefulPartitionedCall Conv64_1/StatefulPartitionedCall2D
 Conv64_2/StatefulPartitionedCall Conv64_2/StatefulPartitionedCall2>
Dense/StatefulPartitionedCallDense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameInput
?q
?
__inference__traced_save_6856
file_prefix.
*savev2_conv64_1_kernel_read_readvariableop,
(savev2_conv64_1_bias_read_readvariableop.
*savev2_conv64_2_kernel_read_readvariableop,
(savev2_conv64_2_bias_read_readvariableop.
*savev2_conv32_1_kernel_read_readvariableop,
(savev2_conv32_1_bias_read_readvariableop.
*savev2_conv32_2_kernel_read_readvariableop,
(savev2_conv32_2_bias_read_readvariableop.
*savev2_conv16_1_kernel_read_readvariableop,
(savev2_conv16_1_bias_read_readvariableop.
*savev2_conv16_2_kernel_read_readvariableop,
(savev2_conv16_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop0
,savev2_classifier_kernel_read_readvariableop.
*savev2_classifier_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv64_1_kernel_m_read_readvariableop3
/savev2_adam_conv64_1_bias_m_read_readvariableop5
1savev2_adam_conv64_2_kernel_m_read_readvariableop3
/savev2_adam_conv64_2_bias_m_read_readvariableop5
1savev2_adam_conv32_1_kernel_m_read_readvariableop3
/savev2_adam_conv32_1_bias_m_read_readvariableop5
1savev2_adam_conv32_2_kernel_m_read_readvariableop3
/savev2_adam_conv32_2_bias_m_read_readvariableop5
1savev2_adam_conv16_1_kernel_m_read_readvariableop3
/savev2_adam_conv16_1_bias_m_read_readvariableop5
1savev2_adam_conv16_2_kernel_m_read_readvariableop3
/savev2_adam_conv16_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop7
3savev2_adam_classifier_kernel_m_read_readvariableop5
1savev2_adam_classifier_bias_m_read_readvariableop5
1savev2_adam_conv64_1_kernel_v_read_readvariableop3
/savev2_adam_conv64_1_bias_v_read_readvariableop5
1savev2_adam_conv64_2_kernel_v_read_readvariableop3
/savev2_adam_conv64_2_bias_v_read_readvariableop5
1savev2_adam_conv32_1_kernel_v_read_readvariableop3
/savev2_adam_conv32_1_bias_v_read_readvariableop5
1savev2_adam_conv32_2_kernel_v_read_readvariableop3
/savev2_adam_conv32_2_bias_v_read_readvariableop5
1savev2_adam_conv16_1_kernel_v_read_readvariableop3
/savev2_adam_conv16_1_bias_v_read_readvariableop5
1savev2_adam_conv16_2_kernel_v_read_readvariableop3
/savev2_adam_conv16_2_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop7
3savev2_adam_classifier_kernel_v_read_readvariableop5
1savev2_adam_classifier_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ? 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value?B?:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv64_1_kernel_read_readvariableop(savev2_conv64_1_bias_read_readvariableop*savev2_conv64_2_kernel_read_readvariableop(savev2_conv64_2_bias_read_readvariableop*savev2_conv32_1_kernel_read_readvariableop(savev2_conv32_1_bias_read_readvariableop*savev2_conv32_2_kernel_read_readvariableop(savev2_conv32_2_bias_read_readvariableop*savev2_conv16_1_kernel_read_readvariableop(savev2_conv16_1_bias_read_readvariableop*savev2_conv16_2_kernel_read_readvariableop(savev2_conv16_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop,savev2_classifier_kernel_read_readvariableop*savev2_classifier_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv64_1_kernel_m_read_readvariableop/savev2_adam_conv64_1_bias_m_read_readvariableop1savev2_adam_conv64_2_kernel_m_read_readvariableop/savev2_adam_conv64_2_bias_m_read_readvariableop1savev2_adam_conv32_1_kernel_m_read_readvariableop/savev2_adam_conv32_1_bias_m_read_readvariableop1savev2_adam_conv32_2_kernel_m_read_readvariableop/savev2_adam_conv32_2_bias_m_read_readvariableop1savev2_adam_conv16_1_kernel_m_read_readvariableop/savev2_adam_conv16_1_bias_m_read_readvariableop1savev2_adam_conv16_2_kernel_m_read_readvariableop/savev2_adam_conv16_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop3savev2_adam_classifier_kernel_m_read_readvariableop1savev2_adam_classifier_bias_m_read_readvariableop1savev2_adam_conv64_1_kernel_v_read_readvariableop/savev2_adam_conv64_1_bias_v_read_readvariableop1savev2_adam_conv64_2_kernel_v_read_readvariableop/savev2_adam_conv64_2_bias_v_read_readvariableop1savev2_adam_conv32_1_kernel_v_read_readvariableop/savev2_adam_conv32_1_bias_v_read_readvariableop1savev2_adam_conv32_2_kernel_v_read_readvariableop/savev2_adam_conv32_2_bias_v_read_readvariableop1savev2_adam_conv16_1_kernel_v_read_readvariableop/savev2_adam_conv16_1_bias_v_read_readvariableop1savev2_adam_conv16_2_kernel_v_read_readvariableop/savev2_adam_conv16_2_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop3savev2_adam_classifier_kernel_v_read_readvariableop1savev2_adam_classifier_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@@:@:@ : :  : : ::::???:?:	?:: : : : : : : : : :@:@:@@:@:@ : :  : : ::::???:?:	?::@:@:@@:@:@ : :  : : ::::???:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
: : 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::'#
!
_output_shapes
:???:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :, (
&
_output_shapes
:  : !

_output_shapes
: :,"(
&
_output_shapes
: : #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::'&#
!
_output_shapes
:???:!'

_output_shapes	
:?:%(!

_output_shapes
:	?: )

_output_shapes
::,*(
&
_output_shapes
:@: +

_output_shapes
:@:,,(
&
_output_shapes
:@@: -

_output_shapes
:@:,.(
&
_output_shapes
:@ : /

_output_shapes
: :,0(
&
_output_shapes
:  : 1

_output_shapes
: :,2(
&
_output_shapes
: : 3

_output_shapes
::,4(
&
_output_shapes
:: 5

_output_shapes
::'6#
!
_output_shapes
:???:!7

_output_shapes	
:?:%8!

_output_shapes
:	?: 9

_output_shapes
:::

_output_shapes
: 
?

`
A__inference_dropout_layer_call_and_return_conditional_losses_5735

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype0*

seed**
seed2*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:???????????q
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:???????????k
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:???????????[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?#
 __inference__traced_restore_7037
file_prefix:
 assignvariableop_conv64_1_kernel:@.
 assignvariableop_1_conv64_1_bias:@<
"assignvariableop_2_conv64_2_kernel:@@.
 assignvariableop_3_conv64_2_bias:@<
"assignvariableop_4_conv32_1_kernel:@ .
 assignvariableop_5_conv32_1_bias: <
"assignvariableop_6_conv32_2_kernel:  .
 assignvariableop_7_conv32_2_bias: <
"assignvariableop_8_conv16_1_kernel: .
 assignvariableop_9_conv16_1_bias:=
#assignvariableop_10_conv16_2_kernel:/
!assignvariableop_11_conv16_2_bias:5
 assignvariableop_12_dense_kernel:???-
assignvariableop_13_dense_bias:	?8
%assignvariableop_14_classifier_kernel:	?1
#assignvariableop_15_classifier_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: #
assignvariableop_21_total: #
assignvariableop_22_count: %
assignvariableop_23_total_1: %
assignvariableop_24_count_1: D
*assignvariableop_25_adam_conv64_1_kernel_m:@6
(assignvariableop_26_adam_conv64_1_bias_m:@D
*assignvariableop_27_adam_conv64_2_kernel_m:@@6
(assignvariableop_28_adam_conv64_2_bias_m:@D
*assignvariableop_29_adam_conv32_1_kernel_m:@ 6
(assignvariableop_30_adam_conv32_1_bias_m: D
*assignvariableop_31_adam_conv32_2_kernel_m:  6
(assignvariableop_32_adam_conv32_2_bias_m: D
*assignvariableop_33_adam_conv16_1_kernel_m: 6
(assignvariableop_34_adam_conv16_1_bias_m:D
*assignvariableop_35_adam_conv16_2_kernel_m:6
(assignvariableop_36_adam_conv16_2_bias_m:<
'assignvariableop_37_adam_dense_kernel_m:???4
%assignvariableop_38_adam_dense_bias_m:	??
,assignvariableop_39_adam_classifier_kernel_m:	?8
*assignvariableop_40_adam_classifier_bias_m:D
*assignvariableop_41_adam_conv64_1_kernel_v:@6
(assignvariableop_42_adam_conv64_1_bias_v:@D
*assignvariableop_43_adam_conv64_2_kernel_v:@@6
(assignvariableop_44_adam_conv64_2_bias_v:@D
*assignvariableop_45_adam_conv32_1_kernel_v:@ 6
(assignvariableop_46_adam_conv32_1_bias_v: D
*assignvariableop_47_adam_conv32_2_kernel_v:  6
(assignvariableop_48_adam_conv32_2_bias_v: D
*assignvariableop_49_adam_conv16_1_kernel_v: 6
(assignvariableop_50_adam_conv16_1_bias_v:D
*assignvariableop_51_adam_conv16_2_kernel_v:6
(assignvariableop_52_adam_conv16_2_bias_v:<
'assignvariableop_53_adam_dense_kernel_v:???4
%assignvariableop_54_adam_dense_bias_v:	??
,assignvariableop_55_adam_classifier_kernel_v:	?8
*assignvariableop_56_adam_classifier_bias_v:
identity_58??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9? 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value?B?:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*?
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_conv64_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv64_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv64_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv64_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv32_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv32_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv32_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv32_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv16_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv16_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv16_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv16_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp assignvariableop_12_dense_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp%assignvariableop_14_classifier_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_classifier_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv64_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv64_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv64_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv64_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv32_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv32_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv32_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv32_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv16_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv16_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv16_2_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv16_2_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_dense_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_classifier_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_classifier_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv64_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv64_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv64_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv64_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv32_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv32_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv32_2_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv32_2_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv16_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv16_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv16_2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv16_2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_dense_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp%assignvariableop_54_adam_dense_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_classifier_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp*assignvariableop_56_adam_classifier_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: ?

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*?
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
\
@__inference_Pool_2_layer_call_and_return_conditional_losses_5428

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_Conv32_1_layer_call_and_return_conditional_losses_6457

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_6171

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@ 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:???

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_5637o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_5672	
input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@ 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:???

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_5637o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameInput
?
\
@__inference_Pool_1_layer_call_and_return_conditional_losses_6437

inputs
identity?
MaxPoolMaxPoolinputs*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
b
IdentityIdentityMaxPool:output:0*
T0*1
_output_shapes
:???????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_6630

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
A
%__inference_Pool_3_layer_call_fn_6547

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_3_layer_call_and_return_conditional_losses_5578h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
$__inference_Dense_layer_call_fn_6604

inputs
unknown:???
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense_layer_call_and_return_conditional_losses_5606p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_Conv16_1_layer_call_and_return_conditional_losses_5551

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?=
?
?__inference_model_layer_call_and_return_conditional_losses_5637

inputs'
conv64_1_5472:@
conv64_1_5474:@'
conv64_2_5489:@@
conv64_2_5491:@'
conv32_1_5512:@ 
conv32_1_5514: '
conv32_2_5529:  
conv32_2_5531: '
conv16_1_5552: 
conv16_1_5554:'
conv16_2_5569:
conv16_2_5571:

dense_5607:???

dense_5609:	?"
classifier_5631:	?
classifier_5633:
identity??"Classifier/StatefulPartitionedCall? Conv16_1/StatefulPartitionedCall? Conv16_2/StatefulPartitionedCall? Conv32_1/StatefulPartitionedCall? Conv32_2/StatefulPartitionedCall? Conv64_1/StatefulPartitionedCall? Conv64_2/StatefulPartitionedCall?Dense/StatefulPartitionedCall?
Rescaling/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Rescaling_layer_call_and_return_conditional_losses_5458?
 Conv64_1/StatefulPartitionedCallStatefulPartitionedCall"Rescaling/PartitionedCall:output:0conv64_1_5472conv64_1_5474*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv64_1_layer_call_and_return_conditional_losses_5471?
 Conv64_2/StatefulPartitionedCallStatefulPartitionedCall)Conv64_1/StatefulPartitionedCall:output:0conv64_2_5489conv64_2_5491*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv64_2_layer_call_and_return_conditional_losses_5488?
Pool_1/PartitionedCallPartitionedCall)Conv64_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_1_layer_call_and_return_conditional_losses_5498?
 Conv32_1/StatefulPartitionedCallStatefulPartitionedCallPool_1/PartitionedCall:output:0conv32_1_5512conv32_1_5514*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv32_1_layer_call_and_return_conditional_losses_5511?
 Conv32_2/StatefulPartitionedCallStatefulPartitionedCall)Conv32_1/StatefulPartitionedCall:output:0conv32_2_5529conv32_2_5531*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv32_2_layer_call_and_return_conditional_losses_5528?
Pool_2/PartitionedCallPartitionedCall)Conv32_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_2_layer_call_and_return_conditional_losses_5538?
 Conv16_1/StatefulPartitionedCallStatefulPartitionedCallPool_2/PartitionedCall:output:0conv16_1_5552conv16_1_5554*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv16_1_layer_call_and_return_conditional_losses_5551?
 Conv16_2/StatefulPartitionedCallStatefulPartitionedCall)Conv16_1/StatefulPartitionedCall:output:0conv16_2_5569conv16_2_5571*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv16_2_layer_call_and_return_conditional_losses_5568?
Pool_3/PartitionedCallPartitionedCall)Conv16_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_3_layer_call_and_return_conditional_losses_5578?
Flatten/PartitionedCallPartitionedCallPool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Flatten_layer_call_and_return_conditional_losses_5586?
dropout/PartitionedCallPartitionedCall Flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5593?
Dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
dense_5607
dense_5609*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense_layer_call_and_return_conditional_losses_5606?
dropout_1/PartitionedCallPartitionedCall&Dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5617?
"Classifier/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0classifier_5631classifier_5633*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Classifier_layer_call_and_return_conditional_losses_5630z
IdentityIdentity+Classifier/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^Classifier/StatefulPartitionedCall!^Conv16_1/StatefulPartitionedCall!^Conv16_2/StatefulPartitionedCall!^Conv32_1/StatefulPartitionedCall!^Conv32_2/StatefulPartitionedCall!^Conv64_1/StatefulPartitionedCall!^Conv64_2/StatefulPartitionedCall^Dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2H
"Classifier/StatefulPartitionedCall"Classifier/StatefulPartitionedCall2D
 Conv16_1/StatefulPartitionedCall Conv16_1/StatefulPartitionedCall2D
 Conv16_2/StatefulPartitionedCall Conv16_2/StatefulPartitionedCall2D
 Conv32_1/StatefulPartitionedCall Conv32_1/StatefulPartitionedCall2D
 Conv32_2/StatefulPartitionedCall Conv32_2/StatefulPartitionedCall2D
 Conv64_1/StatefulPartitionedCall Conv64_1/StatefulPartitionedCall2D
 Conv64_2/StatefulPartitionedCall Conv64_2/StatefulPartitionedCall2>
Dense/StatefulPartitionedCallDense/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
?__inference_model_layer_call_and_return_conditional_losses_5915

inputs'
conv64_1_5868:@
conv64_1_5870:@'
conv64_2_5873:@@
conv64_2_5875:@'
conv32_1_5879:@ 
conv32_1_5881: '
conv32_2_5884:  
conv32_2_5886: '
conv16_1_5890: 
conv16_1_5892:'
conv16_2_5895:
conv16_2_5897:

dense_5903:???

dense_5905:	?"
classifier_5909:	?
classifier_5911:
identity??"Classifier/StatefulPartitionedCall? Conv16_1/StatefulPartitionedCall? Conv16_2/StatefulPartitionedCall? Conv32_1/StatefulPartitionedCall? Conv32_2/StatefulPartitionedCall? Conv64_1/StatefulPartitionedCall? Conv64_2/StatefulPartitionedCall?Dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
Rescaling/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Rescaling_layer_call_and_return_conditional_losses_5458?
 Conv64_1/StatefulPartitionedCallStatefulPartitionedCall"Rescaling/PartitionedCall:output:0conv64_1_5868conv64_1_5870*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv64_1_layer_call_and_return_conditional_losses_5471?
 Conv64_2/StatefulPartitionedCallStatefulPartitionedCall)Conv64_1/StatefulPartitionedCall:output:0conv64_2_5873conv64_2_5875*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv64_2_layer_call_and_return_conditional_losses_5488?
Pool_1/PartitionedCallPartitionedCall)Conv64_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_1_layer_call_and_return_conditional_losses_5498?
 Conv32_1/StatefulPartitionedCallStatefulPartitionedCallPool_1/PartitionedCall:output:0conv32_1_5879conv32_1_5881*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv32_1_layer_call_and_return_conditional_losses_5511?
 Conv32_2/StatefulPartitionedCallStatefulPartitionedCall)Conv32_1/StatefulPartitionedCall:output:0conv32_2_5884conv32_2_5886*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv32_2_layer_call_and_return_conditional_losses_5528?
Pool_2/PartitionedCallPartitionedCall)Conv32_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_2_layer_call_and_return_conditional_losses_5538?
 Conv16_1/StatefulPartitionedCallStatefulPartitionedCallPool_2/PartitionedCall:output:0conv16_1_5890conv16_1_5892*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv16_1_layer_call_and_return_conditional_losses_5551?
 Conv16_2/StatefulPartitionedCallStatefulPartitionedCall)Conv16_1/StatefulPartitionedCall:output:0conv16_2_5895conv16_2_5897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv16_2_layer_call_and_return_conditional_losses_5568?
Pool_3/PartitionedCallPartitionedCall)Conv16_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_3_layer_call_and_return_conditional_losses_5578?
Flatten/PartitionedCallPartitionedCallPool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Flatten_layer_call_and_return_conditional_losses_5586?
dropout/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5735?
Dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
dense_5903
dense_5905*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Dense_layer_call_and_return_conditional_losses_5606?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&Dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5702?
"Classifier/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0classifier_5909classifier_5911*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Classifier_layer_call_and_return_conditional_losses_5630z
IdentityIdentity+Classifier/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^Classifier/StatefulPartitionedCall!^Conv16_1/StatefulPartitionedCall!^Conv16_2/StatefulPartitionedCall!^Conv32_1/StatefulPartitionedCall!^Conv32_2/StatefulPartitionedCall!^Conv64_1/StatefulPartitionedCall!^Conv64_2/StatefulPartitionedCall^Dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2H
"Classifier/StatefulPartitionedCall"Classifier/StatefulPartitionedCall2D
 Conv16_1/StatefulPartitionedCall Conv16_1/StatefulPartitionedCall2D
 Conv16_2/StatefulPartitionedCall Conv16_2/StatefulPartitionedCall2D
 Conv32_1/StatefulPartitionedCall Conv32_1/StatefulPartitionedCall2D
 Conv32_2/StatefulPartitionedCall Conv32_2/StatefulPartitionedCall2D
 Conv64_1/StatefulPartitionedCall Conv64_1/StatefulPartitionedCall2D
 Conv64_2/StatefulPartitionedCall Conv64_2/StatefulPartitionedCall2>
Dense/StatefulPartitionedCallDense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?c
?
?__inference_model_layer_call_and_return_conditional_losses_6364

inputsA
'conv64_1_conv2d_readvariableop_resource:@6
(conv64_1_biasadd_readvariableop_resource:@A
'conv64_2_conv2d_readvariableop_resource:@@6
(conv64_2_biasadd_readvariableop_resource:@A
'conv32_1_conv2d_readvariableop_resource:@ 6
(conv32_1_biasadd_readvariableop_resource: A
'conv32_2_conv2d_readvariableop_resource:  6
(conv32_2_biasadd_readvariableop_resource: A
'conv16_1_conv2d_readvariableop_resource: 6
(conv16_1_biasadd_readvariableop_resource:A
'conv16_2_conv2d_readvariableop_resource:6
(conv16_2_biasadd_readvariableop_resource:9
$dense_matmul_readvariableop_resource:???4
%dense_biasadd_readvariableop_resource:	?<
)classifier_matmul_readvariableop_resource:	?8
*classifier_biasadd_readvariableop_resource:
identity??!Classifier/BiasAdd/ReadVariableOp? Classifier/MatMul/ReadVariableOp?Conv16_1/BiasAdd/ReadVariableOp?Conv16_1/Conv2D/ReadVariableOp?Conv16_2/BiasAdd/ReadVariableOp?Conv16_2/Conv2D/ReadVariableOp?Conv32_1/BiasAdd/ReadVariableOp?Conv32_1/Conv2D/ReadVariableOp?Conv32_2/BiasAdd/ReadVariableOp?Conv32_2/Conv2D/ReadVariableOp?Conv64_1/BiasAdd/ReadVariableOp?Conv64_1/Conv2D/ReadVariableOp?Conv64_2/BiasAdd/ReadVariableOp?Conv64_2/Conv2D/ReadVariableOp?Dense/BiasAdd/ReadVariableOp?Dense/MatMul/ReadVariableOpU
Rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *???;W
Rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    s
Rescaling/mulMulinputsRescaling/Cast/x:output:0*
T0*1
_output_shapes
:????????????
Rescaling/addAddV2Rescaling/mul:z:0Rescaling/Cast_1/x:output:0*
T0*1
_output_shapes
:????????????
Conv64_1/Conv2D/ReadVariableOpReadVariableOp'conv64_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv64_1/Conv2DConv2DRescaling/add:z:0&Conv64_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
Conv64_1/BiasAdd/ReadVariableOpReadVariableOp(conv64_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Conv64_1/BiasAddBiasAddConv64_1/Conv2D:output:0'Conv64_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@l
Conv64_1/ReluReluConv64_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
Conv64_2/Conv2D/ReadVariableOpReadVariableOp'conv64_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv64_2/Conv2DConv2DConv64_1/Relu:activations:0&Conv64_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
?
Conv64_2/BiasAdd/ReadVariableOpReadVariableOp(conv64_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
Conv64_2/BiasAddBiasAddConv64_2/Conv2D:output:0'Conv64_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@l
Conv64_2/ReluReluConv64_2/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@?
Pool_1/MaxPoolMaxPoolConv64_2/Relu:activations:0*1
_output_shapes
:???????????@*
ksize
*
paddingVALID*
strides
?
Conv32_1/Conv2D/ReadVariableOpReadVariableOp'conv32_1_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv32_1/Conv2DConv2DPool_1/MaxPool:output:0&Conv32_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
Conv32_1/BiasAdd/ReadVariableOpReadVariableOp(conv32_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv32_1/BiasAddBiasAddConv32_1/Conv2D:output:0'Conv32_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? l
Conv32_1/ReluReluConv32_1/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? ?
Conv32_2/Conv2D/ReadVariableOpReadVariableOp'conv32_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv32_2/Conv2DConv2DConv32_1/Relu:activations:0&Conv32_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
Conv32_2/BiasAdd/ReadVariableOpReadVariableOp(conv32_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
Conv32_2/BiasAddBiasAddConv32_2/Conv2D:output:0'Conv32_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? l
Conv32_2/ReluReluConv32_2/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? ?
Pool_2/MaxPoolMaxPoolConv32_2/Relu:activations:0*/
_output_shapes
:?????????@@ *
ksize
*
paddingVALID*
strides
?
Conv16_1/Conv2D/ReadVariableOpReadVariableOp'conv16_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv16_1/Conv2DConv2DPool_2/MaxPool:output:0&Conv16_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
Conv16_1/BiasAdd/ReadVariableOpReadVariableOp(conv16_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Conv16_1/BiasAddBiasAddConv16_1/Conv2D:output:0'Conv16_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@j
Conv16_1/ReluReluConv16_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
Conv16_2/Conv2D/ReadVariableOpReadVariableOp'conv16_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv16_2/Conv2DConv2DConv16_1/Relu:activations:0&Conv16_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
?
Conv16_2/BiasAdd/ReadVariableOpReadVariableOp(conv16_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Conv16_2/BiasAddBiasAddConv16_2/Conv2D:output:0'Conv16_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@j
Conv16_2/ReluReluConv16_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@?
Pool_3/MaxPoolMaxPoolConv16_2/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
^
Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? @  
Flatten/ReshapeReshapePool_3/MaxPool:output:0Flatten/Const:output:0*
T0*)
_output_shapes
:???????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout/dropout/MulMulFlatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*)
_output_shapes
:???????????]
dropout/dropout/ShapeShapeFlatten/Reshape:output:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype0*

seed**
seed2*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:????????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:????????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*)
_output_shapes
:????????????
Dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*!
_output_shapes
:???*
dtype0?
Dense/MatMulMatMuldropout/dropout/Mul_1:z:0#Dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
Dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Dense/BiasAddBiasAddDense/MatMul:product:0$Dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????]

Dense/ReluReluDense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶??
dropout_1/dropout/MulMulDense/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:??????????_
dropout_1/dropout/ShapeShapeDense/Relu:activations:0*
T0*
_output_shapes
:?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed**
seed2*e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
 Classifier/MatMul/ReadVariableOpReadVariableOp)classifier_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Classifier/MatMulMatMuldropout_1/dropout/Mul_1:z:0(Classifier/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!Classifier/BiasAdd/ReadVariableOpReadVariableOp*classifier_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Classifier/BiasAddBiasAddClassifier/MatMul:product:0)Classifier/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????l
Classifier/SoftmaxSoftmaxClassifier/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
IdentityIdentityClassifier/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp"^Classifier/BiasAdd/ReadVariableOp!^Classifier/MatMul/ReadVariableOp ^Conv16_1/BiasAdd/ReadVariableOp^Conv16_1/Conv2D/ReadVariableOp ^Conv16_2/BiasAdd/ReadVariableOp^Conv16_2/Conv2D/ReadVariableOp ^Conv32_1/BiasAdd/ReadVariableOp^Conv32_1/Conv2D/ReadVariableOp ^Conv32_2/BiasAdd/ReadVariableOp^Conv32_2/Conv2D/ReadVariableOp ^Conv64_1/BiasAdd/ReadVariableOp^Conv64_1/Conv2D/ReadVariableOp ^Conv64_2/BiasAdd/ReadVariableOp^Conv64_2/Conv2D/ReadVariableOp^Dense/BiasAdd/ReadVariableOp^Dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2F
!Classifier/BiasAdd/ReadVariableOp!Classifier/BiasAdd/ReadVariableOp2D
 Classifier/MatMul/ReadVariableOp Classifier/MatMul/ReadVariableOp2B
Conv16_1/BiasAdd/ReadVariableOpConv16_1/BiasAdd/ReadVariableOp2@
Conv16_1/Conv2D/ReadVariableOpConv16_1/Conv2D/ReadVariableOp2B
Conv16_2/BiasAdd/ReadVariableOpConv16_2/BiasAdd/ReadVariableOp2@
Conv16_2/Conv2D/ReadVariableOpConv16_2/Conv2D/ReadVariableOp2B
Conv32_1/BiasAdd/ReadVariableOpConv32_1/BiasAdd/ReadVariableOp2@
Conv32_1/Conv2D/ReadVariableOpConv32_1/Conv2D/ReadVariableOp2B
Conv32_2/BiasAdd/ReadVariableOpConv32_2/BiasAdd/ReadVariableOp2@
Conv32_2/Conv2D/ReadVariableOpConv32_2/Conv2D/ReadVariableOp2B
Conv64_1/BiasAdd/ReadVariableOpConv64_1/BiasAdd/ReadVariableOp2@
Conv64_1/Conv2D/ReadVariableOpConv64_1/Conv2D/ReadVariableOp2B
Conv64_2/BiasAdd/ReadVariableOpConv64_2/BiasAdd/ReadVariableOp2@
Conv64_2/Conv2D/ReadVariableOpConv64_2/Conv2D/ReadVariableOp2<
Dense/BiasAdd/ReadVariableOpDense/BiasAdd/ReadVariableOp2:
Dense/MatMul/ReadVariableOpDense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_Conv16_1_layer_call_and_return_conditional_losses_6517

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?

`
A__inference_dropout_layer_call_and_return_conditional_losses_6595

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype0*

seed**
seed2*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:???????????q
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:???????????k
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:???????????[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
\
@__inference_Pool_1_layer_call_and_return_conditional_losses_6432

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_6134	
input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@ 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:???

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_5407o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameInput
?
]
A__inference_Flatten_layer_call_and_return_conditional_losses_5586

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? @  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  :W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_5987	
input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@ 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:???

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_5915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameInput
?

?
D__inference_Classifier_layer_call_and_return_conditional_losses_6662

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_Conv64_1_layer_call_fn_6386

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv64_1_layer_call_and_return_conditional_losses_5471y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_5617

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_Conv32_1_layer_call_fn_6446

inputs!
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv32_1_layer_call_and_return_conditional_losses_5511y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
\
@__inference_Pool_3_layer_call_and_return_conditional_losses_5440

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
\
@__inference_Pool_2_layer_call_and_return_conditional_losses_5538

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@@ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
'__inference_Conv16_2_layer_call_fn_6526

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv16_2_layer_call_and_return_conditional_losses_5568w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
\
@__inference_Pool_3_layer_call_and_return_conditional_losses_6552

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
\
@__inference_Pool_3_layer_call_and_return_conditional_losses_6557

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_6642

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed**
seed2*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
A
%__inference_Pool_3_layer_call_fn_6542

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_3_layer_call_and_return_conditional_losses_5440?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
B
&__inference_dropout_layer_call_fn_6573

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5593b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
(__inference_Rescaling_layer_call_fn_6369

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Rescaling_layer_call_and_return_conditional_losses_5458j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_Conv32_2_layer_call_and_return_conditional_losses_5528

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?

?
?__inference_Dense_layer_call_and_return_conditional_losses_5606

inputs3
matmul_readvariableop_resource:???.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:???*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_1_layer_call_fn_6625

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_5702p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_Conv16_1_layer_call_fn_6506

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv16_1_layer_call_and_return_conditional_losses_5551w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
B__inference_Conv16_2_layer_call_and_return_conditional_losses_6537

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
A
%__inference_Pool_1_layer_call_fn_6422

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_1_layer_call_and_return_conditional_losses_5416?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_Conv64_1_layer_call_and_return_conditional_losses_5471

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_Conv16_2_layer_call_and_return_conditional_losses_5568

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_5702

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*

seed**
seed2*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_Conv32_1_layer_call_and_return_conditional_losses_5511

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
B__inference_Conv64_2_layer_call_and_return_conditional_losses_6417

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
$__inference_model_layer_call_fn_6208

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@#
	unknown_3:@ 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:???

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_5915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_6583

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:???????????]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
_
&__inference_dropout_layer_call_fn_6578

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5735q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_Conv64_2_layer_call_fn_6406

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Conv64_2_layer_call_and_return_conditional_losses_5488y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
B__inference_Conv64_2_layer_call_and_return_conditional_losses_5488

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:???????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_5593

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:???????????]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
A
%__inference_Pool_2_layer_call_fn_6482

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Pool_2_layer_call_and_return_conditional_losses_5428?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
\
@__inference_Pool_1_layer_call_and_return_conditional_losses_5416

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
\
@__inference_Pool_2_layer_call_and_return_conditional_losses_6497

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????@@ *
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
Input8
serving_default_Input:0???????????>

Classifier0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
7	variables
8trainable_variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

;kernel
<bias
=	variables
>trainable_variables
?regularization_losses
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

]kernel
^bias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
citer

dbeta_1

ebeta_2
	fdecay
glearning_ratem?m?!m?"m?+m?,m?1m?2m?;m?<m?Am?Bm?Sm?Tm?]m?^m?v?v?!v?"v?+v?,v?1v?2v?;v?<v?Av?Bv?Sv?Tv?]v?^v?"
	optimizer
?
0
1
!2
"3
+4
,5
16
27
;8
<9
A10
B11
S12
T13
]14
^15"
trackable_list_wrapper
?
0
1
!2
"3
+4
,5
16
27
;8
<9
A10
B11
S12
T13
]14
^15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@2Conv64_1/kernel
:@2Conv64_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@@2Conv64_2/kernel
:@2Conv64_2/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
#	variables
$trainable_variables
%regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
'	variables
(trainable_variables
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@ 2Conv32_1/kernel
: 2Conv32_1/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
-	variables
.trainable_variables
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'  2Conv32_2/kernel
: 2Conv32_2/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' 2Conv16_1/kernel
:2Conv16_1/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'2Conv16_2/kernel
:2Conv16_2/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:???2Dense/kernel
:?2
Dense/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"	?2Classifier/kernel
:2Classifier/bias
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
_	variables
`trainable_variables
aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
?
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
10
11
12
13
14
15"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
.:,@2Adam/Conv64_1/kernel/m
 :@2Adam/Conv64_1/bias/m
.:,@@2Adam/Conv64_2/kernel/m
 :@2Adam/Conv64_2/bias/m
.:,@ 2Adam/Conv32_1/kernel/m
 : 2Adam/Conv32_1/bias/m
.:,  2Adam/Conv32_2/kernel/m
 : 2Adam/Conv32_2/bias/m
.:, 2Adam/Conv16_1/kernel/m
 :2Adam/Conv16_1/bias/m
.:,2Adam/Conv16_2/kernel/m
 :2Adam/Conv16_2/bias/m
&:$???2Adam/Dense/kernel/m
:?2Adam/Dense/bias/m
):'	?2Adam/Classifier/kernel/m
": 2Adam/Classifier/bias/m
.:,@2Adam/Conv64_1/kernel/v
 :@2Adam/Conv64_1/bias/v
.:,@@2Adam/Conv64_2/kernel/v
 :@2Adam/Conv64_2/bias/v
.:,@ 2Adam/Conv32_1/kernel/v
 : 2Adam/Conv32_1/bias/v
.:,  2Adam/Conv32_2/kernel/v
 : 2Adam/Conv32_2/bias/v
.:, 2Adam/Conv16_1/kernel/v
 :2Adam/Conv16_1/bias/v
.:,2Adam/Conv16_2/kernel/v
 :2Adam/Conv16_2/bias/v
&:$???2Adam/Dense/kernel/v
:?2Adam/Dense/bias/v
):'	?2Adam/Classifier/kernel/v
": 2Adam/Classifier/bias/v
?2?
$__inference_model_layer_call_fn_5672
$__inference_model_layer_call_fn_6171
$__inference_model_layer_call_fn_6208
$__inference_model_layer_call_fn_5987?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_model_layer_call_and_return_conditional_losses_6279
?__inference_model_layer_call_and_return_conditional_losses_6364
?__inference_model_layer_call_and_return_conditional_losses_6038
?__inference_model_layer_call_and_return_conditional_losses_6089?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_5407Input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_Rescaling_layer_call_fn_6369?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Rescaling_layer_call_and_return_conditional_losses_6377?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Conv64_1_layer_call_fn_6386?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Conv64_1_layer_call_and_return_conditional_losses_6397?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Conv64_2_layer_call_fn_6406?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Conv64_2_layer_call_and_return_conditional_losses_6417?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_Pool_1_layer_call_fn_6422
%__inference_Pool_1_layer_call_fn_6427?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_Pool_1_layer_call_and_return_conditional_losses_6432
@__inference_Pool_1_layer_call_and_return_conditional_losses_6437?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Conv32_1_layer_call_fn_6446?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Conv32_1_layer_call_and_return_conditional_losses_6457?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Conv32_2_layer_call_fn_6466?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Conv32_2_layer_call_and_return_conditional_losses_6477?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_Pool_2_layer_call_fn_6482
%__inference_Pool_2_layer_call_fn_6487?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_Pool_2_layer_call_and_return_conditional_losses_6492
@__inference_Pool_2_layer_call_and_return_conditional_losses_6497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Conv16_1_layer_call_fn_6506?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Conv16_1_layer_call_and_return_conditional_losses_6517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Conv16_2_layer_call_fn_6526?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Conv16_2_layer_call_and_return_conditional_losses_6537?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_Pool_3_layer_call_fn_6542
%__inference_Pool_3_layer_call_fn_6547?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_Pool_3_layer_call_and_return_conditional_losses_6552
@__inference_Pool_3_layer_call_and_return_conditional_losses_6557?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_Flatten_layer_call_fn_6562?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_Flatten_layer_call_and_return_conditional_losses_6568?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dropout_layer_call_fn_6573
&__inference_dropout_layer_call_fn_6578?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dropout_layer_call_and_return_conditional_losses_6583
A__inference_dropout_layer_call_and_return_conditional_losses_6595?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_Dense_layer_call_fn_6604?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_Dense_layer_call_and_return_conditional_losses_6615?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dropout_1_layer_call_fn_6620
(__inference_dropout_1_layer_call_fn_6625?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_1_layer_call_and_return_conditional_losses_6630
C__inference_dropout_1_layer_call_and_return_conditional_losses_6642?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_Classifier_layer_call_fn_6651?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_Classifier_layer_call_and_return_conditional_losses_6662?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_6134Input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
D__inference_Classifier_layer_call_and_return_conditional_losses_6662]]^0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_Classifier_layer_call_fn_6651P]^0?-
&?#
!?
inputs??????????
? "???????????
B__inference_Conv16_1_layer_call_and_return_conditional_losses_6517l;<7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@
? ?
'__inference_Conv16_1_layer_call_fn_6506_;<7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@?
B__inference_Conv16_2_layer_call_and_return_conditional_losses_6537lAB7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
'__inference_Conv16_2_layer_call_fn_6526_AB7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
B__inference_Conv32_1_layer_call_and_return_conditional_losses_6457p+,9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0??????????? 
? ?
'__inference_Conv32_1_layer_call_fn_6446c+,9?6
/?,
*?'
inputs???????????@
? ""???????????? ?
B__inference_Conv32_2_layer_call_and_return_conditional_losses_6477p129?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
'__inference_Conv32_2_layer_call_fn_6466c129?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
B__inference_Conv64_1_layer_call_and_return_conditional_losses_6397p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????@
? ?
'__inference_Conv64_1_layer_call_fn_6386c9?6
/?,
*?'
inputs???????????
? ""????????????@?
B__inference_Conv64_2_layer_call_and_return_conditional_losses_6417p!"9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
'__inference_Conv64_2_layer_call_fn_6406c!"9?6
/?,
*?'
inputs???????????@
? ""????????????@?
?__inference_Dense_layer_call_and_return_conditional_losses_6615_ST1?.
'?$
"?
inputs???????????
? "&?#
?
0??????????
? z
$__inference_Dense_layer_call_fn_6604RST1?.
'?$
"?
inputs???????????
? "????????????
A__inference_Flatten_layer_call_and_return_conditional_losses_6568b7?4
-?*
(?%
inputs?????????  
? "'?$
?
0???????????
? 
&__inference_Flatten_layer_call_fn_6562U7?4
-?*
(?%
inputs?????????  
? "?????????????
@__inference_Pool_1_layer_call_and_return_conditional_losses_6432?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
@__inference_Pool_1_layer_call_and_return_conditional_losses_6437l9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0???????????@
? ?
%__inference_Pool_1_layer_call_fn_6422?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_Pool_1_layer_call_fn_6427_9?6
/?,
*?'
inputs???????????@
? ""????????????@?
@__inference_Pool_2_layer_call_and_return_conditional_losses_6492?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
@__inference_Pool_2_layer_call_and_return_conditional_losses_6497j9?6
/?,
*?'
inputs??????????? 
? "-?*
#? 
0?????????@@ 
? ?
%__inference_Pool_2_layer_call_fn_6482?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_Pool_2_layer_call_fn_6487]9?6
/?,
*?'
inputs??????????? 
? " ??????????@@ ?
@__inference_Pool_3_layer_call_and_return_conditional_losses_6552?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
@__inference_Pool_3_layer_call_and_return_conditional_losses_6557h7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????  
? ?
%__inference_Pool_3_layer_call_fn_6542?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_Pool_3_layer_call_fn_6547[7?4
-?*
(?%
inputs?????????@@
? " ??????????  ?
C__inference_Rescaling_layer_call_and_return_conditional_losses_6377l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
(__inference_Rescaling_layer_call_fn_6369_9?6
/?,
*?'
inputs???????????
? ""?????????????
__inference__wrapped_model_5407?!"+,12;<ABST]^8?5
.?+
)?&
Input???????????
? "7?4
2

Classifier$?!

Classifier??????????
C__inference_dropout_1_layer_call_and_return_conditional_losses_6630^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_6642^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? }
(__inference_dropout_1_layer_call_fn_6620Q4?1
*?'
!?
inputs??????????
p 
? "???????????}
(__inference_dropout_1_layer_call_fn_6625Q4?1
*?'
!?
inputs??????????
p
? "????????????
A__inference_dropout_layer_call_and_return_conditional_losses_6583`5?2
+?(
"?
inputs???????????
p 
? "'?$
?
0???????????
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_6595`5?2
+?(
"?
inputs???????????
p
? "'?$
?
0???????????
? }
&__inference_dropout_layer_call_fn_6573S5?2
+?(
"?
inputs???????????
p 
? "????????????}
&__inference_dropout_layer_call_fn_6578S5?2
+?(
"?
inputs???????????
p
? "?????????????
?__inference_model_layer_call_and_return_conditional_losses_6038{!"+,12;<ABST]^@?=
6?3
)?&
Input???????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_6089{!"+,12;<ABST]^@?=
6?3
)?&
Input???????????
p

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_6279|!"+,12;<ABST]^A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
?__inference_model_layer_call_and_return_conditional_losses_6364|!"+,12;<ABST]^A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
$__inference_model_layer_call_fn_5672n!"+,12;<ABST]^@?=
6?3
)?&
Input???????????
p 

 
? "???????????
$__inference_model_layer_call_fn_5987n!"+,12;<ABST]^@?=
6?3
)?&
Input???????????
p

 
? "???????????
$__inference_model_layer_call_fn_6171o!"+,12;<ABST]^A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
$__inference_model_layer_call_fn_6208o!"+,12;<ABST]^A?>
7?4
*?'
inputs???????????
p

 
? "???????????
"__inference_signature_wrapper_6134?!"+,12;<ABST]^A?>
? 
7?4
2
Input)?&
Input???????????"7?4
2

Classifier$?!

Classifier?????????