"?>
BHostIDLE"IDLE1     n?@A     n?@ap?|D???ip?|D????Unknown
?HostTfqAdjointGradient"RAdam/gradients/cond/else/_9/Adam/gradients/cond/PartitionedCall/TfqAdjointGradient(1     d?@9     d?@A     d?@I     d?@a?#W?????if?b?B????Unknown
?HostTfqSimulateExpectation"7sequential_2/pqc_2/expectation_2/TfqSimulateExpectation(1     ??@9     ??@A     ??@I     ??@a8q9It??i????T????Unknown
?HostTfqAppendCircuit"1sequential_2/pqc_2/add_circuit_2/TfqAppendCircuit(1     ?@9     ?@A     ?@I     ?@aa???Њ??i ???y???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?~@9     ?~@A     ?~@I     ?~@a?7HBs???i?????????Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      O@9      O@A      O@I      O@a?˞??Q?ilB?7?????Unknown
vHost_FusedMatMul"sequential_2/dense_1/BiasAdd(1      I@9      I@A      I@I      I@a?c???tL?i?b?s?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      A@9      A@A      A@I      A@aNK?YC?i?&I??????Unknown
s	HostDataset"Iterator::Model::ParallelMapV2(1      @@9      @@A      @@I      @@aV+??\6B?id?i{L????Unknown
i
HostWriteSummary"WriteSummary(1      7@9      7@A      7@I      7@aK%?$.:?id	@?????Unknown?
rHostSelectV2"gradient_tape/hinge/SelectV2(1      5@9      5@A      5@I      5@a???kY?7?i#?6+?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      2@9      2@A      2@I      2@a?p>(}4?i?D9?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      0@I      0@aV+??\62?i6?ɛe????Unknown
^HostGatherV2"GatherV2(1      ,@9      ,@A      ,@I      ,@a???!?/?i????c????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      *@I      *@ak??TV?-?i?<M=????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      $@9      $@A      $@I      $@a+?????&?i?v?R?????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      $@9      $@A      $@I      $@a+?????&?iA???????Unknown
dHostDataset"Iterator::Model(1      D@9      D@A       @I       @aV+??\6"?idމ?8????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @aV+??\6"?i?R]\????Unknown?
[HostAddV2"Adam/add(1      @9      @A      @I      @a???!??i?4aV[????Unknown
^HostGreater"	Greater_1(1      @9      @A      @I      @a???!??iC]pOZ????Unknown
?HostUnpack"5gradient_tape/sequential_2/pqc_2/Tile_1/input/unstack(1      @9      @A      @I      @a ASĊQ?i???4????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a+??????i??c??????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      F@9      F@A      @I      @a+??????i?? ?????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @a+??????i_֝:W????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a+??????i5?:Z????Unknown
HostMatMul"+gradient_tape/sequential_2/dense_1/MatMul_1(1      @9      @A      @I      @a+??????i?y?????Unknown
~HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @aV+??\6?i'?,U????Unknown
vHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @aV+??\6?i->???????Unknown
oHostReadVariableOp"Adam/ReadVariableOp(1      @9      @A      @I      @aV+??\6?i>U??x????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @aV+??\6?iOlhE
????Unknown
u HostSum"$gradient_tape/sequential_2/pqc_2/Sum(1      @9      @A      @I      @aV+??\6?i`?L??????Unknown
Y!HostPow"Adam/Pow(1      @9      @A      @I      @a ASĊQ?i??w>	????Unknown
\"HostGreater"Greater(1      @9      @A      @I      @a ASĊQ?i????v????Unknown
z#HostGreaterEqual" gradient_tape/hinge/GreaterEqual(1      @9      @A      @I      @a ASĊQ?iG????????Unknown
?$HostBiasAddGrad"6gradient_tape/sequential_2/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a ASĊQ?i???Q????Unknown
^%HostEqual"hinge/Equal(1      @9      @A      @I      @a ASĊQ?i??#W?????Unknown
`&HostEqual"hinge/Equal_1(1      @9      @A      @I      @a ASĊQ?i.?N?+????Unknown
r'HostMul"!hinge/cond/then/_0/hinge/cond/mul(1      @9      @A      @I      @a ASĊQ?i{?y??????Unknown
t(HostReadVariableOp"Adam/Cast/ReadVariableOp(1       @9       @A       @I       @aV+??\6?i???????Unknown
])HostCast"Adam/Cast_1(1       @9       @A       @I       @aV+??\6?i?^?*????Unknown
[*HostPow"
Adam/Pow_1(1       @9       @A       @I       @aV+??\6?i?os????Unknown
?+HostCast"~ArithmeticOptimizer/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_ReorderCastLikeAndValuePreserving_float_Cast_1(1       @9       @A       @I       @aV+??\6?i?*BI?????Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @aV+??\6?i(6?"????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @aV+??\6?i?A&?M????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @aV+??\6?i:M?Ֆ????Unknown
X/HostEqual"Equal(1       @9       @A       @I       @aV+??\6?i?X
??????Unknown
b0HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @aV+??\6?iLd|?(????Unknown
y1HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @aV+??\6?i?o?aq????Unknown
}2HostMatMul")gradient_tape/sequential_2/dense_1/MatMul(1       @9       @A       @I       @aV+??\6?i^{`;?????Unknown
Z3HostAll"	hinge/All(1       @9       @A       @I       @aV+??\6?i???????Unknown
f4Host	LogicalOr"hinge/LogicalOr(1       @9       @A       @I       @aV+??\6?ip?D?K????Unknown
b5HostMaximum"hinge/Maximum(1       @9       @A       @I       @aV+??\6?i???ǔ????Unknown
r6HostSub"!hinge/cond/then/_0/hinge/cond/sub(1       @9       @A       @I       @aV+??\6?i??(??????Unknown
Z7HostMul"	hinge/mul(1       @9       @A       @I       @aV+??\6?i??z&????Unknown
Z8HostSub"	hinge/sub(1       @9       @A       @I       @aV+??\6?i??To????Unknown
h9HostSum"hinge/weighted_loss/Sum(1       @9       @A       @I       @aV+??\6?i?~-?????Unknown
v:HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??aV+??\6?>i??7??????Unknown
a;HostIdentity"Identity(1      ??9      ??A      ??I      ??aV+??\6?>i???????Unknown?
`<HostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??aV+??\6?>iiݩs%????Unknown
w=HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??aV+??\6?>i-?b?I????Unknown
l>HostNeg"gradient_tape/hinge/sub/Neg(1      ??9      ??A      ??I      ??aV+??\6?>i??Mn????Unknown
o?HostDivNoNan"hinge/weighted_loss/value(1      ??9      ??A      ??I      ??aV+??\6?>i??Թ?????Unknown
?@HostReadVariableOp"+sequential_2/dense_1/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??aV+??\6?>iy??&?????Unknown
?AHostReadVariableOp"*sequential_2/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??aV+??\6?>i=?F??????Unknown
?BHostReadVariableOp".sequential_2/pqc_2/Tile_1/input/ReadVariableOp(1      ??9      ??A      ??I      ??aV+??\6?>i     ???Unknown
HCHostReadVariableOp"div_no_nan/ReadVariableOp(i     ???Unknown
JDHostReadVariableOp"div_no_nan_1/ReadVariableOp(i     ???Unknown
AEHostMul"gradient_tape/hinge/mul/Mul_1(i     ???Unknown*?=
?HostTfqAdjointGradient"RAdam/gradients/cond/else/_9/Adam/gradients/cond/PartitionedCall/TfqAdjointGradient(1     d?@9     d?@A     d?@I     d?@a#??˺/??i#??˺/???Unknown
?HostTfqSimulateExpectation"7sequential_2/pqc_2/expectation_2/TfqSimulateExpectation(1     ??@9     ??@A     ??@I     ??@a?f7?????iv^I?????Unknown
?HostTfqAppendCircuit"1sequential_2/pqc_2/add_circuit_2/TfqAppendCircuit(1     ?@9     ?@A     ?@I     ?@a????FJ??i? ??n????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ?~@9     ?~@A     ?~@I     ?~@a?\G?2???ir[??n???Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1      O@9      O@A      O@I      O@aU}_?? d?i???hт???Unknown
vHost_FusedMatMul"sequential_2/dense_1/BiasAdd(1      I@9      I@A      I@I      I@ae??!`?iT???????Unknown
lHostIteratorGetNext"IteratorGetNext(1      A@9      A@A      A@I      A@av???i?U?i?e?T?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1      @@9      @@A      @@I      @@a`? 	?T?i?uYY>????Unknown
i	HostWriteSummary"WriteSummary(1      7@9      7@A      7@I      7@a?????M?i?!??????Unknown?
r
HostSelectV2"gradient_tape/hinge/SelectV2(1      5@9      5@A      5@I      5@aΩ???K?i2?p????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1      2@9      2@A      2@I      2@a??$;?:G?iV?(2?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      5@9      5@A      0@I      0@a`? 	?D?iv?n?h????Unknown
^HostGatherV2"GatherV2(1      ,@9      ,@A      ,@I      ,@a4q?GB?i?$l?????Unknown
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1      2@9      2@A      *@I      *@ai?c??@?i,E?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1      $@9      $@A      $@I      $@a??(^??9?i@а?X????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      $@9      $@A      $@I      $@a??(^??9?iT???????Unknown
dHostDataset"Iterator::Model(1      D@9      D@A       @I       @a`? 	?4?id??d'????Unknown
eHost
LogicalAnd"
LogicalAnd(1       @9       @A       @I       @a`? 	?4?it?b%?????Unknown?
[HostAddV2"Adam/add(1      @9      @A      @I      @a4q?G2?iAaN?????Unknown
^HostGreater"	Greater_1(1      @9      @A      @I      @a4q?G2?i??_w@????Unknown
?HostUnpack"5gradient_tape/sequential_2/pqc_2/Tile_1/input/unstack(1      @9      @A      @I      @a?0??.?i?':0????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a??(^??)?i&
? ?????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1      F@9      F@A      @I      @a??(^??)?i????i????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      @9      @A      @I      @a??(^??)?i:?[?????Unknown
?HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a??(^??)?iı??????Unknown
HostMatMul"+gradient_tape/sequential_2/dense_1/MatMul_1(1      @9      @A      @I      @a??(^??)?iN???@????Unknown
~HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1      @9      @A      @I      @a`? 	?$?iVYD?????Unknown
vHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1      @9      @A      @I      @a`? 	?$?i^????????Unknown
oHostReadVariableOp"Adam/ReadVariableOp(1      @9      @A      @I      @a`? 	?$?if| ????Unknown
vHostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a`? 	?$?in?fj????Unknown
uHostSum"$gradient_tape/sequential_2/pqc_2/Sum(1      @9      @A      @I      @a`? 	?$?iv?ƴ????Unknown
Y HostPow"Adam/Pow(1      @9      @A      @I      @a?0???i????????Unknown
\!HostGreater"Greater(1      @9      @A      @I      @a?0???i?ayW?????Unknown
z"HostGreaterEqual" gradient_tape/hinge/GreaterEqual(1      @9      @A      @I      @a?0???i???????Unknown
?#HostBiasAddGrad"6gradient_tape/sequential_2/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?0???i??S??????Unknown
^$HostEqual"hinge/Equal(1      @9      @A      @I      @a?0???i????????Unknown
`%HostEqual"hinge/Equal_1(1      @9      @A      @I      @a?0???i??-y?????Unknown
r&HostMul"!hinge/cond/then/_0/hinge/cond/mul(1      @9      @A      @I      @a?0???i 	?A{????Unknown
t'HostReadVariableOp"Adam/Cast/ReadVariableOp(1       @9       @A       @I       @a`? 	??i$??q ????Unknown
](HostCast"Adam/Cast_1(1       @9       @A       @I       @a`? 	??i(?,??????Unknown
[)HostPow"
Adam/Pow_1(1       @9       @A       @I       @a`? 	??i,Lu?j????Unknown
?*HostCast"~ArithmeticOptimizer/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_ReorderCastLikeAndValuePreserving_float_Cast_1(1       @9       @A       @I       @a`? 	??i0?????Unknown
t+HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a`? 	??i4?3?????Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a`? 	??i8?OcZ????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_3(1       @9       @A       @I       @a`? 	??i<P???????Unknown
X.HostEqual"Equal(1       @9       @A       @I       @a`? 	??i@?ä????Unknown
b/HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a`? 	??iD?)?I????Unknown
y0HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a`? 	??iH?r$?????Unknown
}1HostMatMul")gradient_tape/sequential_2/dense_1/MatMul(1       @9       @A       @I       @a`? 	??iLT?T?????Unknown
Z2HostAll"	hinge/All(1       @9       @A       @I       @a`? 	??iP?9????Unknown
f3Host	LogicalOr"hinge/LogicalOr(1       @9       @A       @I       @a`? 	??iT?L??????Unknown
b4HostMaximum"hinge/Maximum(1       @9       @A       @I       @a`? 	??iX????????Unknown
r5HostSub"!hinge/cond/then/_0/hinge/cond/sub(1       @9       @A       @I       @a`? 	??i\X?)????Unknown
Z6HostMul"	hinge/mul(1       @9       @A       @I       @a`? 	??i`'F?????Unknown
Z7HostSub"	hinge/sub(1       @9       @A       @I       @a`? 	??id?ovs????Unknown
h8HostSum"hinge/weighted_loss/Sum(1       @9       @A       @I       @a`? 	??ih???????Unknown
v9HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??a`? 	??i???>k????Unknown
a:HostIdentity"Identity(1      ??9      ??A      ??I      ??a`? 	??il\׽????Unknown?
`;HostDivNoNan"
div_no_nan(1      ??9      ??A      ??I      ??a`? 	??i??%o????Unknown
w<HostReadVariableOp"div_no_nan/ReadVariableOp_1(1      ??9      ??A      ??I      ??a`? 	??ipJc????Unknown
l=HostNeg"gradient_tape/hinge/sub/Neg(1      ??9      ??A      ??I      ??a`? 	??i?}n??????Unknown
o>HostDivNoNan"hinge/weighted_loss/value(1      ??9      ??A      ??I      ??a`? 	??itޒ7????Unknown
??HostReadVariableOp"+sequential_2/dense_1/BiasAdd/ReadVariableOp(1      ??9      ??A      ??I      ??a`? 	??i?>??Z????Unknown
?@HostReadVariableOp"*sequential_2/dense_1/MatMul/ReadVariableOp(1      ??9      ??A      ??I      ??a`? 	??ix??g?????Unknown
?AHostReadVariableOp".sequential_2/pqc_2/Tile_1/input/ReadVariableOp(1      ??9      ??A      ??I      ??a`? 	??i?????????Unknown
HBHostReadVariableOp"div_no_nan/ReadVariableOp(i?????????Unknown
JCHostReadVariableOp"div_no_nan_1/ReadVariableOp(i?????????Unknown
ADHostMul"gradient_tape/hinge/mul/Mul_1(i?????????Unknown2CPU