import onnx
import torch
from onnx import helper
from onnx import TensorProto

"""onnx流程"""
# # calculate output = a * x + b
# # ValueInfoProto
# a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
# x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
# b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
# output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 10])
#
# # NodeProto
# mul = helper.make_node('Mul', ['a', 'x'], ['c'])
# add = helper.make_node('Add', ['c', 'b'], ['output'])
#
# # GraphProto, 计算图节点必须以拓扑序给出。[mul, add], 而不是[add, mul]
# graph = helper.make_graph([mul, add], 'linear_func', [a, x, b], [output])
#
# # ModelProto
# model = helper.make_model(graph)
#
# # check model
# onnx.checker.check_model(model)
# print(model)
# onnx.save(model, 'linear_func.onnx')


"""打印测试模型"""
# # 读取模型
# model1 = onnx.load('linear_func.onnx')
# graph = model1.graph
# node = graph.node
# input = graph.input
# output = graph.output
#
# # 直接修改模型属性，不违反onnx规范的前提下
# model2 = onnx.load('linear_func.onnx')
# node1 = model2.graph.node
# node1[1].op_type = 'Sub'  # Add to Sub
#
# onnx.checker.check_model(model2)
# onnx.save_model(model2, 'linear_func_1.onnx')


"""Debugging onnx model"""
# # model
# class Model(torch.nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.convs1 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
#                                           torch.nn.Conv2d(3, 3, 3),
#                                           torch.nn.Conv2d(3, 3, 3))
#         self.convs2 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
#                                           torch.nn.Conv2d(3, 3, 3))
#         self.convs3 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
#                                           torch.nn.Conv2d(3, 3, 3))
#         self.convs4 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
#                                           torch.nn.Conv2d(3, 3, 3),
#                                           torch.nn.Conv2d(3, 3, 3))
#
#     def forward(self, x):
#         x = self.convs1(x)
#         x1 = self.convs2(x)
#         x2 = self.convs3(x)
#         x = x1 + x2
#         x = self.convs4(x)
#         return x
#
# model = Model()
# input = torch.randn(1, 3, 20, 20)
#
# torch.onnx.export(model, input, 'whole_model.onnx')

# submodel extract, 根据节点序号提取
# onnx.utils.extract_model('whole_model.onnx', 'partial_model.onnx', ['input.4'], ['input.20'])

# 添加额外输出, 在使用 ONNX 模型时，最常见的一个需求是能够用推理引擎输出中间节点的值。
# 这多见于深度学习框架模型和 ONNX 模型的精度对齐中，因为只要能够输出中间节点的值，就能定位到精度出现偏差的算子。
onnx.utils.extract_model('whole_model.onnx', 'submodel_1.onnx', ['input.4'], ['onnx::Add_27','input.20'])
