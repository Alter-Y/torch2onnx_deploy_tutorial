import torch

"""trace and script"""
# class Model(torch.nn.Module):
#     def __init__(self, n):
#         super().__init__()
#         self.n = n
#         self.conv = torch.nn.Conv2d(3, 3, 3)
#
#     def forward(self, x):
#         for i in range(self.n):
#             x = self.conv(x)
#         return x
#
# models = [Model(2), Model(3)]
# model_names = ['model_2', 'model_3']
#
# for model, model_name in zip(models, model_names):
#     dummy_input = torch.rand(1, 3, 10, 10)
#     dummy_output = model(dummy_input)
#     model_trace = torch.jit.trace(model, dummy_input)
#     model_script = torch.jit.script(model)
#
#     # 跟踪法与直接 torch.onnx.export(model, ...)等价
#     torch.onnx.export(model_trace, dummy_input, f'{model_name}_trace.onnx')
#     # 记录法必须先调用 torch.jit.sciprt
#     torch.onnx.export(model_script, dummy_input, f'{model_name}_script.onnx')


"""asinh"""
import torchvision
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return torch.asinh(x)
#
# from torch.onnx.symbolic_registry import register_op
#
# def asinh_symbolic(g, input, *, out=None):
#     return g.op("Asinh", input)
#
# register_op('asinh', asinh_symbolic, '', 9)
#
# model = Model()
# input = torch.rand(1, 3, 10, 10)
# torch.onnx.export(model, input, 'asinh.onnx')


"""deformconv2d"""
# class Model(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = torch.nn.Conv2d(3, 18, 3)
#         self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)
#
#     def forward(self, x):
#         return self.conv2(x, self.conv1(x))
#
# from torch.onnx import register_custom_op_symbolic
# from torch.onnx.symbolic_helper import parse_args
#
# # torchscript算子的符号函数要求标注出每一个输入参数的类型。需要装饰器，@parse_args
# @parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none")
# # 符号函数，前向推理接口参数：查看的torchvision的源码定义
# def symbolic(g,
#              input,
#              weight,
#              offset,
#              mask,
#              bias,
#              stride_h,
#              stride_w,
#              pad_h,
#              pad_w,
#              dil_h,
#              dil_w,
#              n_weight_grps,
#              n_offset_grps,
#              use_mask,
#              ):
#     # 定义onnx算子的函数，对于自定义算子，第一个参数是一个带命名空间的算子名。
#     # "::"前面的内容就是我们的命名空间。该概念和 C++ 的命名空间类似，是为了防止命名冲突而设定的。
#     # 如果在 g.op() 里不加前面的命名空间，则算子会被默认成 ONNX 的官方算子。
#     return g.op("custom::deform_conv2d", input, offset)
#
# # 不同于前面asinh，自定义算子符号函数注册方式用这个库。
# register_custom_op_symbolic("torchvision::deform_conv2d", symbolic, 9)
#
# model = Model()
# input = torch.rand(1, 3, 10, 10)
# torch.onnx.export(model, input, 'dcn.onnx')


"""2a+b ops"""
import my_lib

# Function类本身表示为pytorch的一个可导函数。pytorch会自动调度该函数，合适的执行前向和反向计算。
# 如果定义了symbolic的静态方法，该Function在执行torch.onnx.export()时，会根据symbolic中定义的规则转换成onnx算子。即符号函数。
class MyAddFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        return my_lib.my_add(a, b)

    @staticmethod
    def symbolic(g, a, b):
        two = g.op("Constant", value_t=torch.tensor([2])) # 对于op中参数，查pytorch中onnx算子映射规则。https://github.com/pytorch/pytorch/blob/master/torch/onnx/
        a = g.op("Mul", a, two)
        return g.op("Add", a, b)

# apply是torch.autograd.Function 的一个方法，这个方法完成了 Function 在前向推理或者反向传播时的调度。
# 我们在使用 Function 的派生类做推理时，不应该显式地调用 forward()，而应该调用其 apply 方法。
my_add = MyAddFunction.apply
class MyAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return my_add(a, b)

model = MyAdd()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, (input, input), 'my_add.onnx')