import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np

def getOnesTensor(shape, name):
    values = np.ones(shape).flatten().astype(float)
    return helper.make_tensor(name=name, data_type=TensorProto.FLOAT, dims=shape, vals=values)

# create input
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 8, 5, 5])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [6, 4, 2, 2])

# create output
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 6, 4, 4])

# Convolution without padding
node_def = helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    groups=2,
    kernel_shape=[2, 2],
    strides=[1, 1],
    pads=[0, 0, 0, 0],
    # Default values for other attributes: dilations=[1, 1]
)

# create the graph
graph_def = helper.make_graph(
    [node_def],
    'test_group_Conv',
    [x, W],
    [y],
    [getOnesTensor([6, 4, 2, 2], 'W')]
)

# create the model
model_def = helper.make_model(
    graph_def,
    producer_name = 'onnc-tutorial'
)

onnx.save(model_def, 'test_group_Conv.onnx')
