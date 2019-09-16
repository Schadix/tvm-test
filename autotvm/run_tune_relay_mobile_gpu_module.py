import tvm
import numpy as np
import argparse

from tvm.contrib import graph_runtime

# input_shape = [1, 3, 512, 512]

def inference(module_path, json_path, params_path):
    # model_dir = "./"
    # dev_lib = tvm.module.load(os.path.join(model_dir, "tx2-cuda.ptx"))
    # loaded_lib.import_module(dev_lib)
    loaded_json = open(json_path).read()
    loaded_lib = tvm.module.load(module_path)
    loaded_params = bytearray(open(params_path, "rb").read())

    ctx = tvm.gpu(0)
    batch_size = 1
    input_shape = (batch_size, 3, 224, 224)
    dtype = 'float32'

    x = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))

    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.run(data=x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-path", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--params-path", required=True)

    args = parser.parse_args()

    inference(module_path=args.module_path, json_path=args.json_path, params_path=args.params_path)

