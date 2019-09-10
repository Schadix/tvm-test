import os
import numpy as np
import tvm
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
from gluoncv import model_zoo, data, utils
import os
import warnings
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from gluoncv.model_zoo.yolo.darknet import _conv2d, darknet53
from gluoncv.model_zoo.mobilenet import get_mobilenet
from gluoncv.model_zoo.yolo.yolo_target import YOLOV3TargetMerger
from gluoncv.model_zoo.yolo import get_yolov3
from gluoncv.loss import YOLOV3Loss

def yolo3_darknet53_custom(classes, transfer=None, pretrained_base=True, pretrained=False,
                           norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with darknet53 base network on custom dataset.
    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from yolo networks trained on other
        datasets.
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        base_net = darknet53(
            pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]] #256, #512 #1024
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]]
        anchors_half = [
            [  5,   6,   8,  15,  16,  11],
            [ 15,  30,  31,  22,  29,  59],
            [ 58,  45,  78,  99, 186, 163]]
        anchors_fourth = [
            [ 2,  3,  4,  7,  8,  5],
            [ 7, 15, 15, 11, 14, 29],
            [29, 22, 39, 49, 93, 81]    ]
        strides = [8, 16, 32]
        net = get_yolov3(
            'darknet53', stages, [512, 256, 128], anchors, strides, classes, '',
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    else:
        from gluoncv.model_zoo import get_model
        net = get_model('yolo3_darknet53_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net


def get_network(name, params='retrain_yolo3_darknet_pig_best.params'):
    """Get the symbol definition and random weight of a network"""
    input_shape = (1, 3, 512, 512)
    #net = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
    net = yolo3_darknet53_custom(['pigs'], pretrained=False, pretrained_base=False)
    net.load_parameters(params)
    dshape = (1, 3, 512, 512)
    mod, params = relay.frontend.from_mxnet(net, {"data": dshape})
    return mod, params, input_shape

#### DEVICE CONFIG ####
target = tvm.target.cuda()
target_host = 'llvm -target=aarch64-linux-gnu'

#### TUNING OPTION ####
network = 'darknet53'
log_file = "%s.log" % network
dtype = 'float32'

tuning_opt = tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        #runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        runner=autotvm.RPCRunner(
            'nano',  # change the device key to your key
            '0.0.0.0', 8192,
            number=20, repeat=3, timeout=4, min_repeat_ms=150)
    ),
}
tuning_opt = tuning_option
# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)
        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])
    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def tune_and_evaluate(tuning_opt, target_host="llvm -target=aarch64-linux-gnu"):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape = get_network('test', params='retrain_yolo3_darknet_pig_best.params')
    if target_host:
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, target_host=target_host,
                                              params=params, ops=(relay.op.nn.conv2d,))
    else:
        tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params, ops=(relay.op.nn.conv2d,))
    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)
    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)
        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))
        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)
        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))