import os


def set_cpu_option():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


