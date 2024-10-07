"""
目标：实现多进程，包括controller,worker,webui,fastapi这四个
"""

import argparse
import asyncio
import multiprocessing
import os
import subprocess
import sys
from fastapi import FastAPI
from config.log import *
import uvicorn
import fastchat

fastchat.constants.LOGDIR = os.path.join(".", LOG_PATH)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        action="store_true",
        help="run controller/worker/webui/fastapi server",
        dest="all_webui",
    )
    args = parser.parse_args()
    return args


def _set_app_event(app, started_event):
    @app.on_event("startup")
    async def on_start():
        if started_event is not None:
            started_event.set()


def create_controller_app():
    from fastchat.serve.controller import app, Controller

    controller = Controller("shortest_queue")
    sys.modules["fastchat.serve.controller"].controller = controller
    app.title = "YLangchain-Chatchat Controller Server"
    return app


def run_controller_server(started_event):
    from server.controller import get_controller_address

    # 创建controller app
    app = create_controller_app()

    @app.get("/test_controller")
    def controller_test():
        return {"code": 200, "msg": "contorller test"}

    # 获得contorller server的host和port
    host, port = get_controller_address()
    # 设置started_event事件，通知controller进程启动成功
    _set_app_event(app, started_event)
    # 启动controller app
    uvicorn.run(app, host=host, port=port)


def create_worker_app(**kwargs):
    """kwargs是一个字典，包含了：
    device:运行的设备 : cpu/cuda
    controller_address:controller的地址 :http://127.0.0.1:8000
    worker_address:worker的地址: http://127.0.0.1:8001
    model_names:模型名称的列表 :["Qwen2-0.5B-Instruct"]
    model_path:模型的路径 :r"E:\Code\YLangchain-Chatchat\models\Qwen2-0.5B-Instruct"
    """
    from fastchat.serve.model_worker import (
        app,
        GptqConfig,
        AWQConfig,
        ModelWorker,
        worker_id,
    )

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    for k, v in kwargs.items():
        setattr(args, k, v)

    args.gpus = "0"  # GPU的编号,如果有多个GPU，可以设置为"0,1,2,3"
    args.max_gpu_memory = "6GiB"
    args.num_gpus = 1  # model worker的切分是model并行，这里填写显卡的数量
    args.load_8bit = False  # 是否使用8bit模型
    args.cpu_offloading = None  # 是否使用CPU offloading，即将模型的部分加载到CPU上运行
    # gptq和awq是两种量化方式
    args.gptq_ckpt = None  # 指定量化模型的检查点路径，None表示不使用检查点
    args.gptq_wbits = 16  # gptq的量化位宽
    args.gptq_groupsize = -1  # gptq的量化组数，-1表示自动设置
    args.gptq_act_order = False  # gptq的激活顺序，False表示不启用
    args.awq_ckpt = None  # 指定awq量化模型的检查点路径，None表示不使用检查点
    args.awq_wbits = 16
    args.awq_groupsize = -1
    args.conv_template = None  # 指定对话模板
    args.limit_worker_concurrency = 5  # 限制每个worker的并发数
    args.stream_interval = 2  # 推理流水线间隔，单位秒
    args.no_register = False  # 是否禁止注册模型，False表示不排除注册，即允许注册
    args.embed_in_truncate = False  # 是否进行截断处理
    if args.gpus:
        if args.num_gpus is None:
            args.num_gpus = len(args.gpus.split(","))
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        groupsize=args.gptq_groupsize,
        act_order=args.gptq_act_order,
    )
    awq_config = AWQConfig(
        ckpt=args.awq_ckpt or args.model_path,
        wbits=args.awq_wbits,
        groupsize=args.awq_groupsize,
    )

    worker = ModelWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        model_path=args.model_path,
        model_names=args.model_names,
        limit_worker_concurrency=args.limit_worker_concurrency,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        stream_interval=args.stream_interval,
        conv_template=args.conv_template,
        embed_in_truncate=args.embed_in_truncate,
    )
    sys.modules["fastchat.serve.model_worker"].args = args
    # sys.modules["fastchat.serve.model_worker"].worker = worker
    app.title = "YLangchain-Chatchat Model Worker Server"
    return app


def run_worker_server(started_event):
    from server.controller import get_controller_address
    from server.model_worker import get_worker_address
    from server.utils import get_model_device

    controller_host, controller_port = get_controller_address()
    # 获得worker server的host和port
    worker_host, worker_port = get_worker_address()
    kwargs = {
        "device": get_model_device(),
        "controller_address": f"http://{controller_host}:{controller_port}",
        "worker_address": f"http://{worker_host}:{worker_port}",
        "model_names": ["Qwen2-0.5B-Instruct"],
        "model_path": r"E:\Code\YLangchain-Chatchat\models\Qwen2-0.5B-Instruct",
    }
    # 创建worker app
    app = create_worker_app(**kwargs)

    @app.get("/test_worker")
    def worker_test():
        return {"code": 200, "msg": "worker test"}

    # 设置started_event事件，通知worker进程启动成功
    _set_app_event(app, started_event)
    # 启动worker app
    uvicorn.run(app, host=worker_host, port=worker_port)


def run_fastapi_server(started_event):
    from server.fastapi import app, get_fastapi_address

    # 创建fastapi app
    @app.get("/test_fastapi")
    def fastapi_test():
        return {"code": 200, "msg": "fastapi test"}

    # 获得fastapi server的host和port
    host, port = get_fastapi_address()
    # 设置started_event事件，通知fastapi进程启动成功
    _set_app_event(app, started_event)
    # 启动fastapi app
    uvicorn.run(app, host=host, port=port)


def run_weibui_server(started_event):

    from server.webui import get_webui_address

    host, port = get_webui_address()
    cmd = [
        "streamlit",
        "run",
        "webui.py",
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--theme.base",
        "light",
        "--theme.primaryColor",
        "#165dff",
        "--theme.secondaryBackgroundColor",
        "#f5f5f5",
        "--theme.textColor",
        "#000000",
    ]
    p = subprocess.Popen(cmd)
    started_event.set()
    p.wait()


def dump_server_info():
    pass


async def start_main():
    # 解析命令行参数
    args = parse_args()
    # 创建多进程管理器
    manager = multiprocessing.Manager()
    # 创建worker之间通信的队列queue
    queue = manager.Queue()
    # 创建一个字典processes用来保存多进程
    processes = {}
    # 创建controller进程和contoller_started事件,事件用来通知controller进程启动成功
    controller_started = manager.Event()
    process = multiprocessing.Process(
        target=run_controller_server,
        name="controller",
        kwargs={"started_event": controller_started},
        daemon=True,
    )
    processes["controller"] = process
    # 创建worker进程
    worker_started = manager.Event()
    process = multiprocessing.Process(
        target=run_worker_server,
        name="worker",
        kwargs={"started_event": worker_started},
        daemon=True,
    )
    processes["worker"] = process
    # 创建webui进程
    webui_started = manager.Event()
    process = multiprocessing.Process(
        target=run_weibui_server,
        name="webui",
        kwargs={"started_event": webui_started},
        daemon=True,
    )
    processes["webui"] = process
    # 创建fastapi进程
    fast_started = manager.Event()
    process = multiprocessing.Process(
        target=run_fastapi_server,
        name="fastapi",
        kwargs={"started_event": fast_started},
        daemon=True,
    )
    processes["fastapi"] = process

    if len(processes) == 0:
        print("no process")
    else:
        try:
            # 运行processes中的进程，需要等待上个进程启动后再启动下个进程
            if p := processes.get("controller"):
                p.start()
                controller_started.wait()
            if p := processes.get("worker"):
                p.start()
                worker_started.wait()
            if p := processes.get("fastapi"):
                p.start()
                fast_started.wait()
            if p := processes.get("webui"):
                p.start()
                webui_started.wait()
            dump_server_info()
            # 根据queue,跟新worker的状态
            while True:
                cmd = queue.get()
                if isinstance(cmd, list):
                    # 根据queue中的命令，更新worker
                    pass
        except Exception as e:
            print(f"error:{e}")
        finally:
            for p in processes.values():
                if p and p.is_alive():
                    p.kill()
            for p in processes.values():
                print(f"Process status: {p}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_main())
    # started_event = multiprocessing.Manager().Event()
    # run_weibui_server(started_event)
