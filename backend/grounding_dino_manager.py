import atexit
import subprocess
import time
from urllib.request import urlopen
from urllib.error import URLError

from config import(
    GROUNDING_DINO_AUTO_START,
    GROUNDING_DINO_PYTHON_PATH,
    GROUNDING_DINO_WORKER_SCRIPT,
    GROUNDING_DINO_WORKER_HOST,
    GROUNDING_DINO_WORKER_PORT,
    GROUNDING_DINO_STARTUP_TIMEOUT,
)

_grounding_dino_process=None

def start_grounding_dino_worker():
    '''启动Grounding DINO worker'''
    global _grounding_dino_process

    if not GROUNDING_DINO_AUTO_START:
        print("Grounding DINO worker自动启动已关闭")
        return
       
    if _is_worker_alive():
        print("Grounding DINO worker已经在运行")
        return
    
    print("="*60)
    print("正在启动Grounding DINO worker...")
    print("python:",GROUNDING_DINO_PYTHON_PATH)
    print("worker:",GROUNDING_DINO_WORKER_SCRIPT)
    print("="*60)
    
    _grounding_dino_process=subprocess.Popen(
        [
            GROUNDING_DINO_PYTHON_PATH,
            str(GROUNDING_DINO_WORKER_SCRIPT)
        ]
    )

    atexit.register(stop_grounding_dino_worker)

    _wait_worker_ready()

    print("Grounding DINO worker启动完成")

def stop_grounding_dino_worker():
    '''关闭Grounding DINO worker'''
    global _grounding_dino_process

    if _grounding_dino_process is None:
        return
    
    if _grounding_dino_process.poll() is not None:
        return
    
    print("正在关闭Grounding DINO worker")

    _grounding_dino_process.terminate()

    try:
        _grounding_dino_process.wait(timeout=10)
        print("Grounding DINO worker已关闭")
    except subprocess.TimeoutExpired:
        print("Grounding DINO worker关闭超时,强制结束")
        _grounding_dino_process.kill()


def _is_worker_alive():
    '''检查worker是否已经可用'''
    health_url=f"http://{GROUNDING_DINO_WORKER_HOST}:{GROUNDING_DINO_WORKER_PORT}/health"

    try:
        with urlopen(health_url,timeout=2) as response:
            return response.status==200
    except Exception:
        return False
    
def _wait_worker_ready():
    '''等待worker加载模型并启动HTTP服务'''
    start=time.perf_counter()

    while True:
        if _is_worker_alive():
            return
        
        if _grounding_dino_process is not None:
            if _grounding_dino_process.poll() is not None:
                raise RuntimeError("Grounding DINO worker启动失败,进程已经退出")

        elapsed=time.perf_counter()-start

        if elapsed>GROUNDING_DINO_STARTUP_TIMEOUT:
            raise TimeoutError("等待Grounding DINO worker启动超时")
        
        time.sleep(1)

