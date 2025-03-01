import time
import base64
import datetime
from collections import deque
from threading import Lock, Thread
from cv2 import VideoCapture, imencode


class WebcamStreamV1:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class WebcamStream:
    def __init__(self, buffer_size=20):
        self.stream = VideoCapture(0)
        self.running = False
        self.lock = Lock()
        self.frame_buffer = deque(maxlen=buffer_size)  # 帧缓冲区
        self._init_buffer(buffer_size)

    def _init_buffer(self, buffer_size):
        """初始化时填充缓冲区"""
        for _ in range(buffer_size):
            ret, frame = self.stream.read()
            if ret:
                self.frame_buffer.append({
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "frame": frame
                })

    def start(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._update)
            self.thread.start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.stream.read()
            if ret:
                with self.lock:
                    self.frame_buffer.append({
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        "frame": frame
                    })
            time.sleep(1)

    def capture_frames(self, num_frames):
        """即时获取最新帧的三种方式"""
        with self.lock:
            # # 方法1：直接获取缓冲区最后N帧
            # recent_frames = list(self.frame_buffer)[-num_frames:]

            # 方法2：若需要保证帧不重复（适合高速摄像头）
            recent_frames = []
            seen = set()
            for f in reversed(self.frame_buffer):
                if f["time"] not in seen:
                    seen.add(f["time"])
                    recent_frames.append(f)
                    if len(recent_frames) == num_frames:
                        break
            recent_frames.reverse()

        # 编码处理
        result = []
        for f in recent_frames[:num_frames]:
            _, buffer = imencode(".jpeg", f["frame"])
            result.append({
                "frame_time": f["time"],
                "frame": base64.b64encode(buffer)
            })
        return result

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def release(self):
        self.stop()
        self.stream.release()
