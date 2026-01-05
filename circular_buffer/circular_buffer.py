import torch
import numpy as np

class CircularBuffer:
    def __init__(
            self,
            max_length: int,
            fixed_shape: tuple = None,
            backend: str = 'torch',  # 'torch' or 'numpy'
            device: str = 'cpu',     # torch only
            dtype = None             # torch.float32 or np.float32
    ):
        self.max_length = max_length
        self.backend = backend.lower()
        self.device = device
        
        # Default Dtypes
        if dtype is None:
            self.dtype = torch.float32 if self.backend == 'torch' else np.float32
        else:
            self.dtype = dtype
        
        self.buffer = None
        self.ptr = 0
        self.current_size = 0
        
        if fixed_shape is not None:
            self._init_buffer(fixed_shape)

    def __call__(self):
        return self.get_buffer()
    
    def __len__(self):
        return self.current_size

    def _init_buffer(self, frame_shape):
        full_shape = (self.max_length, ) + tuple(frame_shape)
        
        if self.backend == 'torch':
            self.buffer = torch.zeros(full_shape, device=self.device, dtype=self.dtype)
        else:
            # NumPy: device 파라미터 무시
            self.buffer = np.zeros(full_shape, dtype=self.dtype)

    def append(self, frame):
        # 1. 초기화
        if self.buffer is None:
            self._init_buffer(frame.shape)
            
        # 2. 타입 및 디바이스 정합성 체크 (Torch only)
        if self.backend == 'torch':
            if isinstance(frame, np.ndarray):
                frame = torch.from_numpy(frame)
            if frame.device != self.buffer.device or frame.dtype != self.buffer.dtype:
                frame = frame.to(device=self.buffer.device, dtype=self.buffer.dtype)
        else:
            # NumPy: 텐서가 들어오면 numpy로 변환
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
            if frame.dtype != self.buffer.dtype:
                frame = frame.astype(self.buffer.dtype)
            
        # 3. 버퍼 삽입
        self.buffer[self.ptr] = frame
        
        # 4. 포인터 이동
        self.ptr = (self.ptr + 1) % self.max_length
        self.current_size = min(self.current_size + 1, self.max_length)

    def get_buffer(self):
        if self.buffer is None:
            if self.backend == 'torch':
                return torch.empty(0, device=self.device)
            else:
                return np.empty((0,))

        # 버퍼가 꽉 차지 않았을 때
        if not self.is_full():
            return self.buffer[:self.current_size]
        
        # 버퍼가 꽉 찼을 때 (롤링)
        if self.backend == 'torch':
            return torch.roll(self.buffer, shifts=-self.ptr, dims=0)
        else:
            return np.roll(self.buffer, shift=-self.ptr, axis=0)

    def reset(self):
        self.ptr = 0
        self.current_size = 0
        if self.buffer is not None:
            if self.backend == 'torch':
                self.buffer.zero_()
            else:
                self.buffer.fill(0)

    def is_full(self):
        return self.current_size >= self.max_length