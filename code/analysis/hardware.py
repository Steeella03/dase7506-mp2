"""Record local hardware without including the user's hostname or account."""
import ctypes
import json
import os
from pathlib import Path
import platform
import struct


def main():
    result = {'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
              'os': platform.platform(), 'python': platform.python_version(),
              'training_device': 'CPU', 'gpu_used': False}
    if os.name == 'nt':
        buffer = ctypes.create_string_buffer(64)
        struct.pack_into('I', buffer, 0, 64)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(buffer)):
            result['physical_ram_gib'] = round(struct.unpack_from('Q', buffer, 8)[0] / 2**30, 2)
    output = Path('runs/hardware.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
