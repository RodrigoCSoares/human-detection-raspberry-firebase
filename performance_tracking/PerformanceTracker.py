import psutil
from datetime import datetime

def getCurrentPerformanceStatus():
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent()
    cpu_frq = psutil.cpu_freq()    
    return { 
        "data": {
            "memory_available": memory.total,
            "memory_used": memory.used,
            "memory_percent_used": memory.percent,
            "cpu_current_frequency":  cpu_frq.current,
            "cpu_max_frequency": cpu_frq.max,
            "cpu_min_frequency": cpu_frq.min,
            "cpu_percent_used": cpu
        },
        "datetime": datetime.today().strftime('%Y-%m-%dT%H:%M')
    }