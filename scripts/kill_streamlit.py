import os
import signal
import psutil


def kill_streamlit():
    port = 8501
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    print(f"Killing process {proc.info['pid']} on port {port}...")
                    proc.terminate()
        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
            psutil.AccessDenied,
        ):
            pass


if __name__ == "__main__":
    kill_streamlit()
