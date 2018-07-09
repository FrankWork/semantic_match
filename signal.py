import time
import sys
import signal

def clean_up():
    print('finally clean')
    sys.stdout.flush()

def SIGTERM_handler(sig, frame):
    print('SIGTERM handler')
    # clean_up()
    exit(0)

signal.signal(signal.SIGTERM, SIGTERM_handler) # Linux kill command: kill <PID>
# SIGKILL cannot be catched, Linux kill command: kill -9 <PID>

try:
    while True:
        time.sleep(1)
        print('wake.')
        sys.stdout.flush()
except SystemExit as e:
    print(f'catch {type(e)}')
    sys.stdout.flush()
except BaseException as e:
    print(f'catch {type(e)}')
    sys.stdout.flush()
finally:
    clean_up()
