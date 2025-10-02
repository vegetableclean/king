class Watchdog(object):
    def __init__(self, timeout=1.0):
        self._timeout = timeout + 1.0  # buffer to avoid overlap with other CARLA timeouts
        self._failed = False
        self._timer = None

    def start(self):
        self.stop()  # ensure no previous timer is dangling
        self._timer = Timer(self._timeout, self._event)
        self._timer.daemon = True
        self._timer.start()

    def update(self):
        self.start()

    def _event(self):
        print('Watchdog exception - Timeout of {} seconds occurred'.format(self._timeout))
        self._failed = True
        self.stop()
        thread.interrupt_main()

    def stop(self):
        t = getattr(self, "_timer", None)
        if t is not None:
            try:
                t.cancel()
            finally:
                self._timer = None

    def get_status(self):
        return not self._failed

