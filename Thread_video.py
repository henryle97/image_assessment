import threading

class MyThread(threading.Thread):
    def __init__(self, model, img_list, type):
        threading.Thread.__init__(self, daemon=True)
        self.model = model
        self._result = None
        self.img_list = img_list
        self.type = type

    def run(self):
        result = self.model.assessment_video(self.img_list, self.type)
        self._result = result

    def join(self, *args):
        threading.Thread.join(self)
        return self._result, self.type
