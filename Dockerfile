FROM tensorflow/tensorflow:1.15.4-gpu-py3
RUN pip install --upgrade pip && pip install Pillow scipy