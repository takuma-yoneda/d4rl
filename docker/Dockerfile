FROM takumaynd/robosuite:latest

# For the rendering script
RUN pip install -U pip
RUN pip install matplotlib einops

ADD . .
RUN pip install .

# It's gonna build in the first call
RUN python -c 'import mujoco_py'
RUN pip install -I gym==0.23.1
RUN pip install scikit-image
