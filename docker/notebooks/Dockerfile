FROM jupyter/scipy-notebook:hub-2.1.1

# Install Tensorflow
RUN mamba install --yes \
    'tensorflow=2.7.0' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# shell exit on command fail
# RUN set -ex \
#     'tensorflow==2.7.0' \
#     'mlflow==1.14.1' \
#     'dask==2021.7.0'
