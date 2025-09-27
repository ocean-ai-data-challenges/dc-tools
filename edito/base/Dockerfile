ARG BASE_IMAGE=inseefrlab/onyxia-jupyter-python:py3.13.7-gpu
FROM $BASE_IMAGE
LABEL maintainer="emmanuel.braux@imt-atlantique.fr"
LABEL org.opencontainers.image.source="https://github.com/ppr-ocean-ia/dc-tools"
LABEL org.opencontainers.image.description="DC-tools image for GPU workloads on Edito Datalab"

# For security reasons, we do not want to run the container as root
USER ${USERNAME}
WORKDIR ${WORKSPACE_DIR}

# =======================
# Manage Jupyterlab launch
# =======================
# 
# CMD on onyxia-jupyter-python is buggy with WORKSPACE_DIR
#  - ${WORKSPACE_DIR} is not expanded if used directly in CMD
# On Edito datalab Edito, the launch command is overridden anyway in the k8s deployment.yaml file
#     command: ["/bin/sh","-c"]
#     args: ["{{ .Values.init.standardInitPath }} jupyter lab --no-browser --ip '0.0.0.0' --LabApp.token='$(PASSWORD)' --ContentsManager.allow_hidden=True"]
#   "standardInitPath" is defined in values.yaml file : `standardInitPath: /opt/onyxia-init.sh`
# So the command is at last ["/bin/sh","-c", jupyter lab --no-browser --ip '0.0.0.0' --LabApp.token='$(PASSWORD)' --ContentsManager.allow_hidden=True"]
# We use a bash -c command to expand the variable  
# CMD ["jupyter", "lab", "--no-browser", "--ip", "0.0.0.0", "--notebook-dir", "${WORKSPACE_DIR}"]
CMD ["/bin/sh", "-c", "jupyter lab --no-browser --ip 0.0.0.0 --LabApp.token=password --ContentsManager.allow_hidden=True"]
# --NotebookApp.websocket_ping_interval=90000 --NotebookApp.websocket_ping_timeout=90000

# Le lancement dans edito est défini 