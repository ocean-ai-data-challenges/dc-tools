#FROM inseefrlab/onyxia-jupyter-python:py3.13.7-gpu
FROM inseefrlab/onyxia-jupyter-pytorch:py3.13.7-gpu

USER root

# Créer un rapport de taille
RUN echo "=== ANALYSE DE L'IMAGE DE BASE ===" > /tmp/size_report.txt && \
    echo "" >> /tmp/size_report.txt && \
    echo "Python location:" >> /tmp/size_report.txt && \
    which python3 >> /tmp/size_report.txt && \
    python3 --version >> /tmp/size_report.txt && \
    echo "" >> /tmp/size_report.txt && \
    echo "Python sys.path:" >> /tmp/size_report.txt && \
    python3 -c "import sys; print('\n'.join(sys.path))" >> /tmp/size_report.txt && \
    echo "" >> /tmp/size_report.txt && \
    echo "Searching for site-packages:" >> /tmp/size_report.txt && \
    find / -type d -name "site-packages" 2>/dev/null >> /tmp/size_report.txt && \
    echo "" >> /tmp/size_report.txt && \
    echo "Site-packages sizes:" >> /tmp/size_report.txt && \
    for dir in $(find / -type d -name "site-packages" 2>/dev/null); do \
        echo "Size of $dir:" >> /tmp/size_report.txt && \
        du -sh "$dir" >> /tmp/size_report.txt; \
    done && \
    echo "" >> /tmp/size_report.txt && \
    echo "PyTorch location and size:" >> /tmp/size_report.txt && \
    python3 -c "import torch, os; print('Torch path:', os.path.dirname(torch.__file__))" >> /tmp/size_report.txt 2>&1 && \
    du -sh $(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null) 2>/dev/null >> /tmp/size_report.txt || echo "Torch size calculation failed" >> /tmp/size_report.txt && \
    echo "" >> /tmp/size_report.txt && \
    echo "Top 20 largest directories in Python paths:" >> /tmp/size_report.txt && \
    for dir in $(python3 -c "import sys; print(' '.join([p for p in sys.path if p]))" 2>/dev/null); do \
        if [ -d "$dir" ]; then \
            du -sh "$dir"/* 2>/dev/null; \
        fi; \
    done | sort -rh | head -20 >> /tmp/size_report.txt && \
    echo "" >> /tmp/size_report.txt && \
    echo "Installed pip packages:" >> /tmp/size_report.txt && \
    pip list >> /tmp/size_report.txt && \
    cat /tmp/size_report.txt

CMD ["cat", "/tmp/size_report.txt"]