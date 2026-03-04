FROM python:3.11-slim

COPY wheelhouse /wheels
RUN pip install --upgrade pip \
    && pip install --no-cache-dir /wheels/*cp311*.whl \
    && rm -rf /wheels

CMD ["python", "-c", "import xllim; print(xllim.__version__)"]