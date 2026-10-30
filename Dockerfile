FROM python:3.13-slim

# Hugging Face Spaces runs containers as a non-root user with id 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]