FROM python:3.7
COPY setup.py .
RUN pip install -e .
RUN pip install streamlit
WORKDIR /app
