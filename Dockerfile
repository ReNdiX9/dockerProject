# small Python base
FROM python:3.9-slim

# headless matplotlib + unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# minimal libs for matplotlib to render text/axes
RUN apt-get update && apt-get install -y --no-install-recommends \
      libfreetype6 libpng16-16 fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# copy your code and csv into the image (simple for this task)
COPY . /app

# python deps (seaborn allowed by your spec, but not required)
RUN pip install --no-cache-dir numpy pandas matplotlib seaborn

# default run: process CSV and put outputs into ./outputs
CMD ["python", "data_analysis.py", "All_Diets.csv", "--out", "outputs"]
