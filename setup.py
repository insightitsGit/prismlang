from setuptools import setup, find_packages

setup(
    name="prismlang",
    version="0.1.0",
    author="Amin Parva",
    author_email="prismrag@insightits.com",
    description="Deterministic vector language protocol for multi-agent AI orchestration",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "langgraph>=0.2.0",
        "onnxruntime>=1.17.0",
        "numpy>=1.26.0",
        "huggingface-hub>=0.20.0",
        "tokenizers>=0.15.0",
    ],
    extras_require={
        "postgres": ["psycopg2-binary>=2.9.0"],
    },
)
