"""Setup configuration for RAG FAQ Chatbot."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag-faq-chatbot",
    version="1.0.0",
    author="Henry AI Task",
    description="RAG-based FAQ Support Chatbot for NVIDIA Financial Information",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-faq-chatbot",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-community>=0.0.10",
        "faiss-cpu>=1.7.4",
        "openai>=1.7.0",
        "tiktoken>=0.5.2",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.12.1",
            "flake8>=6.1.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "faq-build-index=src.build_index:main",
            "faq-query=src.query:main",
        ],
    },
)
