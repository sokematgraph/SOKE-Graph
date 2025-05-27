from setuptools import setup, find_packages

setup(
    name="sokegraph",
    version="1.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "sokegraph=sokegraph.__main__:main"
        ]
    },
    author="",
    description="A CLI tool called sokegraph",
    install_requires=[],
    python_requires=">=3.6"
)
