import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="audio-data-utils",
    version="0.0.3",
    author='Sarthak Yadav',
    description="Audio data loading utils for personal research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SarthakYadav/audio_data_utils",
    # package_dir={"": ""},
    packages=[
        "audio_utils",
        "audio_utils.common"
    ],
    python_requires=">=3.8"
)
