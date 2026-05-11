uv sync --extra dev;
cd mamba_install;
python -m pip install . --no-build-isolation;
python -m pip install numpy==1.26.4;
cd ..;