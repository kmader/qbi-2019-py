export PATH="$HOME/miniconda3/bin:$PATH"
source activate qbi2019
export QBI_HOME=$(pwd)

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

# Generating lectures
for cpy in Lectures/*py; do
    python $cpy |& tee nb_to_md.txt
    traceback=$(grep "Traceback (most recent call last):" nb_to_md.txt)
    if [[ $traceback ]]; then
        exit 1
    fi
done
# Generating exercises
for cpy in Exercises/*py; do
    python $cpy  |& tee nb_to_md.txt
    traceback=$(grep "Traceback (most recent call last):" nb_to_md.txt)
    if [[ $traceback ]]; then
        exit 1
    fi
done

cd ~
mkdir -p ${HOME}/doc
mkdir -p ${HOME}/doc/Lectures
mkdir -p ${HOME}/doc/Exercises
cp -r ${QBI_HOME}/Lectures/* ${HOME}/doc/Lectures
cp -r ${QBI_HOME}/Exercises/* ${HOME}/doc/Exercises
cp -r ./doc ${CIRCLE_ARTIFACTS}
