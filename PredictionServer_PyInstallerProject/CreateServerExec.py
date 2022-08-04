"""
CreateServerExec.py
---------------

Before you run
---------------
    - Conda environment must be setup and activated using CondaEnvSetup.bat
    - Put the name of the python file you want to install as the first arg in PyInstaller.__main__.run
"""

import os
import PyInstaller.__main__
import shutil

env_path = os.environ['CONDA_PREFIX']
env_path = env_path.replace(chr(92), '/')

tf_dir = f'{env_path}/Lib/site-packages/tensorflow'
tf_temp_dir = f'{env_path}/Lib/site-packages/tensorflow_temp'
sklearn_dir = f'{env_path}/Lib/site-packages/sklearn'
sklearn_temp_dir = f'{env_path}/Lib/site-packages/sklearn_temp'

# copy tensorflow and sklearn to temp directories
if not os.path.exists(tf_temp_dir):
    shutil.copytree(tf_dir, tf_temp_dir)
if not os.path.exists(sklearn_temp_dir):
    shutil.copytree(sklearn_dir, sklearn_temp_dir)

# files that are already included within target python package
# if these are not removed from additional files, they will be added twice causing warnings
to_remove = [f'{tf_temp_dir}\python\_pywrap_mlir.pyd',
f'{tf_temp_dir}/python\_pywrap_parallel_device.pyd',
f'{tf_temp_dir}/python\_pywrap_py_exception_registry.pyd',
f'{tf_temp_dir}/python\_pywrap_quantize_training.pyd',
f'{tf_temp_dir}/python\_pywrap_sanitizers.pyd',
f'{tf_temp_dir}/python\_pywrap_tensorflow_internal.pyd',
f'{tf_temp_dir}/python\_pywrap_tfe.pyd',
f'{tf_temp_dir}/python\_pywrap_toco_api.pyd',
f'{tf_temp_dir}/python\client\_pywrap_debug_events_writer.pyd',
f'{tf_temp_dir}/python\client\_pywrap_device_lib.pyd',
f'{tf_temp_dir}/python\client\_pywrap_events_writer.pyd',
f'{tf_temp_dir}/python\client\_pywrap_tf_session.pyd',
f'{tf_temp_dir}/python\data\experimental\service\_pywrap_server_lib.pyd',
f'{tf_temp_dir}/python\data\experimental\service\_pywrap_utils.pyd',
f'{tf_temp_dir}/python/framework\_dtypes.pyd',
f'{tf_temp_dir}/python/framework\_op_def_registry.pyd',
f'{tf_temp_dir}/python/framework\_proto_comparators.pyd',
f'{tf_temp_dir}/python/framework\_pywrap_python_op_gen.pyd',
f'{tf_temp_dir}/python\grappler\_pywrap_tf_cluster.pyd',
f'{tf_temp_dir}/python\grappler\_pywrap_tf_optimizer.pyd',
f'{tf_temp_dir}/python\lib\core\_pywrap_bfloat16.pyd',
f'{tf_temp_dir}/python\lib\core\_pywrap_py_func.pyd',
f'{tf_temp_dir}/python\lib\io\_pywrap_file_io.pyd',
f'{tf_temp_dir}/python\lib\io\_pywrap_record_io.pyd',
f'{tf_temp_dir}/python\platform\_pywrap_stacktrace_handler.pyd',
f'{tf_temp_dir}/python\platform\_pywrap_tf2.pyd',
f'{tf_temp_dir}/python\profiler\internal\_pywrap_profiler.pyd',
f'{tf_temp_dir}/python\profiler\internal\_pywrap_traceme.pyd',
f'{tf_temp_dir}/python\saved_model\experimental\pywrap_libexport.pyd',
f'{tf_temp_dir}/python/util/_pywrap_checkpoint_reader.pyd',
f'{tf_temp_dir}/python/util\_pywrap_nest.pyd',
f'{tf_temp_dir}/python/util\_pywrap_tensor_float_32_execution.pyd',
f'{tf_temp_dir}/python/util\_pywrap_tfprof.pyd',
f'{tf_temp_dir}/python/util\_pywrap_util_port.pyd',
f'{tf_temp_dir}/python/util\_pywrap_utils.pyd',
f'{tf_temp_dir}/python/util\_tf_stack.pyd',
f'{tf_temp_dir}/python/util/fast_module_type.pyd',
f'{tf_temp_dir}/lite\python\interpreter_wrapper\_pywrap_tensorflow_interpreter_wrapper.pyd',
f'{tf_temp_dir}/lite\python\metrics_wrapper\_pywrap_tensorflow_lite_metrics_wrapper.pyd',
f'{sklearn_temp_dir}/__check_build\_check_build.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/_isotonic.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/cluster\_dbscan_inner.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/cluster\_hierarchical_fast.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/cluster\_k_means_common.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/cluster\_k_means_elkan.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/cluster\_k_means_lloyd.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/cluster\_k_means_minibatch.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/decomposition\_cdnmf_fast.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/decomposition\_online_lda_fast.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/linear_model\_cd_fast.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/linear_model\_sag_fast.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/linear_model\_sgd_fast.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/manifold\_barnes_hut_tsne.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/manifold\_utils.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/metrics\_dist_metrics.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/metrics\_pairwise_fast.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/metrics\cluster\_expected_mutual_info_fast.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/neighbors\_ball_tree.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/neighbors\_kd_tree.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/preprocessing\_csr_polynomial_expansion.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/svm\_liblinear.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/svm\_libsvm.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/svm\_libsvm_sparse.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/utils\_fast_dict.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/utils\_logistic_sigmoid.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/utils\_openmp_helpers.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/utils\_random.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/utils\_readonly_array_wrapper.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/utils\_seq_dataset.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/utils/arrayfuncs.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/utils\murmurhash.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/utils\sparsefuncs_fast.cp39-win_amd64.pyd',
f'{sklearn_temp_dir}/utils/arrayfuncs.cp39-win_amd64.pyd']

# remove files from temp dirs
for file in to_remove:
    file = file.replace(chr(92), '/')
    if os.path.exists(file):
        os.remove(file)

# pyinstaller data additions
src_tensorflow = f'{tf_temp_dir};tensorflow/'
src_sklearn = f'{sklearn_temp_dir};sklearn/'
src_joblib = f'{env_path}/Lib/site-packages/joblib;joblib/'
src_xgboost = f'{env_path}/Lib/site-packages/xgboost;xgboost/'
src_xgboost_binary = f'{env_path}/Library/mingw-w64/bin/xgboost.dll;Library/mingw-w64/bin/'

# calls pyinstaller to build executable in dist folder
PyInstaller.__main__.run(['PrioritizationPredictionServer.py', '--noconfirm',
                          '--onefile', '--console',
                          '--add-data', src_tensorflow,
                          '--add-data', src_joblib,
                          '--add-data', src_xgboost,
                          '--add-data', src_xgboost_binary,
                          '--add-data', src_sklearn])
