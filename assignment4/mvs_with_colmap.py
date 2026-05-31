import os
import subprocess
import argparse
import shutil

# Allow COLMAP (Qt-based) to run on headless servers without an X display.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def find_colmap_executable():
    colmap_path = os.environ.get('COLMAP_EXE') or shutil.which('colmap.exe') or shutil.which('colmap')
    if colmap_path and colmap_path.lower().endswith(('.bat', '.cmd')):
        exe_path = os.path.join(os.path.dirname(colmap_path), 'bin', 'colmap.exe')
        if os.path.exists(exe_path):
            return exe_path
    if colmap_path:
        return colmap_path

    default_path = os.path.expanduser(r'~\Desktop\colmap-x64-windows-nocuda\bin\colmap.exe')
    if os.path.exists(default_path):
        return default_path

    raise FileNotFoundError('Cannot find COLMAP. Please add colmap.exe to PATH or set COLMAP_EXE.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run COLMAP for multi-view stereo')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the input directory containing images in data_dir/images')
    args = parser.parse_args()
    data_dir = args.data_dir
    colmap_cmd = find_colmap_executable()

    # Feature extraction with shared intrinsics (assume it's the same camera)
    # 修改: COLMAP 4.x uses FeatureExtraction.use_gpu instead of SiftExtraction.use_gpu
    subprocess.run([colmap_cmd, 'feature_extractor', '--image_path', os.path.join(data_dir, 'images'), '--database_path', os.path.join(data_dir, 'database.db'), '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'PINHOLE', '--FeatureExtraction.use_gpu', '0'], check=True)

    # Feature matching
    # 修改: COLMAP 4.x uses FeatureMatching.use_gpu instead of SiftMatching.use_gpu
    subprocess.run([colmap_cmd, 'exhaustive_matcher', '--database_path', os.path.join(data_dir, 'database.db'), '--FeatureMatching.use_gpu', '0'], check=True)

    # Create sparse reconstruction folder
    os.makedirs(os.path.join(data_dir, 'sparse'), exist_ok=True)

    # Sparse reconstruction
    subprocess.run([colmap_cmd, 'mapper', '--image_path', os.path.join(data_dir, 'images'), '--database_path', os.path.join(data_dir, 'database.db'), '--output_path', os.path.join(data_dir, 'sparse')], check=True)

    # Convert binary model to text format
    os.makedirs(os.path.join(data_dir, 'sparse', '0_text'), exist_ok=True)
    subprocess.run([colmap_cmd, 'model_converter', '--input_path', os.path.join(data_dir, 'sparse', '0'), '--output_path', os.path.join(data_dir, 'sparse', '0_text'), '--output_type', 'TXT'], check=True)

    print("COLMAP multi-view stereo pipeline completed successfully!")
    print("Sparse 3D reconstruction saved in:", os.path.join(data_dir, 'sparse', '0_text'))
    
