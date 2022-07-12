# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['ClassificationPredictionServer.py'],
             pathex=[],
             binaries=[],
             datas=[('C:/Users/jmoore2/Anaconda3/envs/tf_gpu/Lib/site-packages/tensorflow_temp', 'tensorflow/'), ('C:/Users/jmoore2/Anaconda3/envs/tf_gpu/Lib/site-packages/joblib', 'joblib/'), ('C:/Users/jmoore2/Anaconda3/envs/tf_gpu/Lib/site-packages/xgboost', 'xgboost/'), ('C:/Users/jmoore2/Anaconda3/envs/tf_gpu/Library/mingw-w64/bin/xgboost.dll', 'Library/mingw-w64/bin/'), ('C:/Users/jmoore2/Anaconda3/envs/tf_gpu/Lib/site-packages/sklearn_temp', 'sklearn/')],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=['runtime_hook.py'],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='ClassificationPredictionServer',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
