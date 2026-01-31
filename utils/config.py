import os.path as osp
from addict import Dict
import tempfile
import re
import sys
import ast
from importlib import import_module

'''
Thanks the code from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py wrote by Open-MMLab.
The `Config` class here uses some parts of this reference.
'''

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

class Config:
    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')

        if filename is not None:
            cfg_dict = self._file2dict(filename, True)
            filename = filename

        super(Config, self).__setattr__('_cfg_dict', cfg_dict)
        super(Config, self).__setattr__('_filename', filename)
    
    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, 'r') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    # temp_config_name:临时文件本身
    def _substitute_predefined_vars(filename, temp_config_name):
        file_dirname = osp.dirname(filename)  # ./configs
        file_basename = osp.basename(filename)  # standardCE.py
        file_basename_no_extension = osp.splitext(file_basename)[0]  # standardCE
        file_extname = osp.splitext(filename)[1]    # .py
        support_templates = dict(
            fileDirname=file_dirname,   # ./configs
            fileBasename=file_basename, # standardCE.py
            fileBasenameNoExtension=file_basename_no_extension, # standardCE
            fileExtname=file_extname)   # .py
        with open(filename, 'r') as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w') as tmp_config_file:
            tmp_config_file.write(config_file)



    @staticmethod
    # filename: ./configs/standardCE.py ...
    def _file2dict(filename, use_predefined_variables=True):
        # filename: C:\Paper Reproduction\Co-learning\Co-learning-master\configs\standardCE.py
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]   # fileExtname: .py
        if fileExtname not in ['.py']:
            raise IOError('Only py type are supported now!')


        # abs_dir(temp_config_dir):C:\Paper Reproduction\Co-learning\Co-learning-master\temp\tmpbwl88c80
        # 改

        temp_config_name = 'tmp.py'
        temp_config_file = 'tmp\\tmp.py'
        temp_config_dir = 'tmp'
        # with tempfile.TemporaryDirectory(dir='./tmp') as temp_config_dir:
        #     temp_config_file = tempfile.NamedTemporaryFile(
        #         dir='./tmp', suffix=fileExtname)
        #     # temp_config_file.name: C:\Paper Reproduction\Co-learning\Co-learning-master\temp\tmpbwl88c80\tmpg725r61s.py
        #     # tmpg725r61s.py
        #     temp_config_name = osp.basename(temp_config_file.name)

            # Substitute predefined variables
        if use_predefined_variables:
            # 将filename里面的内容写到临时文件里面
            Config._substitute_predefined_vars(filename,
                                               temp_config_file)
        else:
            shutil.copyfile(filename, temp_config_file.name)

        if filename.endswith('.py'):
            temp_module_name = osp.splitext(temp_config_name)[0]
            sys.path.insert(0, temp_config_dir)
            Config._validate_py_syntax(filename)
            mod = import_module(temp_module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith('__')
            }
            # delete imported module
            del sys.modules[temp_module_name]
        # close temp file
        # temp_config_file.close()
        return cfg_dict

    @staticmethod
    def fromfile(filename, use_predefined_variables=True):
        cfg_dict = Config._file2dict(filename, use_predefined_variables)
        return Config(cfg_dict, filename=filename)
