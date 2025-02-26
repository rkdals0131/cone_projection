import yaml, os
from ament_index_python.packages import get_package_share_directory

def extract_configuration():
    config_file = os.path.join( # 컨픽 파일 경로 설정
        get_package_share_directory('cone_projection'),
        'config',
        'general_configuration.yaml'
    )

    with open(config_file, 'r') as file: # 읽기모드로 열기
        config = yaml.safe_load(file) # 파일 내용을 읽어서 딕셔너리로 변환

    return config # 컨픽 딕셔너리 반환