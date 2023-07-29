# gta5-selfdriving
CNN을 사용해 GTA5에서 자율주행자동차를 만드는 프로젝트입니다.
추후, 코드를 다듬고 자세한 설명을 올리겠습니다.

* collect_data : screenshot.py와 get_joystick.py를 사용해, 데이터를 저장함.
* get_joystick : 조이스틱의 값을 가져옴.
* preprocess : 데이터 파일들을 가져와 한 파일로 합치고 image augmentation를 수행함.
* run_model : 모델을 실행하여, 자동차를 조종함.
* screenshot : 화면 사진을 찍음.
* show_data : 데이터(steering, throttle)를 그래프로 보여주고 저장한 데이터와 사진을 보여줌.
* train_model : NVIDIA의 'End-to-End Deep Learning for Self-Driving Cars'를 바탕으로 화면 전체에 대한 CNN을 구현함. 이를 미니맵에 대한 CNN과 합침.
* vjoy : vjoy, x360ce를 통해 GTA5의 입력을 보냄.
