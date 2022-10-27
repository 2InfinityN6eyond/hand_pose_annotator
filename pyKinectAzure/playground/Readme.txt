IR image :  capture.get_ir_image()
    
    NFOV_UNBINNED
        576x640

    NFOV_BINNED
        288x320

    WFOV_2X2BINNED    
        512 x 512
        uint16

    PASSIVE_IR
        1024 * 1024

    이미지를 256으로 나누면 opencv에서 시각화하기 좋다.
    depth_mode 가 passive면,
    주변의 IR이 너무 부족해서 형광등처럼 밝은 물체만 값이 아주 높게 나오고,

    active_mode면 빔 프로젝터 스크린에 IR이 죄다 반사돼 구분이 어려운 문제가 있다.

    신기하게 이미지에 10을 곱하면 ACTIVE_MODE일떄 opencv에서 손이 아주 잘 구별된다.
    뭔가 안보이는 패턴이 있는 것 같음


depth image :  capture.get_depth_image()

    IR 이미지와 사이즈는 똑같고 dtype는 uint16

    WFOV_2X2BINNED
        512 x 512
        uint8

    
    PASSIVE_IR
        안나옴