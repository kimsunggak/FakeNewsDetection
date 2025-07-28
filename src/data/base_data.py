class BaseData:
    """
    외부 데이터 수집을 일관되게 하기위한 상위 클래스 선언
    외부 데이터 출처를 하나로만 구현하였기 때문에 비워둠
    """
    def __init__(self, data):
        self.data = data