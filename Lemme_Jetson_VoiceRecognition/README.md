반드시 직접 발급한 Open AI API Key를 넣을 것  

Yeelight(불 켜고 끄기 관련해서는) 관련 제품 구매 후, 
yeelight 앱 설치, yeelight 초기화(껐다 켰다 5번). 그 후 Yeelight 2대 연결 (참고: https://m.blog.naver.com/mi-inter/221536569057)
yeelight는 반드시 와이파이 2.4G에 연결
오렌지파이 또한 Yeelight가 연결된 와이파이와 동일한 와이파이에 연결
연결 후 yeelight 앱에서 메뉴>LAN Control에서 모두 활성화 시켜주어야 Yeelight와 파이썬 연결 가능
IP address는 yeelight 앱에서 전구>설정>디바이스 정보>IP Address에서 확인가능. (앱 버전별로 조금씩 다를 수 있음.)
확인된 2개의 IP address를 파이썬 코드에서 수정해주어야 작동 가능
