from flask import Flask, render_template, Response
import cv2
import threading

app = Flask(__name__)

# 웹캠 캡처 객체 생성
cap = cv2.VideoCapture(0)

# 스레드 종료를 위한 플래그
thread_exit_flag = False

def generate_frames():
    global thread_exit_flag

    while not thread_exit_flag:
        # 웹캠에서 프레임 읽기
        success, frame = cap.read()
        if not success:
            break
        else:
            # 프레임을 바이트 스트림으로 인코딩하여 전송
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_stream():
    app.run(debug=False)

if __name__ == '__main__':
    # 웹 서버를 쓰레드로 실행
    server_thread = threading.Thread(target=start_stream)
    server_thread.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        # Ctrl+C 입력 시 스레드 종료
        thread_exit_flag = True
        server_thread.join()

    # 웹캠 캡처 객체 해제
    cap.release()
    cv2.destroyAllWindows()
