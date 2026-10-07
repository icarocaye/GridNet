import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,rtp,udp"

import cv2
import time
import threading
from pyparrot.Bebop import Bebop

# Use o mesmo .sdp que funcionou no seu teste
SDP = "/media/icaro/Extra/GitHub/GridNet/code/flight/bebop_local.sdp"

os.makedirs("frames", exist_ok=True)


class FrameRecorder:
    """Lê o stream do drone em uma thread e salva ~1 frame por segundo."""

    def __init__(self, sdp_path, interval=1.0):
        self.sdp_path = sdp_path
        self.interval = interval
        self.index = 0
        self.got_first_frame = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        cap = cv2.VideoCapture(self.sdp_path, cv2.CAP_FFMPEG)
        print("Câmera aberta:", cap.isOpened())
        last_save = 0

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            self.got_first_frame.set()

            now = time.time()
            if now - last_save >= self.interval:
                filename = "frames/frame_%06d.jpg" % self.index
                cv2.imwrite(filename, frame)
                print("Imagem salva:", filename)
                self.index += 1
                last_save = now

        cap.release()


bebop = Bebop()
print("Conectando...")
success = bebop.connect(10)
print("Conexão:", success)

if success:
    recorder = None
    in_air = False

    try:
        bebop.smart_sleep(3)
        bebop.ask_for_state_update()

        # ----- Limites de segurança (ANTES de decolar) -----
        bebop.set_max_vertical_speed(2)   # m/s
        bebop.set_max_altitude(30)        # m
        bebop.set_max_tilt(15)            # graus, limita a velocidade horizontal

        # ----- Câmera -----
        bebop.set_picture_format('jpeg')
        bebop.start_video_stream()
        bebop.smart_sleep(3)

        recorder = FrameRecorder(SDP, interval=1.0)
        recorder.start()

        # Só voa se o vídeo estiver funcionando
        if not recorder.got_first_frame.wait(timeout=15):
            raise RuntimeError("Nenhum frame recebido em 15 s. Abortando antes de decolar.")
        print("Vídeo OK, decolando...")

        # ----- Voo -----
        bebop.safe_takeoff(10)
        in_air = True

        # Aponta a câmera para baixo
        bebop.pan_tilt_camera_velocity(tilt_velocity=-30, pan_velocity=0, duration=3)

        # roll = lateral, pitch = frente/trás, yaw = giro, vertical_movement = subir/descer
        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=100, duration=15)
        bebop.fly_direct(roll=0, pitch=100, yaw=0, vertical_movement=0, duration=5)
        bebop.smart_sleep(2)

        bebop.ask_for_state_update()
        bebop.smart_sleep(0.5)
        vx = bebop.sensors.sensors_dict.get("SpeedChanged_speedX", 0)
        vy = bebop.sensors.sensors_dict.get("SpeedChanged_speedY", 0)
        print(f"Vx = {vx} m/s")
        print(f"Vy = {vy} m/s")

        bebop.fly_direct(roll=0, pitch=-100, yaw=0, vertical_movement=0, duration=5)
        bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-100, duration=10)

    except (Exception, KeyboardInterrupt) as e:
        print("Erro/interrupção durante o voo:", repr(e))

    finally:
        # Sempre tenta pousar se decolou, mesmo após erro ou Ctrl+C
        if in_air:
            print("Pousando...")
            bebop.safe_land(10)

        if recorder is not None:
            recorder.stop()

        try:
            bebop.stop_video_stream()
        except Exception:
            pass

        print("Desconectando...")
        bebop.disconnect()
else:
    print("Erro ao conectar ao Bebop.")
