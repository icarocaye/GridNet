from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
import cv2
import time
import os

# classe responsável por salvar as imagens da câmera
class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision
        self.last_save = time.time()

    def save_pictures(self, args):
        current_time = time.time()

        # salva uma imagem a cada 1 segundo
        if current_time - self.last_save >= 1:
            img = self.vision.get_latest_valid_picture()

            if img is not None:
                filename = "frames/frame_%06d.jpg" % self.index
                cv2.imwrite(filename, img)
                print("Imagem salva:", filename)

                self.index += 1
                self.last_save = current_time


# criando a pasta onde as imagens serão salvas
os.makedirs("frames", exist_ok=True)

# conectando o código com o drone
bebop = Bebop()
print("Conectando...")
success = bebop.connect(10)
print(success)

if success:
    #-------------CONFIGURAÇÕES DO DRONE--------------------
    bebop.smart_sleep(3) # aguardando estabilizar
    bebop.ask_for_state_update() # retorna os dados do drone

    bebop.safe_takeoff(10) # ele chega a uma altura de 1~1.5 m
    
    bebop.set_max_vertical_speed(2) # velocidade vertical maxima de 2 m/s
    bebop.set_max_altitude(30) # altura máxima de 30 m
    bebop.set_max_tilt(15) # é o jeito de limitar a velocidade
    
    #--------------CONFIGURAÇÕES DA CÂMERA-------------------
    bebop.set_picture_format('jpeg')
    bebop.pan_tilt_camera_velocity(tilt_velocity=-30, pan_velocity=0, duration=3) # virando a camêra pra baixo
    
    # CONFIGURAR A CAMERA PRA GRAVAR DIREITINHO, NAO CONSEGUI LIDAR COM O FFMPEG

    # iniciando a captura de vídeo
    bebopVision = DroneVision(bebop, is_bebop=True)

    userVision = UserVision(bebopVision)

    bebopVision.set_user_callback_function(
        userVision.save_pictures,
        user_callback_args=None
    )

    successVision = bebopVision.open_video()

    if successVision:
        print("Vision successfully started!")
    else:
        print("Erro ao iniciar a câmera.")

    
    #roll = movimento lateral esquerda/direita
    #pitch = movimento pra tras/pra frente
    #yaw = giro em torno do próprio eixo
    #vertical_movement = pra baixo/pra cima
    #duration = tempo pelo qual o movimento será realizado
    bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=100, duration=15) # sobe até a altura máxima
    bebop.fly_direct(roll=0, pitch=100, yaw=0, vertical_movement=0, duration=5) # andando reto
    
    bebop.smart_sleep(2)
    
    # acessamos a velocidade que o drone atinge com esse tilt máximo
    bebop.ask_for_state_update()
    bebop.smart_sleep(0.5)
    vx = bebop.sensors.sensors_dict.get("SpeedChanged_speedX", 0)
    vy = bebop.sensors.sensors_dict.get("SpeedChanged_speedY", 0)
    print(f"Vx = {vx} m/s")
    print(f"Vy = {vy} m/s")    
    
    bebop.fly_direct(roll=0, pitch=-100, yaw=0, vertical_movement=0, duration=5) # retorno pra home
    bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-100, duration=10) # volta pra uma altura melhor pra fazer o pouso
    
    # encerrando a captura de vídeo
    bebopVision.close_video()

    bebop.safe_land(10) # NÃO FAZ O RETURN TO HOME, FAZER MANUALMENTE

    # encerrando a conexão
    bebop.disconnect()
