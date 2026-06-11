import torch.nn as nn   # nn = Neural Network
import torch.nn.functional as F
import torch

# Inicializar os pesos das redes antes do treinamento
def weights_init_normal(m):
    classname = m.__class__.__name__
    # Para Convoluções
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)     # Pesos começam com valores pequenos aleatórios
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)       # Todos os bias começam em zero
    # Normalização Batch (impede que os valores explodam no treinamento)
    # OBS: ela não é chamada nesse arquivo, mas sim InstaceNorm, que normaliza os valores de cada imagem individualmente em vez de um mini-batch (grupo de imagens)
    # Isso vem de alguma implementação genérica do CycleGAN?
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################

# Bloco residual da ResNet. Serve para criar um atalho que permite à rede "lembrar" das informações anteriores e levar elas em consideração
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),                      # Expande as bordas reflexivamente para que a informação delas não se perca
            nn.Conv2d(in_features, in_features, 3),     # Mantém o mesmo número de canais da entrada
            nn.InstanceNorm2d(in_features),             # Recebe os canais e normaliza, para focar mais nas formas, texturas e estruturas do que em valores absolutos como brilho
            nn.ReLU(inplace=True),                      # ReLU converte os valores negativos em 0, para eliminar a linearidade e permitir aprender transformações mais complexas
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)        # Conceito residual: saída = x + F(x) (no gridnet, isso se aplica preservando os elementos da imagem original (x) e ajustar o visual (F(x)))


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]       # Canais = 3 ? (RGB)

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),   # Aumenta os canais de channels para 64
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling - reduz a resolução (stride=2 => pela metade) para cada kernel capturar uma área maior, e aumenta o número de canais, repete isso duas vezes
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]      # É aqui que a transformação A -> B de fato acontece

        # Upsampling - voltar a imagem ao tamanho original e reduzir os canais
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2, mode='bilinear'),   # modo bilinear de Upsampling é mais estável, pois evita padrões quadriculados
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]    # Volta para a quantidade original de canais (channels), nn.Tanh() restringe os valores para [-1, 1]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN) - divide a imagem em patches e analisa cada um separadamente
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
