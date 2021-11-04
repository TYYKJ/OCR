import torchvision


class MobileNetV2Encoder(torchvision.models.MobileNetV2):

    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self.out_channels = out_channels[2:]
        self._in_channels = 3
        del self.classifier

    def get_stages(self):
        return [
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth):
            x = stages[i](x)
            features.append(x)
        features = features[1:]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.1.bias", None)
        state_dict.pop("classifier.1.weight", None)
        super().load_state_dict(state_dict, **kwargs)


mobilenet_encoders = {
    "mobilenet_v2": {
        "encoder": MobileNetV2Encoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "out_channels": (3, 16, 24, 32, 96, 1280),
        },
    },
}
