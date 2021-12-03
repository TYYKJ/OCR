from ocr.rec import CRNNTrainer

m = CRNNTrainer(
    encoder_name='resnet50vd',
    image_path='/home/cat/Documents/icdar2017rctw/icdar2017/recognition/train',
    train_label_path='/home/cat/Documents/icdar2017rctw/icdar2017/recognition/train.txt',
    val_label_path='/home/cat/Documents/icdar2017rctw/icdar2017/recognition/val.txt',
    checkpoint_save_path='../weights',
    classes=3316 + 1,
    alphabet_path='/home/cat/Documents/icdar2017rctw/icdar2017/recognition/dict.txt',
    input_h=32,
    mean=0.5,
    std=0.5,
    batch_size=8,
    num_workers=8,
    optimizer_name='sgd',
    lr=0.001,
)

m.build_trainer(
    gpus=[0],
    max_epochs=300,
)
