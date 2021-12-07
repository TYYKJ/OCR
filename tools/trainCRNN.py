from ocr import CRNNTrainer

m = CRNNTrainer(
    encoder_name='resnet50vd',
    image_path='/home/cat/Documents/all',
    train_label_path='/home/cat/Documents/all/train-no-space.txt',
    val_label_path='/home/cat/Documents/all/val-no-space.txt',
    checkpoint_save_path='../weights',
    classes=62 + 1,
    alphabet_path='/home/cat/Documents/all/dict.txt',
    input_h=32,
    mean=0.46369634,
    std=0.144033,
    batch_size=8,
    num_workers=16,
    optimizer_name='sgd',
    lr=0.001,
    weight_decay=1e-8,
    momentum=0.9
)

m.build_trainer(
    gpus=[0],
    max_epochs=1,
)
