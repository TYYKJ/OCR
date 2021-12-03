from ocr import ClassifyTrainer

m = ClassifyTrainer(
    batch_size=16,
    num_workers=16,
    train_root='/home/cat/Documents/data/train',
    val_root='/home/cat/Documents/data/val',
    model_name='resnet18',
    classes_num=2,
    checkpoint_save_path='/home/cat/PycharmProjects/OCR/weights',
    optimizer_name='adam',
    lr=0.001,
    weight_decay=1e-4
)
m.build_trainer(gpus=[1], max_epochs=1)
