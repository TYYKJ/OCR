from ocr import ClassifyTrainer

m = ClassifyTrainer(
    batch_size=16,
    num_workers=16,
    train_root='/home/cat/Documents/data/train',
    val_root='/home/cat/Documents/data/val',
    model_name='resnet18',
    classes_num=2,
    checkpoint_save_path='/home/cat/PycharmProjects/OCR/weights'
)
m.build_trainer(gpus=[0], max_epochs=100)
