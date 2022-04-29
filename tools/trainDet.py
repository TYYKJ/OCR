from ocr import DBTrainer

m = DBTrainer(
    project_name='DBNet-b16',
    train_data_path='/home/cat/Documents/icdar2015-ok/detection/train.json',
    val_data_path='/home/cat/Documents/icdar2015-ok/detection/test.json',
    checkpoint_save_path='../weights',
    encoder_name='se_resnet50',
    weights='imagenet',
    batch_size=16,
    num_workers=16,
    optimizer_name='sgd',
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    # resume_path='../weights/DB-se_resnet50-epoch=31-hmean=0.43-recall=0.36-precision=0.56.ckpt'
)
m.build_trainer(
    gpus=[1],
    epochs=300,
)
