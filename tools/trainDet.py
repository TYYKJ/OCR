from ocr import DBTrainer

m = DBTrainer(
    train_data_path='/media/cat/D/CCPD2019/train.json',
    val_data_path='/media/cat/D/CCPD2019/val.json',
    checkpoint_save_path='../weights',
    encoder_name='se_resnet50',
    weights='imagenet',
    batch_size=16,
    num_workers=16,
    optimizer_name='sgd',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.,
    # resume_path='../weights/DB-se_resnet50-epoch=31-hmean=0.43-recall=0.36-precision=0.56.ckpt'
)
m.build_trainer(
    gpus=[1],
    max_epochs=300,
)
